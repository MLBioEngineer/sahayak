"""
Unit tests for ESP32 live stream buffering, timeout handling, and chunk continuity.
Run directly with python.
"""
import sys
import os
import time

# Add parent directory to path so routers and services are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routers.ecg import (
    receive_hardware_stream,
    get_latest_stream,
    reset_stream_buffer,
    StreamChunkRequest,
    STREAM_BUFFER,
    STREAM_TIMEOUT_SECONDS,
)
import routers.ecg as ecg_router

def run_all_tests():
    print("=== RUNNING STREAM CONTINUITY & TIMEOUT TESTS ===")
    
    # Test 1: Cold start (hardware not connected)
    reset_stream_buffer()
    res = get_latest_stream()
    assert res["is_active"] is False, f"Expected is_active=False, got {res['is_active']}"
    assert res["samples_count"] == 0, f"Expected samples_count=0, got {res['samples_count']}"
    assert res["samples"] == [], f"Expected samples=[], got {res['samples']}"
    assert len(STREAM_BUFFER) == 0
    print("[PASS] Test 1: Cold start returns clean standby")

    # Test 2: First chunk received from hardware
    chunk1 = StreamChunkRequest(device_id="ESP32_ICU_1", samples=[100.0, 105.0, 110.0, 115.0])
    rec1 = receive_hardware_stream(chunk1)
    assert rec1["status"] == "received"
    assert rec1["buffer_size"] == 4
    assert list(STREAM_BUFFER) == [100.0, 105.0, 110.0, 115.0]
    
    latest1 = get_latest_stream()
    assert latest1["is_active"] is True
    assert latest1["device_id"] == "ESP32_ICU_1"
    assert latest1["samples_count"] == 4
    print("[PASS] Test 2: Active hardware packet accepted and reported active")

    # Test 3: Second chunk with partial overlap (prevents duplicate slices / stitch jumps)
    chunk2 = StreamChunkRequest(device_id="ESP32_ICU_1", samples=[110.0, 115.0, 120.0, 125.0])
    rec2 = receive_hardware_stream(chunk2)
    assert rec2["buffer_size"] == 6 # only 120.0 and 125.0 appended!
    assert list(STREAM_BUFFER) == [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
    print("[PASS] Test 3: Overlapping chunk slices deduplicated smoothly without stitch jump")

    # Test 4: Exact duplicate packet (e.g. Wi-Fi retry)
    chunk_dup = StreamChunkRequest(device_id="ESP32_ICU_1", samples=[120.0, 125.0])
    rec_dup = receive_hardware_stream(chunk_dup)
    assert rec_dup["buffer_size"] == 6 # unchanged
    assert list(STREAM_BUFFER) == [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
    print("[PASS] Test 4: Exact duplicate retry packets ignored cleanly")

    # Test 5: Inactivity timeout (> 3.5s elapsed)
    # Simulate hardware unplugged / stopped posting
    ecg_router.LAST_STREAM_TIME = time.time() - (STREAM_TIMEOUT_SECONDS + 0.5)
    
    latest_timed_out = get_latest_stream()
    assert latest_timed_out["is_active"] is False, f"Expected False, got {latest_timed_out['is_active']}"
    assert latest_timed_out["samples_count"] == 0, f"Expected 0, got {latest_timed_out['samples_count']}"
    assert latest_timed_out["samples"] == [], f"Expected [], got {latest_timed_out['samples']}"
    assert len(STREAM_BUFFER) == 0, f"Buffer must be cleared on timeout, got {len(STREAM_BUFFER)}"
    print("[PASS] Test 5: Inactivity timeout (> 3.5s) switches is_active=False and clears buffer")

    # Test 6: Reconnection after disconnect gap starts fresh (no stitch artifact)
    chunk_new = StreamChunkRequest(device_id="ESP32_ICU_1", samples=[200.0, 205.0, 210.0])
    rec_new = receive_hardware_stream(chunk_new)
    assert rec_new["buffer_size"] == 3
    assert list(STREAM_BUFFER) == [200.0, 205.0, 210.0]
    print("[PASS] Test 6: Reconnection after gap starts a fresh clean contiguous stream")

    # Test 7: Buffer reset endpoint
    reset_res = reset_stream_buffer()
    assert reset_res["status"] == "cleared"
    assert reset_res["buffer_size"] == 0
    assert ecg_router.LAST_STREAM_TIME == 0.0
    latest_after_reset = get_latest_stream()
    assert latest_after_reset["is_active"] is False
    assert latest_after_reset["samples"] == []
    print("[PASS] Test 7: Reset endpoint clears buffer and stream timestamp")

    # Test 8: Empty sample batch must be ignored and must NOT revive active stream
    empty_chunk = StreamChunkRequest(device_id="ESP32_ICU_1", samples=[])
    rec_empty = receive_hardware_stream(empty_chunk)
    assert rec_empty["status"] == "ignored"
    assert rec_empty["samples_received"] == 0
    latest_after_empty = get_latest_stream()
    assert latest_after_empty["is_active"] is False
    assert latest_after_empty["samples"] == []
    print("[PASS] Test 8: Empty sample batch ignored and does not falsely activate stream")

    # Test 9: Buffer empty check: If buffer is empty, is_active is ALWAYS False even if LAST_STREAM_TIME is fresh
    ecg_router.LAST_STREAM_TIME = time.time() # fake recent timestamp
    STREAM_BUFFER.clear()
    latest_empty_buf = get_latest_stream()
    assert latest_empty_buf["is_active"] is False
    assert latest_empty_buf["samples_count"] == 0
    assert latest_empty_buf["samples"] == []
    print("[PASS] Test 9: Empty buffer is strictly reported as inactive standby")

    # Test 10: Single-point baseline match is not falsely dropped as an overlap
    # Feed initial 3 samples ending in 512.0
    receive_hardware_stream(StreamChunkRequest(device_id="ESP32_ICU_1", samples=[500.0, 505.0, 512.0]))
    assert list(STREAM_BUFFER) == [500.0, 505.0, 512.0]
    # New chunk starts with 512.0 (same baseline level), followed by 518.0 and 524.0
    # Because k=1 is a single point and k < len(incoming)=3, it must NOT drop 512.0!
    receive_hardware_stream(StreamChunkRequest(device_id="ESP32_ICU_1", samples=[512.0, 518.0, 524.0]))
    assert list(STREAM_BUFFER) == [500.0, 505.0, 512.0, 512.0, 518.0, 524.0]
    print("[PASS] Test 10: Single-point baseline coincidence is safely preserved without false truncation")

    print("\nALL 10 STREAM CONTINUITY AND TIMEOUT TESTS PASSED!")

if __name__ == "__main__":
    run_all_tests()
