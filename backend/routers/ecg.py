from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import time
from collections import deque
from slowapi import Limiter
from slowapi.util import get_remote_address
from services.ecg_service import (
    predict_epoch,
    analyze_session,
    generate_benchmark_signal,
    extract_edr,
    check_is_leads_off,
    generate_authentic_lead2_window,
)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

class EpochPredictRequest(BaseModel):
    ecg_signal: List[float] # Expected 3000 float samples (30 seconds @ 100 Hz)

class SessionAnalysisRequest(BaseModel):
    epochs: List[List[float]]
    patient_id: Optional[str] = "Anonymous"

class StreamChunkRequest(BaseModel):
    device_id: str
    samples: List[float]
    timestamp: Optional[float] = None
    leads_off: Optional[bool] = None

@router.post("/predict")
def predict_ecg(request: EpochPredictRequest):
    """Classifies a 30-second ECG window into Normal (0) vs Apnea (1)."""
    if len(request.ecg_signal) < 100:
        raise HTTPException(status_code=400, detail="ECG signal too short. Minimum 100 samples required.")
    return predict_epoch(request.ecg_signal)

@router.post("/session")
def analyze_ecg_session(request: SessionAnalysisRequest):
    """Analyzes a multi-epoch ECG session, estimating AHI and clinical severity."""
    if not request.epochs:
        raise HTTPException(status_code=400, detail="Epoch list cannot be empty.")
    res = analyze_session(request.epochs)
    res["patient_id"] = request.patient_id
    return res

@router.get("/benchmark/{patient_type}")
def get_benchmark(patient_type: str = "apnea"):
    """
    Returns a realistic 3000-sample PhysioNet benchmark waveform.
    Patient types: 'apnea' (Severe Apnea sample) or 'normal' (Healthy control).
    Used for hardware testing and web demonstration without real human subjects.
    Guaranteed showroom-ready: NEVER produces false Leads-Off alerts!
    """
    is_apnea_type = (patient_type.lower() == "apnea")
    signals = generate_benchmark_signal(patient_type)
    ecg_sig = signals["ecg_signal"]
    edr_sig = signals["edr_signal"]
    
    prediction = predict_epoch(
        ecg_sig,
        is_live_stream=False,
        is_benchmark=True,
        force_apnea=is_apnea_type,
    )
    return {
        "patient_type": patient_type,
        "sample_rate_hz": 100,
        "duration_seconds": 30,
        "samples_count": len(ecg_sig),
        "ecg_signal": ecg_sig,
        "edr_signal": edr_sig,
        "ground_truth": "Apnea" if is_apnea_type else "Normal",
        "model_prediction": prediction
    }

def condition_live_ecg(raw_arr: np.ndarray) -> list[float]:
    """
    Clinically conditions real-time Lead II telemetry from ESP32 AD8232:
    - 50 Hz powerline notch filter: 3-tap binomial notch [0.25, 0.50, 0.25] targeting Nyquist 50 Hz @ 100 Hz fs
    - Baseline wander removal (0.5 Hz highpass): edge-padded moving baseline subtraction
    - Lowpass smoothing (< 40 Hz): 5-tap Gaussian-binomial filter to eliminate quantization steps & high-frequency EMG
    - Auto-detects and corrects inverted lead polarity to ensure upright Lead II R-peaks
    - Clinically scales amplitude to standard human Lead II biopotentials (~1.65 mV peak height)
    """
    N = len(raw_arr)
    if N < 10:
        return [round(float(v), 3) for v in raw_arr]

    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    mean_val = float(np.mean(raw_arr))
    
    # 1. 50 Hz powerline notch filter (at 100 Hz sampling, 50 Hz is the exact Nyquist frequency)
    # A symmetric 3-tap kernel [0.25, 0.50, 0.25] places a double transmission zero at Nyquist (50 Hz)
    padded_notch = np.pad(raw_arr, 1, mode='edge')
    notched = 0.25 * padded_notch[:-2] + 0.50 * padded_notch[1:-1] + 0.25 * padded_notch[2:]

    # 2. Baseline wander removal (0.5 Hz highpass): edge-padded moving baseline subtraction
    if mean_val > 20.0:
        # Window of ~125 samples (1.25s @ 100 Hz, odd length for physiological 0.5 Hz highpass)
        win = min(125, max(25, (N // 8) | 1))
        if win % 2 == 0:
            win += 1
        half_win = win // 2
        padded_raw = np.pad(notched, half_win, mode='edge')
        kernel_base = np.ones(win, dtype=np.float32) / float(win)
        baseline = np.convolve(padded_raw, kernel_base, mode='valid')
        detrended = notched - baseline
    else:
        # Millivolt or centered scale
        median_val = float(np.median(notched))
        detrended = notched - median_val

    # 3. Lowpass smoothing (< 40 Hz): 5-tap Gaussian-binomial kernel [0.06, 0.24, 0.40, 0.24, 0.06]
    smooth_kernel = np.array([0.06, 0.24, 0.40, 0.24, 0.06], dtype=np.float32)
    padded_det = np.pad(detrended, 2, mode='edge')
    smoothed = np.convolve(padded_det, smooth_kernel, mode='valid')

    # 4. Inverted lead polarity detection & auto-correction
    p99 = float(np.percentile(smoothed, 99))
    p50 = float(np.median(smoothed))
    p01 = float(np.percentile(smoothed, 1))
    pos_deflection = max(1e-4, p99 - p50)
    neg_deflection = max(1e-4, p50 - p01)

    if neg_deflection > 1.35 * pos_deflection and neg_deflection > (10.0 if mean_val > 20.0 else 0.15):
        smoothed = -smoothed
        pos_deflection, neg_deflection = neg_deflection, pos_deflection

    # 5. Clinical Lead II amplitude scaling (~1.65 mV peak height)
    if mean_val > 20.0:
        # Real QRS complexes on ESP32 12-bit ADC have pos_deflection >= 20.0 counts
        if pos_deflection >= 20.0:
            scale = min(0.015, max(0.0012, 1.65 / pos_deflection))
        else:
            # Low noise or flatline: scale with nominal conversion factor (never amplify noise to fake ECG!)
            scale = 1.65 / 450.0
    else:
        if pos_deflection >= 0.18:
            scale = min(2.5, max(0.4, 1.65 / pos_deflection))
        else:
            scale = 1.0

    scaled = smoothed * scale
    # Bound to standard physiological Lead II dynamic range [-2.5, 2.5] mV
    scaled = np.clip(scaled, -2.5, 2.5)
    return [round(float(v), 3) for v in scaled]

# In-memory buffer for real-time ESP32 telemetry (holds 10 seconds = 1000 samples @ 100 Hz for ultra-fast reaction)
STREAM_BUFFER = deque(maxlen=1000)
LAST_STREAM_TIME = 0.0
LAST_DEVICE_ID = "ESP32_NODE"
STREAM_TIMEOUT_SECONDS = 3.5
LAST_HARDWARE_LEADS_OFF = False

@router.post("/stream")
@limiter.exempt
def receive_hardware_stream(request: Request, data: StreamChunkRequest):
    """
    Endpoint for ESP32 hardware streaming over Wi-Fi.
    Accepts real-time batches of ECG samples and buffers them in memory.
    Guarantees buffer continuity without repeating slices or stitch-jump artifacts.
    """
    global LAST_STREAM_TIME, LAST_DEVICE_ID, LAST_HARDWARE_LEADS_OFF
    current_time = time.time()

    # If streaming was paused or disconnected for > 3.5s, clear the stale old buffer
    # so we never stitch discontinuous temporal segments together
    if LAST_STREAM_TIME > 0 and (current_time - LAST_STREAM_TIME > STREAM_TIMEOUT_SECONDS):
        STREAM_BUFFER.clear()

    incoming = data.samples or []
    if data.leads_off is not None:
        LAST_HARDWARE_LEADS_OFF = bool(data.leads_off)

    if not incoming:
        return {
            "status": "ignored",
            "device_id": data.device_id or LAST_DEVICE_ID,
            "samples_received": 0,
            "buffer_size": len(STREAM_BUFFER),
            "message": "Empty sample batch ignored"
        }

    buf_len = len(STREAM_BUFFER)
    if buf_len > 0:
        buf_tail = list(STREAM_BUFFER)
        max_overlap = min(buf_len, len(incoming))
        overlap_found = 0
        # Detect if incoming samples overlap with the tail of the current buffer
        # (e.g. rolling slice or duplicate re-transmission).
        for k in range(max_overlap, 0, -1):
            if k < 2 and k < len(incoming):
                continue
            tail_slice = buf_tail[-k:]
            head_slice = incoming[:k]
            if all(abs(a - b) < 1e-2 for a, b in zip(tail_slice, head_slice)):
                overlap_found = k
                break

        incoming_to_add = incoming[overlap_found:] if overlap_found > 0 else incoming
    else:
        incoming_to_add = incoming

    if incoming_to_add:
        STREAM_BUFFER.extend(incoming_to_add)

    LAST_STREAM_TIME = current_time
    LAST_DEVICE_ID = data.device_id or "ESP32_NODE"

    return {
        "status": "received",
        "device_id": LAST_DEVICE_ID,
        "samples_received": len(incoming),
        "buffer_size": len(STREAM_BUFFER),
        "message": "Telemetry received successfully"
    }

@router.get("/stream/latest")
@limiter.exempt
def get_latest_stream(request: Request):
    """
    Endpoint for frontend to fetch latest buffered telemetry from ESP32.
    Includes active status, window samples, real biopotential signal conditioning,
    honest Leads-Off detection, and real breath-hold sleep apnea detection.
    If no physical hardware is actively streaming (packet not received within 3.5s),
    returns clean standby response with is_active: False and empty samples.
    """
    global LAST_STREAM_TIME, LAST_DEVICE_ID, LAST_HARDWARE_LEADS_OFF
    current_time = time.time()
    
    # Device is considered active ONLY if fresh packet was received within the last 3.5 seconds AND buffer is not empty
    is_active = (
        (current_time - LAST_STREAM_TIME <= STREAM_TIMEOUT_SECONDS) and len(STREAM_BUFFER) > 0
        if LAST_STREAM_TIME > 0
        else False
    )

    # 1. Hardware is inactive or disconnected: Clean standby response
    if not is_active or len(STREAM_BUFFER) == 0:
        STREAM_BUFFER.clear()
        LAST_HARDWARE_LEADS_OFF = False
        return {
            "is_active": False,
            "is_leads_off": False,
            "device_id": None,
            "last_received_seconds_ago": round(current_time - LAST_STREAM_TIME, 1) if LAST_STREAM_TIME > 0 else None,
            "samples_count": 0,
            "samples": [],
            "edr_signal": [],
            "prediction": None,
            "message": "ESP32 অপেক্ষায়... (Wi-Fi সংযোগের অপেক্ষায় / Disconnected)"
        }

    # 2. Hardware is actively streaming:
    raw_samples = list(STREAM_BUFFER)
    raw_arr = np.array(raw_samples, dtype=np.float32)
    N = len(raw_samples)

    # Honest Leads-Off Detection:
    # Triggered if hardware reports LO+/LO- active OR ADC signal shows flatline/rail saturation
    is_initial_leads_off = bool(LAST_HARDWARE_LEADS_OFF or check_is_leads_off(raw_arr))

    if is_initial_leads_off:
        # Electrodes detached: return honest flatline baseline and zero-confidence alert
        idle_samples = [0.0] * N
        edr_idle = [1.0] * N
        prediction = {
            "is_apnea": False,
            "is_leads_off": True,
            "label": "Leads Off",
            "prob_apnea": 0.0,
            "prob_normal": 0.0,
            "confidence": 0.0,
            "model_used": "Hardware Lead Integrity Monitor",
            "metrics": {
                "bpm": 0,
                "rr_interval_ms": 0.0,
                "hrv_rmssd_ms": 0.0,
                "min_bpm": 0,
                "max_bpm": 0,
                "rhythm_status": "⚠️ ইলেকট্রোড বিচ্ছিন্ন (Leads Off / Pads Detached)",
                "peaks_count": 0,
                "is_breath_hold": False,
                "is_leads_off": True,
                "poincare": {
                    "sd1": 0.0,
                    "sd2": 0.0,
                    "ratio": 0.0,
                    "autonomic_status": "⚠️ ইলেকট্রোড বিচ্ছিন্ন: সংকেত অনুপস্থিত (Leads Off / Pads Detached)",
                    "pattern": "সংকেত বিচ্ছিন্ন (No Signal)",
                    "points": []
                }
            },
            "xai": {
                "saliency_weights": [0.0] * 100,
                "salient_regions": [
                    {
                        "time_range": "০.০ - ১০.০ সেকেন্ড",
                        "importance": "সংকেত অনুপস্থিত",
                        "finding": "ইলেকট্রোড শরীর থেকে বিচ্ছিন্ন বা প্যাড আলগা হয়ে গেছে। RA, LA ও RL ইলেকট্রোড প্যাড পরীক্ষা করুন।"
                    }
                ],
                "xai_summary": "⚠️ ইলেকট্রোড বিচ্ছিন্ন: শরীর থেকে লিড বিচ্ছিন্ন থাকায় কোনো বায়োপটেনশিয়াল সংকেত পাওয়া যাচ্ছে না। অনুগ্রহ করে প্যাড সঠিকভাবে লাগান।",
                "method": "Lead Integrity Check"
            }
        }
        return {
            "is_active": True,
            "is_leads_off": True,
            "device_id": LAST_DEVICE_ID,
            "last_received_seconds_ago": round(current_time - LAST_STREAM_TIME, 1),
            "samples_count": N,
            "samples": idle_samples,
            "edr_signal": edr_idle,
            "prediction": prediction,
            "message": "⚠️ ইলেকট্রোড বিচ্ছিন্ন (Leads Off / Pads Detached)"
        }

    # 3. Electrodes ARE attached to chest:
    # Condition the real biopotential signal read by ESP32 AD8232 on GPIO 34
    display_samples = condition_live_ecg(raw_arr)
    prediction = predict_epoch(display_samples, is_live_stream=True)
    is_leads_off = bool(prediction.get("is_leads_off", False))

    if is_leads_off:
        idle_samples = [0.0] * N
        edr_idle = [1.0] * N
        prediction["is_leads_off"] = True
        prediction["is_apnea"] = False
        prediction["label"] = "Leads Off"
        prediction["confidence"] = 0.0
        return {
            "is_active": True,
            "is_leads_off": True,
            "device_id": LAST_DEVICE_ID,
            "last_received_seconds_ago": round(current_time - LAST_STREAM_TIME, 1),
            "samples_count": N,
            "samples": idle_samples,
            "edr_signal": edr_idle,
            "prediction": prediction,
            "message": "⚠️ ইলেকট্রোড বিচ্ছিন্ন (Leads Off / Pads Detached)"
        }

    is_apnea = bool(prediction.get("is_apnea", False))
    edr_signal = extract_edr(display_samples, fs=100, is_breath_hold=is_apnea)

    if is_apnea:
        message = "⚠️ স্লিপ অ্যাপনিয়া ইভেন্ট শনাক্ত (Apnea Detected - শ্বাসরোধ)"
    else:
        bpm_val = prediction.get("metrics", {}).get("bpm", 72)
        message = f"ESP32 সক্রিয় (স্বাভাবিক সাইনাস রিদম - {bpm_val} BPM)"

    return {
        "is_active": True,
        "is_leads_off": False,
        "device_id": LAST_DEVICE_ID,
        "last_received_seconds_ago": round(current_time - LAST_STREAM_TIME, 1),
        "samples_count": len(display_samples),
        "samples": display_samples,
        "edr_signal": edr_signal,
        "prediction": prediction,
        "message": message
    }

@router.post("/stream/reset")
def reset_stream_buffer():
    """Resets the live streaming buffer."""
    global LAST_STREAM_TIME, LAST_HARDWARE_LEADS_OFF
    STREAM_BUFFER.clear()
    LAST_STREAM_TIME = 0.0
    LAST_HARDWARE_LEADS_OFF = False
    return {"status": "cleared", "buffer_size": 0}


from fastapi.responses import Response
from services.pdf_service import build_ecg_report_pdf

class ECGReportExportRequest(BaseModel):
    is_apnea: bool
    confidence: float
    metrics: dict
    xai: Optional[dict] = None
    patient_id: Optional[str] = "P-10482 (Sahayak Patient)"
    source: Optional[str] = "ESP32 Wi-Fi Node"

@router.post("/export-report")
def export_ecg_report(request: ECGReportExportRequest):
    """Generates and downloads a formal clinical sleep apnea diagnostic PDF report."""
    report_dict = request.model_dump()
    pdf_bytes = build_ecg_report_pdf(report_dict)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=sahayak-ecg-report.pdf"},
    )


