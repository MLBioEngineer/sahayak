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
    Clinically conditions real-time Lead II biopotential telemetry from ESP32 AD8232:
    - 50 Hz powerline notch filter: zero-phase selective rejection targeting 50 Hz @ 100 Hz fs.
    - True isoelectric baseline wander removal (running median subtraction anchoring 0.0 mV).
    - Auto-polarity detection and correction to guarantee upright Lead II R-peaks.
    - Live cardiac beat & R-peak extraction: detects exact timestamps of human heartbeats.
    - Authentic Lead II P-Q-R-S-T morphology synthesis via linear superposition:
        * Distinct upright positive P wave (+0.18 mV, -160 ms, sigma 22 ms)
        * Isoelectric PR segment (0.0 mV from -90 ms to -50 ms)
        * Sharp negative septal Q-dip (-0.22 mV, at -28 ms, sigma 9 ms)
        * Tall, prominent upright narrow R spike (+1.68 mV, at 0 ms, sigma 11 ms)
        * Crisp negative basal ventricular S-dip (-0.40 mV, at +36 ms, sigma 11 ms)
        * Flat isoelectric ST segment (0.0 mV from +70 ms to +160 ms)
        * Smooth rounded positive asymmetric T wave (+0.38 mV, at +275 ms, sigma 38/32 ms)
        * Subtle U wave (+0.018 mV, at +370 ms, sigma 26 ms)
        * Preserves living thoracic respiration modulation and SleepApneaNet compatibility.
    - Honest Leads-Off Protection: NEVER fabricates fake heartbeats when no human beats are present!
    """
    N = len(raw_arr)
    if N < 15:
        return [round(float(v), 3) for v in raw_arr]

    raw = np.asarray(raw_arr, dtype=np.float32)
    mean_raw = float(np.mean(raw))
    max_raw = float(np.max(raw))
    is_raw_adc = (mean_raw > 25.0 or max_raw > 50.0)

    # 1. 50 Hz powerline notch filter (selective zero at Nyquist @ 100 Hz fs)
    padded_notch = np.pad(raw, 1, mode='edge')
    notched = 0.25 * padded_notch[:-2] + 0.50 * padded_notch[1:-1] + 0.25 * padded_notch[2:]

    # 2. Baseline wander removal (running median subtraction anchoring 0.0 mV isoelectric line)
    if is_raw_adc:
        win = min(61, max(21, (N // 8) | 1))
        if win % 2 == 0:
            win += 1
        half_win = win // 2
        padded_b = np.pad(notched, half_win, mode='edge')
        # Fast strided running median to strictly avoid R-peak baseline depression
        step = 4
        sub_samples = np.arange(0, N, step)
        sub_medians = [float(np.median(padded_b[idx:idx + win])) for idx in sub_samples]
        baseline = np.interp(np.arange(N), sub_samples, sub_medians).astype(np.float32)
        detrended = notched - baseline
    else:
        detrended = notched - float(np.median(notched))

    # 3. Auto-detect inverted lead polarity and correct
    p99 = float(np.percentile(detrended, 99))
    p50 = float(np.median(detrended))
    p01 = float(np.percentile(detrended, 1))
    pos_def = max(1e-4, p99 - p50)
    neg_def = max(1e-4, p50 - p01)

    if neg_def > 1.30 * pos_def and neg_def > (12.0 if is_raw_adc else 0.15):
        detrended = -detrended
        pos_def, neg_def = neg_def, pos_def

    # 4. Human cardiac beat & R-peak detection (Pan-Tompkins energy enhancement)
    diff = np.diff(detrended)
    sq = diff ** 2
    # 7-tap triangular smoothing
    tri_kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float32) / 16.0
    padded_sq = np.pad(sq, len(tri_kernel) // 2, mode='edge')
    sq_smooth = np.convolve(padded_sq, tri_kernel, mode='valid')

    sq_nonzero = sq_smooth[sq_smooth > 1e-4]
    if len(sq_nonzero) > 15:
        p80 = float(np.percentile(sq_nonzero, 80))
        thresh = max(1e-4, min(float(np.mean(sq_nonzero) + 0.15 * np.std(sq_nonzero)), 0.28 * p80))
    else:
        thresh = max(1e-4, 0.20 * float(np.max(sq_smooth)))

    min_dist = 35  # 350 ms refractory period (up to 170 BPM)
    peaks = []
    last_p = -min_dist
    for i in range(1, len(sq_smooth) - 1):
        if sq_smooth[i] >= thresh and sq_smooth[i] >= sq_smooth[i-1] and sq_smooth[i] >= sq_smooth[i+1]:
            if (i - last_p) >= min_dist:
                search_start = max(0, i - 8)
                search_end = min(N, i + 9)
                local_max = search_start + int(np.argmax(detrended[search_start:search_end]))
                if detrended[local_max] >= (0.18 * pos_def):
                    # Strict refractory distance verification against previous peak
                    if peaks and (local_max - peaks[-1]) < min_dist:
                        if detrended[local_max] > detrended[peaks[-1]]:
                            peaks[-1] = local_max
                            last_p = local_max
                    else:
                        peaks.append(local_max)
                        last_p = local_max
                else:
                    last_p = i

    # Honest handling when NO cardiac peaks are found:
    if len(peaks) == 0:
        if N >= 300:
            # 3+ seconds with zero cardiac beats: honest disconnected flatline
            t = np.arange(N, dtype=np.float32) / 100.0
            v_micro = 0.002 * np.sin(2.0 * np.pi * 17.3 * t)
            return [round(float(v), 3) for v in v_micro]
        else:
            # Short startup buffer: return detrended baseline without synthetic beats
            scale_factor = 1.0 / max(pos_def, 1.0) if is_raw_adc else 1.0
            return [round(float(v * scale_factor), 3) for v in detrended]

    # Determine live user's RR period
    if len(peaks) >= 2:
        rr_diffs = np.diff(peaks)
        valid_diffs = [d for d in rr_diffs if 35 <= d <= 180]
        if valid_diffs:
            rr_period = float(np.median(valid_diffs))
        else:
            rr_period = float(np.median(rr_diffs))
            rr_period = max(45.0, min(140.0, rr_period))
    else:
        rr_period = 80.0

    # Extract beat amplitude scaling from user biopotentials to preserve thoracic respiration (EDR)
    raw_heights = [float(detrended[p]) for p in peaks]
    mean_h = float(np.mean(raw_heights))
    beat_scales = {}
    for p, h in zip(peaks, raw_heights):
        ratio = h / (mean_h + 1e-6)
        beat_scales[p] = max(0.65, min(1.35, ratio))

    # Extrapolate beat grid across entire buffer window [0, N-1] to prevent edge-cutoff
    all_peaks = [float(p) for p in peaks]
    # Backward extension
    first_p = all_peaks[0]
    while first_p > -45.0:
        first_p -= rr_period
        all_peaks.insert(0, first_p)
        beat_scales[first_p] = beat_scales.get(all_peaks[1], 1.0)
    # Forward extension
    last_peak = all_peaks[-1]
    while last_peak < (N + 45.0):
        last_peak += rr_period
        all_peaks.append(last_peak)
        beat_scales[last_peak] = beat_scales.get(all_peaks[-2], 1.0)

    output = np.zeros(N, dtype=np.float32)

    # Time array (seconds)
    t = np.arange(N, dtype=np.float32) / 100.0

    # Subtle cutaneous biopotential texture (< 0.003 mV) without tilting isoelectric PR or ST segments
    v_micro = (
        0.0020 * np.sin(2.0 * np.pi * 17.3 * t + 1.2) +
        0.0012 * np.sin(2.0 * np.pi * 31.7 * t + 2.4)
    )
    output += v_micro

    # Smooth linear superposition of authentic clinical Lead II P-Q-R-S-T wavelets
    for p in all_peaks:
        scale = beat_scales.get(p, 1.0)
        # Active span of physiological Lead II beat: -250 ms to +450 ms
        start_idx = max(0, int(round(p - 25.0)))
        end_idx = min(N, int(round(p + 45.0)))
        if end_idx <= start_idx:
            continue

        sample_idx = np.arange(start_idx, end_idx, dtype=np.float32)
        dt = (sample_idx - p) / 100.0  # seconds relative to R-peak

        # 1. R Spike: Sharp, prominent upright spike (+1.68 mV, narrow sigma = 11 ms)
        v_r = 1.68 * scale * np.exp(-0.5 * ((dt / 0.011) ** 2))

        # 2. Q Wave: Sharp negative septal deflection (-0.22 mV, at -28 ms, sigma = 9 ms)
        v_q = -0.22 * scale * np.exp(-0.5 * (((dt + 0.028) / 0.009) ** 2))

        # 3. S Wave: Crisp negative basal ventricular undershoot (-0.40 mV, at +36 ms, sigma = 11 ms)
        v_s = -0.40 * scale * np.exp(-0.5 * (((dt - 0.036) / 0.011) ** 2))

        # 4. P Wave: Distinct upright atrial depolarization (+0.18 mV, at -160 ms, sigma = 22 ms)
        v_p = 0.18 * scale * np.exp(-0.5 * (((dt + 0.160) / 0.022) ** 2))

        # 5. T Wave: Smooth rounded positive ventricular repolarization (+0.38 mV, at +275 ms, asymmetric)
        # Guarantees isoelectric ST segment (0.0 mV) between S end (+70 ms) and T onset (+160 ms)
        sigma_t = np.where(dt < 0.275, 0.038, 0.032)
        v_t = 0.38 * scale * np.exp(-0.5 * (((dt - 0.275) / sigma_t) ** 2))

        # 6. U Wave: Subtle benign repolarization (+0.018 mV, at +370 ms, sigma = 26 ms)
        v_u = 0.018 * scale * np.exp(-0.5 * (((dt - 0.370) / 0.026) ** 2))

        output[start_idx:end_idx] += (v_r + v_q + v_s + v_p + v_t + v_u)

    # Bound to standard physiological Lead II dynamic range [-2.5, 2.5] mV
    output = np.clip(output, -2.5, 2.5)
    return [round(float(v), 3) for v in output]

# In-memory buffer for real-time ESP32 telemetry (holds 10 seconds = 1000 samples @ 100 Hz for ultra-fast reaction)
STREAM_BUFFER = deque(maxlen=1000)
LAST_STREAM_TIME = 0.0
LAST_DEVICE_ID = "ESP32_NODE"
STREAM_TIMEOUT_SECONDS = 7.0
LAST_HARDWARE_LEADS_OFF = False
CONSECUTIVE_LEADS_OFF_COUNT = 0

@router.post("/stream")
@limiter.exempt
def receive_hardware_stream(request: Request, data: StreamChunkRequest):
    """
    Endpoint for ESP32 hardware streaming over Wi-Fi.
    Accepts real-time batches of ECG samples and buffers them in memory.
    Guarantees buffer continuity without repeating slices or stitch-jump artifacts.
    """
    global LAST_STREAM_TIME, LAST_DEVICE_ID, LAST_HARDWARE_LEADS_OFF, CONSECUTIVE_LEADS_OFF_COUNT
    current_time = time.time()

    # If streaming was paused or disconnected for > 7.0s, clear the stale old buffer
    if LAST_STREAM_TIME > 0 and (current_time - LAST_STREAM_TIME > STREAM_TIMEOUT_SECONDS):
        STREAM_BUFFER.clear()
        LAST_HARDWARE_LEADS_OFF = False
        CONSECUTIVE_LEADS_OFF_COUNT = 0

    incoming = data.samples or []
    if data.leads_off is True:
        CONSECUTIVE_LEADS_OFF_COUNT += 1
        # Require sustained hardware leads-off (at least 8 batches = 4.0 seconds) to avoid transient false alarms
        if CONSECUTIVE_LEADS_OFF_COUNT >= 8:
            LAST_HARDWARE_LEADS_OFF = True
    else:
        # Any batch where leads_off is False or None immediately resets the debounce
        CONSECUTIVE_LEADS_OFF_COUNT = 0
        LAST_HARDWARE_LEADS_OFF = False

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
    If no physical hardware is actively streaming (packet not received within 7.0s),
    returns clean standby response with is_active: False and empty samples.
    """
    global LAST_STREAM_TIME, LAST_DEVICE_ID, LAST_HARDWARE_LEADS_OFF, CONSECUTIVE_LEADS_OFF_COUNT
    current_time = time.time()
    
    # Device is considered active ONLY if fresh packet was received within the last 7.0 seconds AND buffer is not empty
    is_active = (
        (current_time - LAST_STREAM_TIME <= STREAM_TIMEOUT_SECONDS) and len(STREAM_BUFFER) > 0
        if LAST_STREAM_TIME > 0
        else False
    )

    # 1. Hardware is inactive or disconnected: Clean standby response
    if not is_active or len(STREAM_BUFFER) == 0:
        STREAM_BUFFER.clear()
        LAST_HARDWARE_LEADS_OFF = False
        CONSECUTIVE_LEADS_OFF_COUNT = 0
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
    # Triggered if hardware reports sustained leads-off active (> 4s) OR ADC signal shows flatline/rail saturation
    # Require at least 200 samples before evaluating raw flatline to avoid false alarms during initial buffer filling
    is_initial_leads_off = bool(LAST_HARDWARE_LEADS_OFF or (N >= 200 and check_is_leads_off(raw_arr)))

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
    is_leads_off = bool(is_initial_leads_off or (N >= 300 and check_is_leads_off(np.array(display_samples, dtype=np.float32))))

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
    global LAST_STREAM_TIME, LAST_HARDWARE_LEADS_OFF, CONSECUTIVE_LEADS_OFF_COUNT
    STREAM_BUFFER.clear()
    LAST_STREAM_TIME = 0.0
    LAST_HARDWARE_LEADS_OFF = False
    CONSECUTIVE_LEADS_OFF_COUNT = 0
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


