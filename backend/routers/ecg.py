from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.ecg_service import predict_epoch, analyze_session, generate_benchmark_signal

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
    """
    signals = generate_benchmark_signal(patient_type)
    ecg_sig = signals["ecg_signal"]
    edr_sig = signals["edr_signal"]
    
    prediction = predict_epoch(ecg_sig)
    return {
        "patient_type": patient_type,
        "sample_rate_hz": 100,
        "duration_seconds": 30,
        "samples_count": len(ecg_sig),
        "ecg_signal": ecg_sig,
        "edr_signal": edr_sig,
        "ground_truth": "Apnea" if patient_type == "apnea" else "Normal",
        "model_prediction": prediction
    }

import time
from collections import deque

# In-memory buffer for real-time ESP32 telemetry (holds 10 seconds = 1000 samples @ 100 Hz for ultra-fast reaction)
STREAM_BUFFER = deque(maxlen=1000)
LAST_STREAM_TIME = 0.0
LAST_DEVICE_ID = "ESP32_NODE"

@router.post("/stream")
def receive_hardware_stream(data: StreamChunkRequest):
    """
    Endpoint for ESP32 hardware streaming over Wi-Fi.
    Accepts real-time batches of ECG samples and buffers them in memory.
    """
    global LAST_STREAM_TIME, LAST_DEVICE_ID
    if data.samples:
        STREAM_BUFFER.extend(data.samples)
    LAST_STREAM_TIME = time.time()
    LAST_DEVICE_ID = data.device_id
    return {
        "status": "received",
        "device_id": data.device_id,
        "samples_received": len(data.samples),
        "buffer_size": len(STREAM_BUFFER),
        "message": "Telemetry received successfully"
    }

@router.get("/stream/latest")
def get_latest_stream():
    """
    Endpoint for frontend to fetch latest buffered telemetry from ESP32.
    Includes active status, window samples, and live model prediction.
    """
    current_time = time.time()
    # Device is considered active if a chunk was received within last 12 seconds
    is_active = (current_time - LAST_STREAM_TIME < 12.0) if LAST_STREAM_TIME > 0 else False
    samples_list = list(STREAM_BUFFER)
    
    prediction = None
    if len(samples_list) >= 100:
        try:
            # Predict using latest live window (1000 samples = 10 seconds @ 100 Hz)
            window = samples_list[-1000:]
            prediction = predict_epoch(window)
        except Exception as e:
            prediction = None

    return {
        "is_active": is_active,
        "device_id": LAST_DEVICE_ID if LAST_STREAM_TIME > 0 else None,
        "last_received_seconds_ago": round(current_time - LAST_STREAM_TIME, 1) if LAST_STREAM_TIME > 0 else None,
        "samples_count": len(samples_list),
        "samples": samples_list,
        "prediction": prediction
    }

@router.post("/stream/reset")
def reset_stream_buffer():
    """Resets the live streaming buffer."""
    STREAM_BUFFER.clear()
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


