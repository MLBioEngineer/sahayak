"""
Sahayak backend entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routers import chat, pdf, ecg

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="Sahayak API", version="0.1.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Graceful Bengali message when client hits rate limits."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "reply": "আপনি খুব দ্রুত অনেকগুলো অনুরোধ পাঠিয়েছেন। অনুগ্রহ করে এক মিনিট অপেক্ষা করে আবার চেষ্টা করুন।"
        }
    )


# CORS Configuration: allow localhost + any configured frontend domain (e.g. Vercel)
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(pdf.router, prefix="/export-pdf", tags=["pdf"])
app.include_router(ecg.router, prefix="/api/ecg", tags=["ecg"])


@app.get("/")
def root():
    """Root route providing API status and documentation links."""
    return {
        "name": "সহায়ক (Sahayak) Medical AI API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
            "pdf_export": "/export-pdf"
        },
        "disclaimer": "⚠️ এটি পেশাদার চিকিৎসা পরামর্শের বিকল্প নয়।"
    }


@app.get("/health")
def health():
    """Simple check to confirm the API is alive."""
    return {"status": "ok", "service": "sahayak-backend"}
