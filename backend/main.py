"""
Sahayak backend entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, pdf

app = FastAPI(title="Sahayak API", version="0.1.0")

# Allow the frontend (localhost dev + later your deployed domain) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your real frontend domain before public launch
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(pdf.router, prefix="/export-pdf", tags=["pdf"])


@app.get("/health")
def health():
    """Simple check to confirm the API is alive."""
    return {"status": "ok", "service": "sahayak-backend"}
