from fastapi import APIRouter, Request
from models.schemas import ChatRequest, ChatResponse
from services.ai_service import get_ai_response
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

def get_rate_limit(request: Request) -> str:
    """Guest users get strict limits, signed in users get higher limits."""
    if request.headers.get("Authorization"):
        return "60/minute"
    return "3/minute"

@router.post("", response_model=ChatResponse)
@limiter.limit(get_rate_limit)
def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint. Calls self-hosted Ollama AI service with dynamic rate-limiting.
    Guest users are heavily restricted to encourage sign-up.
    """
    history_as_dicts = [m.model_dump() for m in body.history]
    reply = get_ai_response(body.message, history_as_dicts)
    return ChatResponse(reply=reply)
