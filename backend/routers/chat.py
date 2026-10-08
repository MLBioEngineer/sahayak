from fastapi import APIRouter, Request
from models.schemas import ChatRequest, ChatResponse
from services.ai_service import get_ai_response
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

@router.post("", response_model=ChatResponse)
@limiter.limit("15/minute")
def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint. Calls self-hosted Ollama AI service with rate-limiting
    and graceful error handling.
    """
    history_as_dicts = [m.model_dump() for m in body.history]
    reply = get_ai_response(body.message, history_as_dicts)
    return ChatResponse(reply=reply)
