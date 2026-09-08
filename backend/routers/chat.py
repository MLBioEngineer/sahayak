from fastapi import APIRouter

from models.schemas import ChatRequest, ChatResponse
from services.ai_service import get_ai_response

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Currently calls the mock model in ai_service.py.
    Swap ai_service.get_ai_response() to point at your trained model later —
    this route doesn't need to change.
    """
    history_as_dicts = [m.model_dump() for m in request.history]
    reply = get_ai_response(request.message, history_as_dicts)
    return ChatResponse(reply=reply)
