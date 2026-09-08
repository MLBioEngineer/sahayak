from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str


class PDFExportRequest(BaseModel):
    history: list[ChatMessage]
    title: str = "Sahayak Chat Summary"
