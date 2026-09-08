from fastapi import APIRouter
from fastapi.responses import Response

from models.schemas import PDFExportRequest
from services.pdf_service import build_chat_pdf

router = APIRouter()


@router.post("")
def export_pdf(request: PDFExportRequest):
    """Converts the given chat history into a downloadable PDF."""
    history_as_dicts = [m.model_dump() for m in request.history]
    pdf_bytes = build_chat_pdf(request.title, history_as_dicts)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=sahayak-chat.pdf"},
    )
