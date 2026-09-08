"""
Generates a downloadable PDF summary of a chat conversation.
Uses reportlab (pure Python, no system dependencies, free).
"""
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def build_chat_pdf(title: str, history: list[dict]) -> bytes:
    """
    Args:
        title: heading for the PDF
        history: list of {"role": "user"|"assistant", "content": str}

    Returns:
        Raw PDF bytes, ready to stream back to the client.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    user_style = ParagraphStyle("User", parent=styles["Normal"], spaceAfter=10, textColor="#254d40")
    bot_style = ParagraphStyle("Bot", parent=styles["Normal"], spaceAfter=14, textColor="#3c4a45")

    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.5 * cm)]

    for msg in history:
        label = "You" if msg["role"] == "user" else "Sahayak"
        style = user_style if msg["role"] == "user" else bot_style
        story.append(Paragraph(f"<b>{label}:</b> {msg['content']}", style))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "<i>This document is generated from an AI chat and is not a substitute "
        "for professional medical advice.</i>",
        styles["Italic"],
    ))

    doc.build(story)
    return buffer.getvalue()
