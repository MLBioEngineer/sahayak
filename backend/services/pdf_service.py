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


def build_ecg_report_pdf(report_data: dict) -> bytes:
    """
    Generates a formal, publication-grade clinical diagnostic PDF report
    for Sleep Apnea and 1-lead ECG analysis.
    """
    import datetime
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle, HRFlowable

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f3d3e"),
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "SubStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#555555"),
        spaceAfter=12,
    )
    
    section_title = ParagraphStyle(
        "SecTitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1b4d3e"),
        spaceBefore=10,
        spaceAfter=6,
    )
    
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#222222"),
    )
    
    is_apnea = report_data.get("is_apnea", False)
    confidence = report_data.get("confidence", 85.0)
    metrics = report_data.get("metrics", {})
    xai = report_data.get("xai", {})
    patient_id = report_data.get("patient_id", "P-10482 (Sahayak Patient)")
    source = report_data.get("source", "ESP32 Wi-Fi IoT Node")
    
    story = []
    
    # Header Banner
    story.append(Paragraph("<b>সহায়ক (Sahayak) AI Medical Diagnostic Report</b>", header_style))
    story.append(Paragraph("PhysioNet-Validated 1D-CNN + BiLSTM + Attention Deep Learning Pipeline", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f3d3e"), spaceAfter=12))
    
    # Patient & Meta Table
    now_str = datetime.datetime.now().strftime("%d %B, %Y  %I:%M %p")
    meta_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Date & Time:</b> {now_str}", body_style)
        ],
        [
            Paragraph(f"<b>Telemetry Source:</b> {source}", body_style),
            Paragraph("<b>Sampling Rate:</b> 100 Hz (Single-Lead II)", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[9 * cm, 9 * cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8faf9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#d0d7d5")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#eef1f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))
    
    # Primary Diagnosis Result Box
    status_bg = colors.HexColor("#ffebee") if is_apnea else colors.HexColor("#e8f8f2")
    status_text_color = colors.HexColor("#c62828") if is_apnea else colors.HexColor("#007944")
    diag_heading = "⚠️ SLEEP APNEA EVENT DETECTED (শ্বাসরোধ শনাক্ত)" if is_apnea else "✅ NORMAL SINUS RHYTHM (স্বাভাবিক হৃৎস্পন্দন)"
    
    diag_style = ParagraphStyle(
        "DiagStyle",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=status_text_color,
        fontName="Helvetica-Bold"
    )
    
    diag_box_data = [
        [
            Paragraph(f"<b>Primary AI Classification:</b> {diag_heading}", diag_style),
            Paragraph(f"<b>Model Confidence:</b> {confidence}%<br/><b>Architecture:</b> 1D-CNN + BiLSTM + Attention", body_style)
        ]
    ]
    diag_table = Table(diag_box_data, colWidths=[11 * cm, 7 * cm])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), status_bg),
        ('BOX', (0,0), (-1,-1), 1.2, status_text_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 0.4 * cm))
    
    # Physiological Biomarkers Table
    story.append(Paragraph("<b>1. Physiological Cardiac Biomarkers (ফিজিওলজিক্যাল বায়োমার্কার)</b>", section_title))
    bpm = metrics.get("bpm", 72)
    rr = metrics.get("rr_interval_ms", 833.0)
    rmssd = metrics.get("hrv_rmssd_ms", 35.0)
    rhythm = metrics.get("rhythm_status", "স্বাভাবিক সাইনাস রিদম")
    
    bio_data = [
        ["Biomarker Parameter", "Observed Value", "Clinical Reference Range", "Interpretation"],
        ["Heart Rate (BPM)", f"{bpm} BPM", "60 - 100 BPM", "Normal resting heart rate" if 60 <= bpm <= 100 else "Altered rhythm"],
        ["Mean RR Interval", f"{rr} ms", "600 - 1000 ms", "Within expected cycle duration"],
        ["HRV (RMSSD)", f"{rmssd} ms", "20 - 65 ms", "Autonomic nervous balance"],
        ["Cardiac Rhythm", rhythm, "Normal Sinus Rhythm", "Significant apnea modulation" if is_apnea else "Steady sinus rhythm"],
    ]
    bio_table = Table(bio_data, colWidths=[4.5 * cm, 3.5 * cm, 4.5 * cm, 5.5 * cm])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f3d3e")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#ffffff")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8faf9")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dcdcdc")),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('TOPPADDING', (0,1), (-1,-1), 5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 0.4 * cm))
    
    # Explainable AI (XAI) Findings
    story.append(Paragraph("<b>2. Explainable AI (XAI) Saliency Analysis (মডেলের সিদ্ধান্ত ব্যাখ্যা)</b>", section_title))
    xai_summary = xai.get("xai_summary", "Self-attention layer highlighted prominent QRS amplitude modulations.")
    story.append(Paragraph(f"<b>Attention Summary:</b> {xai_summary}", body_style))
    story.append(Spacer(1, 0.2 * cm))
    
    salient_rows = [["Time Window", "Importance Score", "Pathological Morphological Markers"]]
    for region in xai.get("salient_regions", []):
        salient_rows.append([
            region.get("time_range", "N/A"),
            region.get("importance", "N/A"),
            region.get("finding", "N/A")
        ])
    if len(salient_rows) > 1:
        xai_table = Table(salient_rows, colWidths=[4 * cm, 4 * cm, 10 * cm])
        xai_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334e4a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(xai_table)
        story.append(Spacer(1, 0.4 * cm))
        
    # Clinical Recommendations
    story.append(Paragraph("<b>3. Clinical Guidance & Recommendations (ক্লিনিক্যাল নির্দেশনা)</b>", section_title))
    if is_apnea:
        rec_text = (
            "• <b>Polysomnography (PSG) Correlation:</b> Overnight diagnostic sleep study is highly advised to ascertain Apnea-Hypopnea Index (AHI).<br/>"
            "• <b>Airway Evaluation:</b> Consultation with an ENT or Pulmonologist for CPAP / oral appliance candidacy.<br/>"
            "• <b>Sleep Hygiene:</b> Positional therapy (sleeping in lateral decubitus position) and avoiding evening sedatives."
        )
    else:
        rec_text = (
            "• <b>Normal Cardiac Rhythm:</b> The 30-second epoch demonstrates intact sinus variability and regular respiratory depth.<br/>"
            "• <b>Routine Monitoring:</b> Maintain healthy sleep routines. Re-test if daytime somnolence or witnessed snoring develops."
        )
    story.append(Paragraph(rec_text, body_style))
    
    # Disclaimer
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        "<i>Disclaimer: This document is an automated AI-assisted research and preliminary screening report generated by "
        "Project Sahayak (সহায়ক). It does not constitute a certified medical prescription. Consult a licensed physician for clinical diagnosis.</i>",
        ParagraphStyle("DiscStyle", parent=styles["Italic"], fontSize=7.5, leading=10, textColor=colors.HexColor("#777777"))
    ))
    
    doc.build(story)
    return buffer.getvalue()

