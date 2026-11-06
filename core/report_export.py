"""
Turns a schedule + metrics + the LLM's narrative report into a downloadable
PDF (the narrative report, rendered with its Markdown headings/bold/bullets).
"""

import os
import re
import io

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ReportLab's built-in fonts (Helvetica etc.) use WinAnsi encoding, which has no
# Rupee glyph (U+20B9). Register a real Unicode TTF if one is available on this
# machine so currency symbols render correctly; otherwise fall back to "Rs." text.
UNICODE_FONT = None
for _candidate in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
    if os.path.exists(_candidate):
        try:
            pdfmetrics.registerFont(TTFont("UnicodeBody", _candidate))
            UNICODE_FONT = "UnicodeBody"
        except Exception:
            pass
        break


def _safe_currency_text(text: str) -> str:
    return text if UNICODE_FONT else text.replace("\u20b9", "Rs. ")


def _markdown_to_flowables(text: str, styles) -> list:
    flowables = []
    bullet_buffer = []

    def flush_bullets():
        for item in bullet_buffer:
            flowables.append(Paragraph(f"&bull;&nbsp;&nbsp;{item}", styles["BulletItem"]))
        bullet_buffer.clear()

    for raw_line in _safe_currency_text(text).split("\n"):
        line = raw_line.strip()
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        if not line:
            flush_bullets()
            flowables.append(Spacer(1, 6))
        elif line.startswith("## "):
            flush_bullets()
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("# "):
            flush_bullets()
            flowables.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.startswith("- "):
            bullet_buffer.append(line[2:])
        else:
            flush_bullets()
            flowables.append(Paragraph(line, styles["BodyText"]))

    flush_bullets()
    return flowables


def export_to_pdf(schedule: pd.DataFrame, metrics: dict, report_text: str, title: str = "Production Plan Report") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BulletItem", parent=styles["BodyText"], leftIndent=14))
    if UNICODE_FONT:
        for style_name in ("Title", "Heading1", "Heading2", "BodyText", "BulletItem"):
            styles[style_name].fontName = UNICODE_FONT

    flowables = [Paragraph(_safe_currency_text(title), styles["Title"]), Spacer(1, 12)]

    metrics_table = Table(
        [["Metric", "Value"]] + [[k, _safe_currency_text(str(v))] for k, v in metrics.items()],
        colWidths=[8 * cm, 8 * cm],
    )
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, -1), UNICODE_FONT or "Helvetica"),
    ]))
    flowables.append(metrics_table)
    flowables.append(Spacer(1, 16))

    flowables.extend(_markdown_to_flowables(report_text, styles))

    doc.build(flowables)
    return buffer.getvalue()
