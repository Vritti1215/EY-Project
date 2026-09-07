"""
Export Service Module
=====================
Handles exporting chat history and AI-generated content to various
formats including PDF, DOCX, and TXT files for user download.
"""

import os
import io
import re
import base64
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _inline_markdown_to_html(text):
    """
    Convert the small subset of markdown the AI actually produces
    (**bold**, *italic*, `code`) into the limited HTML-like tags
    reportlab's Paragraph understands, after escaping XML special chars.
    Anything else is left as plain text rather than guessed at.
    """
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`([^`\n]+?)`', r'<font face="Courier">\1</font>', text)
    return text


def _get_chart_image(msg):
    """Read the chart_image field off either a ChatHistory object or a dict."""
    if hasattr(msg, 'chart_image'):
        return getattr(msg, 'chart_image', None)
    if isinstance(msg, dict):
        return msg.get('chart_image')
    return None


def _response_to_pdf_flowables(response_text, chart_image_data_uri, ai_style, styles):
    """
    Turn one AI response (markdown-ish text, possibly containing a
    pipe table and/or an embedded "![alt](attachment://...)" chart
    reference) into real reportlab flowables instead of dumping the raw
    markdown as literal text: plain paragraphs with real bold/italic,
    an actual reportlab Table for any markdown table, and the real
    chart image (decoded from the message's stored chart_image base64
    data) instead of the broken-looking markdown image link.
    """
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch

    flowables = []
    lines = (response_text or '').replace('\r\n', '\n').split('\n')
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            text = ' '.join(paragraph_buffer).strip()
            if text:
                flowables.append(Paragraph(_inline_markdown_to_html(text), ai_style))
                flowables.append(Spacer(1, 4))
            paragraph_buffer.clear()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # A markdown image reference ("![Revenue Trend](attachment://x.png)")
        # is not real content once we embed the actual chart below — drop
        # the broken-looking raw markdown line entirely instead of
        # printing it as literal text.
        if re.match(r'^!\[.*?\]\([^)]*\)$', stripped):
            flush_paragraph()
            i += 1
            continue

        # Markdown table: a "|...|" header row followed by a "|---|---|"
        # separator row.
        if stripped.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|?$', lines[i + 1].strip()):
            flush_paragraph()
            header = [c.strip() for c in stripped.strip('|').split('|')]
            table_rows = [header]
            i += 2  # skip the "|---|---|" separator line
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                # Pad/truncate ragged rows so reportlab doesn't choke on
                # a row with a different column count than the header.
                if len(row) < len(header):
                    row += [''] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[:len(header)]
                table_rows.append(row)
                i += 1
            cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=8, leading=10)
            wrapped_rows = [
                [Paragraph(_inline_markdown_to_html(c), cell_style) for c in row]
                for row in table_rows
            ]
            table = Table(wrapped_rows, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            flowables.append(table)
            flowables.append(Spacer(1, 8))
            continue

        if stripped == '':
            flush_paragraph()
            i += 1
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()

    if chart_image_data_uri and isinstance(chart_image_data_uri, str) and ',' in chart_image_data_uri:
        try:
            _, b64data = chart_image_data_uri.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            img = RLImage(io.BytesIO(img_bytes))
            # Fit to page width, preserving aspect ratio.
            max_width = 6.0 * inch
            if img.imageWidth > max_width:
                scale = max_width / float(img.imageWidth)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
            flowables.append(Spacer(1, 6))
            flowables.append(img)
            flowables.append(Spacer(1, 10))
        except Exception:
            logger.warning("Could not embed chart image in PDF export", exc_info=True)

    return flowables


def export_chat_to_txt(messages, conversation_title="Chat Export"):
    """
    Export chat messages to a plain text format.

    Args:
        messages: List of ChatHistory objects or dicts with prompt/response.
        conversation_title: Title for the exported document.

    Returns:
        io.BytesIO: Buffer containing the TXT file content.
    """
    buffer = io.BytesIO()
    content = f"{'=' * 60}\n"
    content += f"  AI Assistant Portal — Chat Export\n"
    content += f"  Conversation: {conversation_title}\n"
    content += f"  Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += f"{'=' * 60}\n\n"

    for msg in messages:
        if hasattr(msg, 'prompt'):
            prompt = msg.prompt
            response = msg.response
            timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else 'N/A'
        else:
            prompt = msg.get('prompt', '')
            response = msg.get('response', '')
            timestamp = msg.get('timestamp', 'N/A')

        content += f"[{timestamp}] USER:\n{prompt}\n\n"
        content += f"[{timestamp}] AI ASSISTANT:\n{response}\n\n"
        content += f"{'-' * 40}\n\n"

    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer


def _response_to_docx_blocks(doc, response_text, chart_image_data_uri):
    """
    Same idea as _response_to_pdf_flowables but writing directly into a
    python-docx Document: real bold runs instead of literal '**text**',
    an actual docx table for a markdown table, the real chart image
    instead of the broken "![...](attachment://...)" markdown line.
    """
    from docx.shared import Pt, Inches

    def add_markdown_paragraph(text):
        p = doc.add_paragraph()
        # Split on **bold** segments, alternating plain/bold runs.
        parts = re.split(r'\*\*(.+?)\*\*', text)
        for idx, part in enumerate(parts):
            if not part:
                continue
            run = p.add_run(part)
            run.font.size = Pt(10)
            run.font.bold = (idx % 2 == 1)
        p.style.font.size = Pt(10)
        return p

    lines = (response_text or '').replace('\r\n', '\n').split('\n')
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            text = ' '.join(paragraph_buffer).strip()
            if text:
                add_markdown_paragraph(text)
            paragraph_buffer.clear()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if re.match(r'^!\[.*?\]\([^)]*\)$', stripped):
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|?$', lines[i + 1].strip()):
            flush_paragraph()
            header = [c.strip() for c in stripped.strip('|').split('|')]
            table_rows = [header]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if len(row) < len(header):
                    row += [''] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[:len(header)]
                table_rows.append(row)
                i += 1
            table = doc.add_table(rows=len(table_rows), cols=len(header))
            table.style = 'Light Grid Accent 1'
            for r, row in enumerate(table_rows):
                for c, val in enumerate(row):
                    cell = table.cell(r, c)
                    cell.text = val
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(9)
                            if r == 0:
                                run.font.bold = True
            doc.add_paragraph()
            continue

        if stripped == '':
            flush_paragraph()
            i += 1
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()

    if chart_image_data_uri and isinstance(chart_image_data_uri, str) and ',' in chart_image_data_uri:
        try:
            _, b64data = chart_image_data_uri.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            doc.add_picture(io.BytesIO(img_bytes), width=Inches(6))
        except Exception:
            logger.warning("Could not embed chart image in DOCX export", exc_info=True)


def export_chat_to_docx(messages, conversation_title="Chat Export"):
    """
    Export chat messages to a DOCX document format.

    Args:
        messages: List of ChatHistory objects or dicts.
        conversation_title: Title for the exported document.

    Returns:
        io.BytesIO: Buffer containing the DOCX file content.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

        doc = Document()

        # Title
        title = doc.add_heading('AI Assistant Portal — Chat Export', level=0)
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        # Subtitle
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run = subtitle.add_run(f'Conversation: {conversation_title}')
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(100, 100, 100)

        # Export date
        date_para = doc.add_paragraph()
        date_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run = date_para.add_run(f'Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(150, 150, 150)

        doc.add_paragraph()  # Spacer

        # Messages
        for msg in messages:
            if hasattr(msg, 'prompt'):
                prompt = msg.prompt
                response = msg.response
                timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else 'N/A'
            else:
                prompt = msg.get('prompt', '')
                response = msg.get('response', '')
                timestamp = msg.get('timestamp', 'N/A')

            # User message
            user_heading = doc.add_paragraph()
            run = user_heading.add_run(f'👤 User  [{timestamp}]')
            run.font.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(59, 130, 246)

            user_msg = doc.add_paragraph(prompt)
            user_msg.style.font.size = Pt(10)

            # AI response
            ai_heading = doc.add_paragraph()
            run = ai_heading.add_run(f'🤖 AI Assistant  [{timestamp}]')
            run.font.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(16, 185, 129)

            _response_to_docx_blocks(doc, response, _get_chart_image(msg))

            doc.add_paragraph('—' * 40)

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    except ImportError:
        logger.error("python-docx not installed for DOCX export")
        return export_chat_to_txt(messages, conversation_title)


def export_chat_to_pdf(messages, conversation_title="Chat Export"):
    """
    Export chat messages to a PDF document format.

    Args:
        messages: List of ChatHistory objects or dicts.
        conversation_title: Title for the exported document.

    Returns:
        io.BytesIO: Buffer containing the PDF file content.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=HexColor('#1e293b'),
            spaceAfter=12
        )

        user_style = ParagraphStyle(
            'UserMsg',
            parent=styles['Normal'],
            fontSize=10,
            textColor=HexColor('#1e40af'),
            leftIndent=20,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )

        ai_style = ParagraphStyle(
            'AIMsg',
            parent=styles['Normal'],
            fontSize=10,
            textColor=HexColor('#065f46'),
            leftIndent=20,
            spaceAfter=6
        )

        label_style = ParagraphStyle(
            'Label',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#6b7280'),
            spaceAfter=4
        )

        story = []

        # Title
        story.append(Paragraph('AI Assistant Portal — Chat Export', title_style))
        story.append(Paragraph(
            f'Conversation: {conversation_title} | Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            label_style
        ))
        story.append(Spacer(1, 20))

        # Messages
        for msg in messages:
            if hasattr(msg, 'prompt'):
                prompt = msg.prompt
                response = msg.response
                timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') if msg.timestamp else 'N/A'
            else:
                prompt = msg.get('prompt', '')
                response = msg.get('response', '')
                timestamp = msg.get('timestamp', 'N/A')

            chart_image_data_uri = _get_chart_image(msg)

            # Escape XML special characters for reportlab (the prompt is
            # plain user text, no markdown formatting to convert).
            prompt = (prompt or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            story.append(Paragraph(f'User [{timestamp}]:', label_style))
            story.append(Paragraph(prompt, user_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f'AI Assistant [{timestamp}]:', label_style))
            # Render the response as real paragraphs/tables/image instead
            # of dumping raw markdown (bold markers, pipe tables, and a
            # broken "![...](attachment://...)" image reference) as
            # literal text.
            story.extend(_response_to_pdf_flowables(response, chart_image_data_uri, ai_style, styles))
            story.append(Spacer(1, 16))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except ImportError:
        logger.error("reportlab not installed for PDF export")
        return export_chat_to_txt(messages, conversation_title)