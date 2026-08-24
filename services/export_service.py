"""
Export Service Module
=====================
Handles exporting chat history and AI-generated content to various
formats including PDF, DOCX, and TXT files for user download.
"""

import os
import io
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


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

            ai_msg = doc.add_paragraph(response)
            ai_msg.style.font.size = Pt(10)

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

            # Escape XML special characters for reportlab
            prompt = prompt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            response = response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            story.append(Paragraph(f'User [{timestamp}]:', label_style))
            story.append(Paragraph(prompt, user_style))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f'AI Assistant [{timestamp}]:', label_style))
            story.append(Paragraph(response, ai_style))
            story.append(Spacer(1, 16))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except ImportError:
        logger.error("reportlab not installed for PDF export")
        return export_chat_to_txt(messages, conversation_title)
