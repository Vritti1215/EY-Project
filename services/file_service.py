"""
File Service Module
===================
Handles file upload processing, text extraction from various document formats
(PDF, DOCX, TXT, CSV, XLSX, XLS), and file validation logic.
"""

import os
import uuid
import logging
import csv
import io

logger = logging.getLogger(__name__)

# Allowed file extensions (including Excel)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_stored_filename(original_filename):
    """Generate a unique filename for server-side storage to prevent collisions."""
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'txt'
    unique_name = f"{uuid.uuid4().hex}_{original_filename}"
    return unique_name


def extract_text(filepath, file_type):
    """
    Extract text content from an uploaded file based on its type.

    Args:
        filepath: Absolute path to the uploaded file.
        file_type: Extension of the file (pdf, docx, txt, csv, xlsx, xls).

    Returns:
        str: Extracted text content from the file.
    """
    try:
        if file_type == 'pdf':
            return _extract_pdf(filepath)
        elif file_type == 'docx':
            return _extract_docx(filepath)
        elif file_type == 'txt':
            return _extract_txt(filepath)
        elif file_type == 'csv':
            return _extract_csv(filepath)
        elif file_type in ('xlsx', 'xls'):
            return _extract_excel(filepath)
        else:
            return "Unsupported file type."
    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {e}")
        return f"Error extracting text: {str(e)}"


def _extract_pdf(filepath):
    """Extract text from a PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return '\n\n'.join(text_parts) if text_parts else "No readable text found in PDF."
    except ImportError:
        return "PDF processing library not available."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def _extract_docx(filepath):
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n\n'.join(paragraphs) if paragraphs else "No readable text found in document."
    except ImportError:
        return "DOCX processing library not available."
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


def _extract_txt(filepath):
    """Extract text from a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading text file: {str(e)}"


def _extract_csv(filepath):
    """Extract and format text from a CSV file with full data for AI analysis."""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        total_rows = len(df)
        summary = f"CSV File: {total_rows} rows x {len(df.columns)} columns\n\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += "Column Data Types:\n"
        for col in df.columns:
            summary += f"  - {col}: {df[col].dtype}\n"
        summary += "\n"

        # Include statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary += "Numeric Summary:\n"
            summary += df[numeric_cols].describe().to_string()
            summary += "\n\n"

        # Include full data up to 200 rows (for AI context)
        preview_rows = min(total_rows, 200)
        summary += f"Data ({preview_rows} of {total_rows} rows):\n"
        summary += df.head(preview_rows).to_string(index=False)

        # Truncate if too large for AI context (keep under 15000 chars)
        if len(summary) > 15000:
            summary = summary[:15000] + f"\n\n... [Truncated. Full file has {total_rows} rows]"

        return summary
    except ImportError:
        # Fallback without pandas
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i > 100:
                        break
                    rows.append(', '.join(row))
                return '\n'.join(rows)
        except Exception as e:
            return f"Error reading CSV: {str(e)}"
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def _extract_excel(filepath):
    """
    Extract and format text from an Excel file (.xlsx/.xls).
    Reads all sheets and includes column headers, data types, statistics,
    and full data preview for AI analysis.
    """
    try:
        import pandas as pd

        # Read all sheets
        xl = pd.ExcelFile(filepath)
        sheet_names = xl.sheet_names

        summary = f"Excel File: {len(sheet_names)} sheet(s)\n"
        summary += f"Sheets: {', '.join(sheet_names)}\n\n"

        for sheet_name in sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            total_rows = len(df)

            summary += f"{'='*60}\n"
            summary += f"Sheet: {sheet_name} ({total_rows} rows x {len(df.columns)} columns)\n"
            summary += f"{'='*60}\n\n"
            summary += f"Columns: {', '.join(df.columns.astype(str).tolist())}\n\n"

            # Column data types
            summary += "Column Data Types:\n"
            for col in df.columns:
                summary += f"  - {col}: {df[col].dtype}\n"
            summary += "\n"

            # Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary += "Numeric Summary:\n"
                summary += df[numeric_cols].describe().to_string()
                summary += "\n\n"

            # Include data preview (up to 200 rows per sheet)
            preview_rows = min(total_rows, 200)
            summary += f"Data ({preview_rows} of {total_rows} rows):\n"
            summary += df.head(preview_rows).to_string(index=False)
            summary += "\n\n"

        # Truncate if too large for AI context
        if len(summary) > 15000:
            summary = summary[:15000] + f"\n\n... [Truncated. File has {len(sheet_names)} sheet(s)]"

        return summary

    except ImportError:
        return "Excel processing library (openpyxl/pandas) not available. Install with: pip install openpyxl pandas"
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"


def get_file_size_display(size_bytes):
    """Convert file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
