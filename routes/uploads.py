"""
Upload Routes
=============
Handles file upload, text extraction, AI summarization,
document Q&A, and upload history management.

CHANGELOG (this revision):
  - ask_about_file() previously sent every question straight to
    ai_service.answer_document_question(), which truncates to the first
    3000 characters of an already-truncated (200-row/15000-char) text
    preview. For CSV/XLSX/XLS uploads, this gave wrong answers on any
    question touching data outside that small window.
    Now: tabular files (CSV/XLSX/XLS) are answered using the SAME
    deterministic engine (services.insight_service) that /chat/send uses
    — exact unique-value lists, exact counts, and exact aggregates
    computed over the FULL dataframe, with the AI only used to phrase a
    computed result or (as a last resort) answer from a "Pre-computed
    figures" context block it's told to quote rather than recalculate.
    Non-tabular files (PDF/DOCX/TXT) are unchanged — no structured engine
    exists for free text, so those still go through
    ai_service.answer_document_question().
"""

import os
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from extensions import db
from models.uploads import UploadedFile
from models.analytics import AnalyticsLog
from services.file_service import allowed_file, generate_stored_filename, extract_text, get_file_size_display
from services.ai_service import ai_service
from services.insight_service import (
    load_dataframe, find_column, detect_group_column,
    monthly_totals, quarterly_totals, category_counts, category_aggregate,
    requested_aggregation, build_ai_context, is_count_query, is_unique_value_query,
    find_column_for_question, find_time_column_by_value, exact_category_count,
    unique_column_values, extract_time_from_question,
    find_numeric_column_generic, find_date_column_generic, find_group_column_generic,
    MONTH_COLUMN_ALIASES, VALUE_COLUMN_ALIASES,
)

uploads_bp = Blueprint('uploads', __name__)

TABULAR_TYPES = ('csv', 'xlsx', 'xls')


@uploads_bp.route('/uploads')
@login_required
def uploads_page():
    """Render the file upload and analysis page."""
    files = UploadedFile.query.filter_by(
        user_id=current_user.id
    ).order_by(UploadedFile.upload_date.desc()).all()

    return render_template('uploads.html', files=files)


@uploads_bp.route('/uploads/upload', methods=['POST'])
@login_required
def upload_file():
    """
    Handle file upload, extract text, and generate AI summary.
    Supports PDF, DOCX, TXT, and CSV files up to 16MB.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: PDF, DOCX, TXT, CSV'}), 400

    try:
        # Secure and generate unique filename
        original_filename = secure_filename(file.filename)
        stored_filename = generate_stored_filename(original_filename)
        file_type = original_filename.rsplit('.', 1)[1].lower()

        # Ensure upload directory exists
        upload_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)

        # Save file
        filepath = os.path.join(upload_dir, stored_filename)
        file.save(filepath)

        # Get file size
        file_size = os.path.getsize(filepath)

        # Extract text
        extracted_text = extract_text(filepath, file_type)

        # Generate AI summary
        summary = ''
        if extracted_text and len(extracted_text) > 50:
            summary = ai_service.summarize_document(extracted_text)

        # Save to database
        upload_record = UploadedFile(
            user_id=current_user.id,
            filename=original_filename,
            stored_filename=stored_filename,
            file_type=file_type,
            file_size=file_size,
            extracted_text=extracted_text,
            summary=summary
        )
        db.session.add(upload_record)

        # Log activity
        log = AnalyticsLog(
            user_id=current_user.id,
            activity_type='upload',
            details=f'Uploaded {original_filename}'
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'success': True,
            'file_id': upload_record.id,
            'filename': original_filename,
            'file_type': file_type,
            'file_size': get_file_size_display(file_size),
            'summary': summary,
            'text_preview': extracted_text[:500] if extracted_text else 'No text extracted'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


def _answer_tabular_question(upload, question):
    """
    Answer a question about a tabular (CSV/XLSX/XLS) upload using the same
    deterministic engine /chat/send uses, computed over the FULL file on
    disk rather than a truncated text preview.

    Returns:
        (answer: str, error: str or None). If error is set, answer is
        empty and the caller should return an error response.
    """
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], upload.stored_filename)
    if not os.path.exists(filepath):
        return '', 'Original file no longer available on the server. Please re-upload it.'

    df = load_dataframe(filepath, upload.file_type)
    if df is None:
        return '', 'Could not parse this file.'

    result_response = None
    computed_context = None

    # Unique-value / distinct-list questions.
    if is_unique_value_query(question):
        unique_col = find_column_for_question(df.columns, question)
        if not unique_col and any(w in question.lower() for w in
                                   ('timing', 'timings', 'time slot', 'test time')):
            time_candidates = [c for c in df.columns if 'time' in str(c).lower()]
            if time_candidates:
                unique_col = time_candidates[0]
        if unique_col:
            values = unique_column_values(df, unique_col)
            if values:
                rows = "\n".join(f"- **{v}**" for v in values)
                result_response = (
                    f"**Distinct values in `{unique_col}`:**\n\n{rows}\n\n"
                    f"Found **{len(values)} unique value(s)** by checking the "
                    f"complete uploaded file."
                )

    # Exact count questions (e.g. "how many students at 9 AM").
    if result_response is None and is_count_query(question):
        question_col = find_column_for_question(df.columns, question)
        requested_time = extract_time_from_question(question)
        if not question_col and requested_time:
            question_col = find_time_column_by_value(df, requested_time)
        if question_col and requested_time:
            label, count, _ = exact_category_count(df, question_col, requested_time, time_value=True)
            result_response = (
                f"**Count for {label}: {count}**\n\n"
                f"Counted directly from the complete column **{question_col}**."
            )

    # Categorical / aggregate / time-series fallbacks — build a
    # pre-computed context block for the AI to quote rather than
    # recalculate, mirroring /chat/send's logic exactly.
    if result_response is None:
        fiscal_start = current_app.config.get('FISCAL_YEAR_START_MONTH', 4)
        # Alias-based detection first, then generic schema-based fallback
        # so this works on any spreadsheet regardless of column naming
        # (see services/schema_service.py).
        month_col = find_column(df.columns, MONTH_COLUMN_ALIASES) \
            or find_date_column_generic(df, question)
        value_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
            or find_numeric_column_generic(df, question, exclude=month_col)
        group_col = detect_group_column(df.columns, question) \
            or find_group_column_generic(df, question, exclude={month_col, value_col})

        message_lower = question.lower()
        time_series_request = any(k in message_lower for k in (
            'sales', 'revenue', 'income', 'profit', 'expense', 'amount',
            'financial', 'month', 'monthly', 'quarter', 'q1', 'q2', 'q3',
            'q4', 'qtr', 'date', 'period'
        ))

        if group_col and not time_series_request:
            operation = requested_aggregation(question)
            if operation == 'count':
                labels, values = category_counts(df, group_col)
            else:
                numeric_col = value_col or find_numeric_column_generic(df, question, exclude=group_col)
                if not numeric_col or numeric_col == group_col:
                    labels, values = category_counts(df, group_col)
                else:
                    labels, values = category_aggregate(df, group_col, numeric_col, operation)
            if labels:
                computed_context = build_ai_context(categories=(labels, values), category_label=group_col)

        elif month_col and value_col:
            monthly = monthly_totals(df, month_col, value_col)
            quarterly = quarterly_totals(df, month_col, value_col, fiscal_start)
            if monthly:
                computed_context = build_ai_context(
                    monthly=monthly, quarterly=quarterly, fiscal_start_month=fiscal_start
                )

        elif group_col:
            operation = requested_aggregation(question)
            if operation == 'count':
                labels, values = category_counts(df, group_col)
            else:
                numeric_col = value_col or find_numeric_column_generic(df, question, exclude=group_col)
                if not numeric_col or numeric_col == group_col:
                    labels, values = category_counts(df, group_col)
                else:
                    labels, values = category_aggregate(df, group_col, numeric_col, operation)
            if labels:
                computed_context = build_ai_context(categories=(labels, values), category_label=group_col)

    if result_response is not None:
        return result_response, None

    prompt = question
    if computed_context:
        prompt = f"{question}\n\n{computed_context}"

    system_prompt = (
        "You are a data analyst assistant. If a 'Pre-computed figures' block "
        "is provided below, treat those numbers as authoritative — quote "
        "them directly and never recalculate, estimate, or contradict them. "
        "Be precise and cite specific rows/columns/categories from the data. "
        "If no pre-computed figures are given and you cannot answer "
        "confidently, say so clearly rather than guessing."
    )
    result = ai_service.get_chat_response(prompt, system_prompt=system_prompt)
    return result.get('response', 'Unable to answer the question.'), None


@uploads_bp.route('/uploads/ask/<int:file_id>', methods=['POST'])
@login_required
def ask_about_file(file_id):
    """
    Answer a question about a specific uploaded file using AI.

    Tabular files (CSV/XLSX/XLS) are answered deterministically over the
    full dataset via services.insight_service — see _answer_tabular_question
    above. Non-tabular files (PDF/DOCX/TXT) use the AI-over-extracted-text
    path since there's no structured engine for free text.

    Args:
        file_id: ID of the uploaded file to query about.
    """
    upload = UploadedFile.query.filter_by(
        id=file_id, user_id=current_user.id
    ).first()

    if not upload:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json()
    question = data.get('question', '').strip() if data else ''

    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    if upload.file_type in TABULAR_TYPES:
        answer, error = _answer_tabular_question(upload, question)
        if error:
            return jsonify({'error': error}), 404 if 'not available' in error else 400
        return jsonify({'success': True, 'answer': answer, 'filename': upload.filename})

    if not upload.extracted_text:
        return jsonify({'error': 'No text content available for this file'}), 400

    answer = ai_service.answer_document_question(upload.extracted_text, question)

    return jsonify({
        'success': True,
        'answer': answer,
        'filename': upload.filename
    })


@uploads_bp.route('/uploads/delete/<int:file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    """Delete an uploaded file from both storage and database."""
    upload = UploadedFile.query.filter_by(
        id=file_id, user_id=current_user.id
    ).first()

    if not upload:
        return jsonify({'error': 'File not found'}), 404

    # Delete physical file
    try:
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], upload.stored_filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass  # Continue even if file deletion fails

    db.session.delete(upload)

    # Log activity — previously missing; every other action in this app
    # (register, login, logout, upload) logs to AnalyticsLog, but delete
    # did not, leaving a gap in activity tracking.
    log = AnalyticsLog(
        user_id=current_user.id,
        activity_type='delete',
        details=f'Deleted {upload.filename}'
    )
    db.session.add(log)

    db.session.commit()

    return jsonify({'success': True, 'message': 'File deleted'})


@uploads_bp.route('/uploads/details/<int:file_id>')
@login_required
def file_details(file_id):
    """Get detailed information about an uploaded file."""
    upload = UploadedFile.query.filter_by(
        id=file_id, user_id=current_user.id
    ).first()

    if not upload:
        return jsonify({'error': 'File not found'}), 404

    return jsonify({
        'success': True,
        'file': {
            'id': upload.id,
            'filename': upload.filename,
            'file_type': upload.file_type,
            'file_size': get_file_size_display(upload.file_size),
            'upload_date': upload.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'text_preview': upload.extracted_text[:1000] if upload.extracted_text else '',
            'summary': upload.summary
        }
    })
