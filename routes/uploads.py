"""
Upload Routes
=============
Handles file upload, text extraction, AI summarization,
document Q&A, and upload history management.
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

uploads_bp = Blueprint('uploads', __name__)


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


@uploads_bp.route('/uploads/ask/<int:file_id>', methods=['POST'])
@login_required
def ask_about_file(file_id):
    """
    Answer a question about a specific uploaded file using AI.

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
