"""
Chatbot Routes
==============
Handles the AI chatbot interface, conversation management,
message sending/receiving, and chat history operations.
All endpoints require authentication.
"""

import os
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from extensions import db
from models.chat import ChatHistory, Conversation
from models.uploads import UploadedFile
from models.analytics import AnalyticsLog
from services.ai_service import ai_service
from services.export_service import export_chat_to_txt, export_chat_to_docx, export_chat_to_pdf
from services.file_service import allowed_file, extract_text, generate_stored_filename, get_file_size_display

chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route('/chat')
@login_required
def chat():
    """
    Render the main chatbot interface.
    Loads the user's conversation history for the sidebar.
    """
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(Conversation.updated_at.desc()).all()

    return render_template('chatbot.html', conversations=conversations)


@chatbot_bp.route('/chat/conversations')
@login_required
def list_conversations():
    """
    Return a JSON list of the user's conversations for the floating widget.
    Used by the widget's conversation history sidebar.
    """
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(Conversation.updated_at.desc()).limit(20).all()

    return jsonify({
        'success': True,
        'conversations': [{
            'id': c.id,
            'title': c.title,
            'updated_at': c.updated_at.strftime('%b %d') if c.updated_at else ''
        } for c in conversations]
    })


@chatbot_bp.route('/chat/new', methods=['POST'])
@login_required
def new_conversation():
    """Create a new conversation and return its ID."""
    conversation = Conversation(
        user_id=current_user.id,
        title='New Conversation'
    )
    db.session.add(conversation)
    db.session.commit()

    return jsonify({
        'success': True,
        'conversation_id': conversation.id,
        'title': conversation.title
    })


@chatbot_bp.route('/chat/send', methods=['POST'])
@login_required
def send_message():
    """
    Process a user message and return an AI response.
    Creates a new conversation if none is specified.
    Supports optional file uploads for analysis.
    """
    # Parse request — always try form data first (FormData from frontend),
    # then fall back to JSON for backward compatibility
    if request.content_type and 'multipart/form-data' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        # Filter out empty file entries (browser sends empty FileStorage when no file selected)
        uploaded_files = [f for f in request.files.getlist('files') if f and f.filename]
    elif request.content_type and 'application/x-www-form-urlencoded' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        uploaded_files = []
    else:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        uploaded_files = []

    if not user_message and not uploaded_files:
        return jsonify({'error': 'Message cannot be empty'}), 400

    if not user_message and uploaded_files:
        user_message = 'Please analyze the uploaded file(s) and provide a summary.'

    # Get or create conversation
    if conversation_id:
        conversation = Conversation.query.filter_by(
            id=conversation_id, user_id=current_user.id
        ).first()
    else:
        conversation = None

    if not conversation:
        conversation = Conversation(
            user_id=current_user.id,
            title=user_message[:50] + ('...' if len(user_message) > 50 else '')
        )
        db.session.add(conversation)
        db.session.commit()

    # Get conversation history for context
    history = ChatHistory.query.filter_by(
        conversation_id=conversation.id
    ).order_by(ChatHistory.timestamp.desc()).limit(10).all()

    history_dicts = [{'prompt': h.prompt, 'response': h.response} for h in reversed(history)]

    prompt_text = user_message
    file_context = []

    if uploaded_files:
        upload_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            if not uploaded_file or uploaded_file.filename == '':
                continue

            if not allowed_file(uploaded_file.filename):
                return jsonify({'error': f'File type not allowed for {uploaded_file.filename}. Supported: PDF, DOCX, TXT, CSV, XLSX'}), 400

            original_filename = secure_filename(uploaded_file.filename)
            stored_filename = generate_stored_filename(original_filename)
            file_type = original_filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(upload_dir, stored_filename)
            uploaded_file.save(filepath)

            # Get file size
            file_size = os.path.getsize(filepath)

            # Extract text for AI context
            extracted_text = extract_text(filepath, file_type)

            # Save file record to database (so it shows in Files section)
            file_record = UploadedFile(
                user_id=current_user.id,
                filename=original_filename,
                stored_filename=stored_filename,
                file_type=file_type,
                file_size=file_size,
                extracted_text=extracted_text or '',
                summary=f'Uploaded via chatbot in conversation'
            )
            db.session.add(file_record)

            if extracted_text:
                file_context.append(f"File: {original_filename}\nType: {file_type.upper()}\nContent:\n{extracted_text[:12000]}")

        if file_context:
            prompt_text = (
                f"{user_message}\n\nAttached file content(s):\n"
                f"{'\n\n'.join(file_context)}\n\n"
                "Please answer using the uploaded file content."
            )

    # Get AI response — use data analysis system prompt if files are attached
    system_prompt = None
    if file_context:
        system_prompt = (
            "You are an expert data analyst AI assistant. The user has uploaded file(s) for analysis. "
            "Analyze the provided data carefully and answer the user's question accurately. "
            "When working with tabular data (CSV, Excel), perform calculations, identify trends, "
            "and provide specific numbers from the data. Format your responses with markdown "
            "tables when showing data comparisons. Be precise with numbers and cite specific "
            "rows/columns from the data."
        )
    result = ai_service.get_chat_response(prompt_text, conversation_history=history_dicts, system_prompt=system_prompt)

    # Save to database
    display_prompt = user_message
    if uploaded_files:
        attached_names = ', '.join([f.filename for f in uploaded_files if f.filename])
        if attached_names:
            display_prompt = f"{user_message} [Files: {attached_names}]"

    chat_entry = ChatHistory(
        user_id=current_user.id,
        conversation_id=conversation.id,
        prompt=display_prompt,
        response=result['response'],
        response_time=result['response_time']
    )
    db.session.add(chat_entry)

    # Update conversation title if it's the first message
    if len(history) == 0:
        conversation.title = user_message[:50] + ('...' if len(user_message) > 50 else '')

    # Log activity
    log = AnalyticsLog(
        user_id=current_user.id,
        activity_type='chat',
        details=f'Message in conversation {conversation.id}'
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'success': True,
        'response': result['response'],
        'response_time': round(result['response_time'], 2),
        'conversation_id': conversation.id,
        'conversation_title': conversation.title,
        'message_id': chat_entry.id,
        'timestamp': chat_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    })


@chatbot_bp.route('/chat/history/<int:conversation_id>')
@login_required
def get_conversation_history(conversation_id):
    """Retrieve the full message history for a specific conversation."""
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()

    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    messages = ChatHistory.query.filter_by(
        conversation_id=conversation_id
    ).order_by(ChatHistory.timestamp.asc()).all()

    return jsonify({
        'success': True,
        'conversation_id': conversation_id,
        'title': conversation.title,
        'messages': [msg.to_dict() for msg in messages]
    })


@chatbot_bp.route('/chat/delete/<int:conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """Delete a conversation and all its messages."""
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()

    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    db.session.delete(conversation)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Conversation deleted'})


@chatbot_bp.route('/chat/clear/<int:conversation_id>', methods=['POST'])
@login_required
def clear_conversation(conversation_id):
    """Clear all messages in a conversation but keep the conversation."""
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()

    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    ChatHistory.query.filter_by(conversation_id=conversation_id).delete()
    db.session.commit()

    return jsonify({'success': True, 'message': 'Conversation cleared'})


@chatbot_bp.route('/chat/export/<int:conversation_id>/<string:format>')
@login_required
def export_conversation(conversation_id, format):
    """
    Export a conversation in the specified format (txt, docx, pdf).

    Args:
        conversation_id: ID of the conversation to export.
        format: Export format — 'txt', 'docx', or 'pdf'.
    """
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=current_user.id
    ).first()

    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    messages = ChatHistory.query.filter_by(
        conversation_id=conversation_id
    ).order_by(ChatHistory.timestamp.asc()).all()

    if not messages:
        return jsonify({'error': 'No messages to export'}), 404

    # Log export activity
    log = AnalyticsLog(
        user_id=current_user.id,
        activity_type='export',
        details=f'Exported conversation {conversation_id} as {format}'
    )
    db.session.add(log)
    db.session.commit()

    title = conversation.title or 'Chat Export'

    if format == 'txt':
        buffer = export_chat_to_txt(messages, title)
        return send_file(buffer, as_attachment=True,
                        download_name=f'chat_export_{conversation_id}.txt',
                        mimetype='text/plain')
    elif format == 'docx':
        buffer = export_chat_to_docx(messages, title)
        return send_file(buffer, as_attachment=True,
                        download_name=f'chat_export_{conversation_id}.docx',
                        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    elif format == 'pdf':
        buffer = export_chat_to_pdf(messages, title)
        return send_file(buffer, as_attachment=True,
                        download_name=f'chat_export_{conversation_id}.pdf',
                        mimetype='application/pdf')
    else:
        return jsonify({'error': 'Unsupported format'}), 400
