"""
Chatbot Routes
==============
Handles the AI chatbot interface, conversation management,
message sending/receiving, chat history operations, and
sales-chart / financial-insight generation from uploaded files.
All endpoints require authentication.

CHANGELOG (this revision):
  - Stale-file fix: the "fall back to last uploaded spreadsheet" logic no
    longer fires when the user attached a NON-tabular file (PDF/DOCX/TXT)
    this turn. Previously, asking a question about a freshly uploaded PDF
    would still pull in a stale, unrelated spreadsheet's computed figures
    as "authoritative context", corrupting the answer.
  - Aggregation fix: the final categorical fallback branch (used when a
    time-series keyword is present but no month column was found) now
    respects the requested aggregation (sum/mean/min/max) instead of
    silently always returning row counts.
  - AI chart-type selection: every chart-generation call site now asks
    insight_service.suggest_chart_type() to pick bar/barh/line/area/pie
    instead of hardcoding 'bar'. This runs a real AI call on every chart
    request (by design) with a deterministic heuristic fallback if that
    call fails. Explicit chart_type_override from the user/frontend still
    always wins.
"""

import os
import logging
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
from services.chart_service import ChartService
from services.insight_service import (
    load_dataframe, find_column, detect_group_column,
    monthly_totals, quarterly_totals, category_counts, category_sums,
    category_aggregate, requested_aggregation,
    wants_chart, wants_quarterly_breakdown, build_ai_context,
    is_count_query, is_unique_value_query, find_column_for_question,
    find_time_column_by_value,
    exact_category_count, unique_column_values, extract_time_from_question,
    suggest_chart_type,
    find_numeric_column_generic, find_date_column_generic, find_group_column_generic,
    MONTH_COLUMN_ALIASES, VALUE_COLUMN_ALIASES,
)

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint('chatbot', __name__)

TABULAR_TYPES = ('csv', 'xlsx', 'xls')

# The underlying model has no real code-execution tool. Left unguarded, it
# will sometimes narrate a fake "let's simulate running this code" transcript
# instead of just answering — this instruction exists specifically to shut
# that down, independent of whether a real chart was generated below.
NO_FAKE_CODE_INSTRUCTION = (
    "You do not have a code execution environment. NEVER write example code and "
    "present it as though you ran it, NEVER say things like 'let's simulate code' "
    "or 'we will now load the file and compute...', and NEVER fabricate console/notebook "
    "output. If the user asked for a chart, a real chart image is generated separately "
    "and shown alongside your message — do not describe writing plotting code for it. "
    "Answer directly in plain prose and markdown tables only, using the real data/figures "
    "already given to you in this conversation."
)


def _latest_tabular_upload(user_id):
    """
    Fall back to the user's most recently uploaded spreadsheet when a chat
    message references data without attaching a fresh file this turn
    (e.g. "now show me that as a quarterly chart").
    """
    return UploadedFile.query.filter(
        UploadedFile.user_id == user_id,
        UploadedFile.file_type.in_(TABULAR_TYPES)
    ).order_by(UploadedFile.upload_date.desc()).first()


def _build_chart(chart_service, labels, values, chart_type, title, xlabel, ylabel, palette, style=None):
    """Small wrapper so chart failures never take down the whole request."""
    try:
        return chart_service.render_base64(
            labels, values, chart_type=chart_type, title=title,
            xlabel=xlabel, ylabel=ylabel, palette=palette, style=style,
        )
    except Exception as e:
        logger.error(f'Chart generation failed: {e}')
        return None


def _resolve_chart_type(chart_type_override, labels, values, is_time_series, group_label=None):
    """
    Central place that decides the chart type for every chart-generation
    call site below. Explicit user/frontend override always wins; otherwise
    delegates to insight_service.suggest_chart_type(), which makes an AI
    call (with a deterministic heuristic fallback) on every invocation.

    Returns:
        tuple(str, str): (chart_type, source) — source is 'override' when
        the caller/frontend explicitly requested a type, or one of
        'ai' / 'heuristic' / 'heuristic_fallback' from suggest_chart_type()
        otherwise. This is surfaced in the API response as
        'chart_type_source' specifically so it's possible to verify the AI
        call actually ran, instead of assuming it did.
    """
    if chart_type_override:
        logger.info(f"Chart type '{chart_type_override}' used from explicit override.")
        return chart_type_override, 'override'
    return suggest_chart_type(
        labels, values,
        is_time_series=is_time_series,
        ai_service=ai_service,
        group_label=group_label,
    )


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

    If the uploaded/previously-uploaded file is tabular (CSV/XLSX/XLS),
    this also tries several deterministic insight paths (unique-value
    lists, exact counts, categorical aggregates, month/sales totals)
    before ever calling the AI, and if the message looks like a chart
    request, renders a seaborn chart server-side as a base64 PNG using
    an AI-assisted chart-type pick (see _resolve_chart_type above).
    """
    # result_response must be initialized here, before any branching — see
    # original comment: it's only ever assigned inside nested "if tabular_
    # filepath: if df is not None:" blocks, but read again later at
    # top-level scope. Any message with no tabular file involved (the
    # common case) never runs those blocks, so an unassigned local would
    # raise UnboundLocalError -> 500 for every such message.
    result_response = None

    # Parse request — always try form data first (FormData from frontend),
    # then fall back to JSON for backward compatibility
    if request.content_type and 'multipart/form-data' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        chart_type_override = request.form.get('chart_type')
        chart_palette_override = request.form.get('chart_palette')
        chart_style_override = request.form.get('chart_style')
        # Filter out empty file entries (browser sends empty FileStorage when no file selected)
        uploaded_files = [f for f in request.files.getlist('files') if f and f.filename]
    elif request.content_type and 'application/x-www-form-urlencoded' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        chart_type_override = request.form.get('chart_type')
        chart_palette_override = request.form.get('chart_palette')
        chart_style_override = request.form.get('chart_style')
        uploaded_files = []
    else:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        chart_type_override = data.get('chart_type')
        chart_palette_override = data.get('chart_palette')
        chart_style_override = data.get('chart_style')
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

    # Tracks the spreadsheet (if any) this turn should analyze for insights/charts
    tabular_filepath = None
    tabular_file_type = None
    tabular_filename = None

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

            # Remember the most recent tabular file uploaded this turn for
            # insight/chart generation below.
            if file_type in TABULAR_TYPES:
                tabular_filepath = filepath
                tabular_file_type = file_type
                tabular_filename = original_filename

        if file_context:
            prompt_text = (
                f"{user_message}\n\nAttached file content(s):\n"
                f"{chr(10).join(file_context)}\n\n"
                "Please answer using the uploaded file content."
            )

    # Only fall back to a previous spreadsheet when this turn didn't already
    # attach a non-tabular file. Attaching a PDF/DOCX/TXT means the user is
    # asking about THAT file — pulling in an unrelated spreadsheet's numbers
    # as "authoritative context" would corrupt the answer.
    attached_non_tabular = bool(uploaded_files) and tabular_filepath is None

    if not tabular_filepath and not attached_non_tabular:
        fallback = _latest_tabular_upload(current_user.id)
        if fallback:
            fallback_path = os.path.join(current_app.config['UPLOAD_FOLDER'], fallback.stored_filename)
            if os.path.exists(fallback_path):
                tabular_filepath = fallback_path
                tabular_file_type = fallback.file_type
                tabular_filename = fallback.filename

    # --- Deterministic financial / categorical insight + chart generation --
    chart_image = None
    chart_type_source = None  # 'ai' | 'heuristic' | 'heuristic_fallback' | 'override' | None
    computed_context = None

    if tabular_filepath:
        df = load_dataframe(tabular_filepath, tabular_file_type)
        if df is not None:
            fiscal_start = current_app.config.get('FISCAL_YEAR_START_MONTH', 4)

            # Deterministic unique-value/list path. Do not let the LLM
            # infer unique values from the truncated file preview.
            if is_unique_value_query(user_message):
                unique_col = find_column_for_question(df.columns, user_message)

                # Natural language "test timings" should map to the Time column.
                if not unique_col and any(word in user_message.lower()
                                          for word in ('timing', 'timings', 'time slot', 'test time')):
                    time_candidates = [
                        col for col in df.columns
                        if 'time' in str(col).lower()
                    ]
                    if time_candidates:
                        unique_col = time_candidates[0]

                if unique_col:
                    values = unique_column_values(df, unique_col)
                    if values:
                        rows = "\n".join(f"- **{v}**" for v in values)
                        result_response = (
                            f"**Distinct values in `{unique_col}`:**\n\n"
                            f"{rows}\n\n"
                            f"Found **{len(values)} unique value(s)** by checking "
                            f"the complete uploaded file."
                        )
                        computed_context = (
                            "AUTHORITATIVE DETERMINISTIC RESULT — DO NOT RECALCULATE:\n"
                            f"Column: {unique_col}\n"
                            f"Unique values: {', '.join(values)}\n"
                        )

            if result_response is None and is_count_query(user_message):
                question_col = find_column_for_question(df.columns, user_message)
                requested_time = extract_time_from_question(user_message)

                # find_column_for_question() is name-based: it requires the
                # literal word "time" (or "test"+"time") in the question.
                # Natural phrasing like "how many students have a test at
                # 9 AM" never says the word "time" — only a time value — so
                # name matching silently fails there. When a time value was
                # extracted but no column was found by name, search by
                # VALUE across all columns instead: this is what actually
                # fixes questions phrased this way.
                if not question_col and requested_time:
                    question_col = find_time_column_by_value(df, requested_time)

                if question_col and requested_time:
                    label, count, _ = exact_category_count(
                        df, question_col, requested_time, time_value=True
                    )
                    result_response = (
                        f"**Students scheduled at {label}: {count}**\n\n"
                        f"I counted the matching rows directly from the complete "
                        f"Excel/CSV column **{question_col}**."
                    )
                    computed_context = (
                        "AUTHORITATIVE DETERMINISTIC RESULT — DO NOT RECALCULATE:\n"
                        f"Column: {question_col}\n"
                        f"Requested value: {label}\n"
                        f"Exact row count: {count}\n"
                    )

            chart_keywords = current_app.config.get('CHART_TRIGGER_KEYWORDS')
            user_wants_chart = wants_chart(user_message, chart_keywords)

            # Alias-based detection first (fast, exact when it hits), then
            # fall back to generic schema-based detection for spreadsheets
            # that don't use the expected English vocabulary (e.g. a
            # column named 'Fees' or 'Cost' instead of 'Amount'/'Sales',
            # or a date column named 'Txn Date' instead of 'Month'/'Date').
            # This is what makes insight generation work on ANY uploaded
            # sheet — student, financial, inventory, stock, etc. — not
            # just ones matching a fixed keyword list.
            month_col = find_column(df.columns, MONTH_COLUMN_ALIASES) \
                or find_date_column_generic(df, user_message)
            value_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
                or find_numeric_column_generic(df, user_message, exclude=month_col)
            group_col = detect_group_column(df.columns, user_message) \
                or find_group_column_generic(df, user_message, exclude={month_col, value_col})

            # Decide whether the user is asking for a time-series/financial
            # calculation. A categorical question such as
            # "How many students are from each campus?" must NOT be routed
            # through the sales/month path merely because the workbook has
            # a numeric column.
            message_lower = user_message.lower()
            time_series_request = any(keyword in message_lower for keyword in (
                'sales', 'revenue', 'income', 'profit', 'expense',
                'amount', 'financial', 'month', 'monthly', 'quarter',
                'q1', 'q2', 'q3', 'q4', 'qtr', 'date', 'period'
            ))

            if computed_context is None and group_col and not time_series_request:
                # ---------------------------------------------------------
                # CATEGORICAL QUESTIONS
                # ---------------------------------------------------------
                # Python reads ALL rows and performs the calculation.
                # Ollama receives the result only for explanation/formatting.
                operation = requested_aggregation(user_message)

                if operation == 'count':
                    labels, values = category_counts(df, group_col)
                    ylabel = 'Number of rows'
                else:
                    numeric_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
                        or find_numeric_column_generic(df, user_message, exclude=group_col)
                    if not numeric_col or numeric_col == group_col:
                        # No numeric metric exists (even after the generic
                        # schema-based fallback), so a numerical aggregate
                        # cannot be performed. Fall back to exact row counts.
                        labels, values = category_counts(df, group_col)
                        operation = 'count'
                        ylabel = 'Number of rows'
                    else:
                        labels, values = category_aggregate(
                            df, group_col, numeric_col, operation
                        )
                        ylabel = numeric_col

                if labels:
                    computed_context = build_ai_context(
                        categories=(labels, values), category_label=group_col
                    )

                    if user_wants_chart:
                        title = f'{tabular_filename} — {group_col} Breakdown'
                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source = _resolve_chart_type(
                            chart_type_override, labels, values,
                            is_time_series=False, group_label=group_col,
                        )
                        chart_image = _build_chart(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            title,
                            group_col,
                            ylabel,
                            chart_palette_override,
                            chart_style_override,
                        )

            elif computed_context is None and month_col and value_col:
                # ---------------------------------------------------------
                # MONTH / SALES QUESTIONS
                # ---------------------------------------------------------
                monthly = monthly_totals(df, month_col, value_col)
                quarterly = quarterly_totals(df, month_col, value_col, fiscal_start)

                if monthly:
                    computed_context = build_ai_context(
                        monthly=monthly,
                        quarterly=quarterly,
                        fiscal_start_month=fiscal_start,
                    )

                    if user_wants_chart:
                        use_quarterly = wants_quarterly_breakdown(user_message) and quarterly
                        if use_quarterly:
                            labels, values = list(quarterly.keys()), list(quarterly.values())
                            title = f'{tabular_filename} — Quarterly {value_col}'
                        else:
                            labels = [name for _, name, _ in monthly]
                            values = [total for _, _, total in monthly]
                            title = f'{tabular_filename} — Monthly {value_col}'

                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source = _resolve_chart_type(
                            chart_type_override, labels, values,
                            is_time_series=True, group_label=value_col,
                        )
                        chart_image = _build_chart(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            title,
                            'Period',
                            value_col,
                            chart_palette_override,
                            chart_style_override,
                        )

            elif computed_context is None and group_col:
                # ---------------------------------------------------------
                # FINAL CATEGORICAL FALLBACK
                # ---------------------------------------------------------
                # Reached when a time-series keyword is present (e.g. the
                # message says "amount" or "total") but no month column was
                # detected. Previously this branch always used
                # category_counts() regardless of what was actually asked,
                # so "total amount per department" silently returned row
                # counts instead of sums. Now it respects the requested
                # aggregation just like the main categorical branch above.
                operation = requested_aggregation(user_message)

                if operation == 'count':
                    labels, values = category_counts(df, group_col)
                    ylabel = 'Number of rows'
                else:
                    numeric_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
                        or find_numeric_column_generic(df, user_message, exclude=group_col)
                    if not numeric_col or numeric_col == group_col:
                        labels, values = category_counts(df, group_col)
                        operation = 'count'
                        ylabel = 'Number of rows'
                    else:
                        labels, values = category_aggregate(
                            df, group_col, numeric_col, operation
                        )
                        ylabel = numeric_col

                if labels:
                    computed_context = build_ai_context(
                        categories=(labels, values),
                        category_label=group_col,
                    )

                    if user_wants_chart:
                        title = f'{tabular_filename} — {group_col} Breakdown'
                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source = _resolve_chart_type(
                            chart_type_override, labels, values,
                            is_time_series=False, group_label=group_col,
                        )
                        chart_image = _build_chart(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            title,
                            group_col,
                            ylabel,
                            chart_palette_override,
                            chart_style_override,
                        )

    # Ground the AI's response in the pre-computed figures instead of
    # letting it re-derive totals from the raw text dump.
    if computed_context:
        prompt_text = f"{prompt_text}\n\n{computed_context}"

    # Get AI response — use a data-analyst system prompt whenever files or
    # computed figures are involved. The no-fake-code instruction is always
    # included here regardless of whether a chart was actually generated,
    # since the underlying model can hallucinate a fake execution transcript
    # for ANY data question, not just chart requests.
    system_prompt = None
    if file_context or computed_context:
        system_prompt = (
            "You are an expert financial and data analyst AI assistant. The user has "
            "uploaded file(s) for analysis. Analyze the provided data carefully and answer "
            "the user's question accurately and directly.\n\n"
            f"{NO_FAKE_CODE_INSTRUCTION}\n\n"
            "When working with tabular data (CSV, Excel), state calculations, identify "
            "trends, and provide specific numbers from the data. If a block of "
            "'Pre-computed figures' or 'AUTHORITATIVE DETERMINISTIC RESULT' is provided, "
            "treat those numbers as authoritative — quote them directly and do not "
            "recalculate or estimate your own totals. Format your responses with markdown "
            "tables when showing data comparisons. Be precise with numbers and cite specific "
            "rows/columns/periods/categories from the data."
        )

    if result_response is not None:
        result = {'response': result_response, 'response_time': 0.0}
    else:
        result = ai_service.get_chat_response(
            prompt_text,
            conversation_history=history_dicts,
            system_prompt=system_prompt
        )

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
        response_time=result['response_time'],
        chart_image=chart_image
    )
    db.session.add(chat_entry)

    # Update conversation title if it's the first message
    if len(history) == 0:
        conversation.title = user_message[:50] + ('...' if len(user_message) > 50 else '')

    # Log activity — chart_type_source included so server logs/analytics
    # can confirm whether the AI actually made the chart-type pick.
    log = AnalyticsLog(
        user_id=current_user.id,
        activity_type='chart' if chart_image else 'chat',
        details=(
            f'Message in conversation {conversation.id}'
            + (f' (chart generated, type source: {chart_type_source})' if chart_image else '')
        )
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
        'timestamp': chat_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'chart_image': chart_image,
        # 'ai' = AI call succeeded and picked this type; 'heuristic' = no
        # chart was requested or no AI call was made; 'heuristic_fallback'
        # = the AI call ran but failed/returned something unusable;
        # 'override' = an explicit chart_type was passed in the request.
        # None when no chart was generated at all this turn.
        'chart_type_source': chart_type_source
    })


@chatbot_bp.route('/chat/chart/<int:file_id>', methods=['POST'])
@login_required
def generate_chart(file_id):
    """
    Regenerate a chart for a previously uploaded spreadsheet with explicit,
    user-chosen parameters — this is the "maximum configurability" control
    surface: a frontend chart-customizer panel can call this directly
    without re-running the whole chat pipeline.

    If chart_type is omitted from the request body, the AI-assisted
    suggest_chart_type() picks one (same as /chat/send); if chart_type IS
    provided, that explicit choice always wins — this endpoint exists
    specifically for manual override, so we never second-guess it here.

    Expects JSON body:
        {
            "chart_type": "bar" | "barh" | "line" | "area" | "pie",   optional
            "period": "month" | "quarter" | "category",
            "palette": "<any seaborn/matplotlib palette name>",   optional
            "style":   "darkgrid" | "whitegrid" | "dark" | "white" | "ticks",  optional
            "month_column": "<override auto-detected column>",    optional
            "value_column": "<override auto-detected column>",    optional
            "group_column": "<explicit categorical column for period='category'>"  optional
        }
    """
    upload = UploadedFile.query.filter_by(id=file_id, user_id=current_user.id).first()
    if not upload:
        return jsonify({'error': 'File not found'}), 404
    if upload.file_type not in TABULAR_TYPES:
        return jsonify({'error': 'File is not tabular data (expected CSV, XLSX, or XLS)'}), 400

    data = request.get_json(silent=True) or {}
    chart_type = data.get('chart_type')  # None -> AI-assisted pick below
    period = data.get('period', 'month')  # 'month' | 'quarter' | 'category'
    palette = data.get('palette')
    style = data.get('style')

    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], upload.stored_filename)
    if not os.path.exists(filepath):
        # Expected on Render's free tier if the app has restarted since the
        # file was uploaded — local disk is ephemeral and does not persist
        # across deploys/restarts.
        return jsonify({'error': 'Original file no longer available on the server. Please re-upload it.'}), 404

    df = load_dataframe(filepath, upload.file_type)
    if df is None:
        return jsonify({'error': 'Could not parse file'}), 400

    fiscal_start = current_app.config.get('FISCAL_YEAR_START_MONTH', 4)

    if period == 'category':
        group_col = data.get('group_column') or detect_group_column(df.columns, upload.filename) \
            or find_group_column_generic(df, upload.filename)
        if not group_col or group_col not in df.columns:
            return jsonify({'error': 'Could not determine a category column. Pass group_column explicitly.'}), 400
        value_col = data.get('value_column') or find_column(df.columns, VALUE_COLUMN_ALIASES) \
            or find_numeric_column_generic(df, upload.filename, exclude=group_col)
        if value_col and value_col in df.columns and value_col != group_col:
            labels, values = category_sums(df, group_col, value_col)
            ylabel = value_col
        else:
            labels, values = category_counts(df, group_col)
            ylabel = 'Count'
        if not labels:
            return jsonify({'error': 'No valid data found for that column'}), 400
        title = f'{upload.filename} — {group_col} Breakdown'
        xlabel = group_col
        is_time_series = False
        group_label = group_col
    else:
        month_col = data.get('month_column') or find_column(df.columns, MONTH_COLUMN_ALIASES) \
            or find_date_column_generic(df, upload.filename)
        value_col = data.get('value_column') or find_column(df.columns, VALUE_COLUMN_ALIASES) \
            or find_numeric_column_generic(df, upload.filename, exclude=month_col)
        if not month_col or not value_col or month_col not in df.columns or value_col not in df.columns:
            return jsonify({'error': 'Could not detect month/sales columns in this file. '
                                      'Pass month_column and value_column explicitly, or use period=category.'}), 400
        if period == 'quarter':
            totals = quarterly_totals(df, month_col, value_col, fiscal_start)
            if not totals:
                return jsonify({'error': 'No valid quarterly data found in this file'}), 400
            labels, values = list(totals.keys()), list(totals.values())
            title = f'{upload.filename} — Quarterly {value_col}'
        else:
            monthly = monthly_totals(df, month_col, value_col)
            if not monthly:
                return jsonify({'error': 'No valid monthly data found in this file'}), 400
            labels = [name for _, name, _ in monthly]
            values = [total for _, _, total in monthly]
            title = f'{upload.filename} — Monthly {value_col}'
        xlabel = 'Period'
        ylabel = value_col
        is_time_series = True
        group_label = value_col

    if chart_type:
        resolved_chart_type, chart_type_source = chart_type, 'override'
    else:
        resolved_chart_type, chart_type_source = suggest_chart_type(
            labels, values,
            is_time_series=is_time_series,
            ai_service=ai_service,
            group_label=group_label,
        )

    chart_service = ChartService.from_app_config(current_app.config)
    try:
        chart_image = chart_service.render_base64(
            labels, values, chart_type=resolved_chart_type, title=title,
            xlabel=xlabel, ylabel=ylabel, palette=palette, style=style,
        )
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 400

    log = AnalyticsLog(
        user_id=current_user.id,
        activity_type='chart',
        details=f'Generated {resolved_chart_type}/{period} chart for {upload.filename} (source: {chart_type_source})'
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'success': True,
        'chart_image': chart_image,
        'chart_type': resolved_chart_type,
        'chart_type_source': chart_type_source,
        'title': title
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
