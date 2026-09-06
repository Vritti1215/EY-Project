"""
Chatbot Routes
==============
Handles the AI chatbot interface, conversation management,
message sending/receiving, chat history operations, and
sales-chart / financial-insight generation from uploaded files.
All endpoints require authentication.
"""

import os
import logging
import json
import statistics
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

# --- NEW: feature modules -----------------------------------------------
from services.chart_recommender import recommend_chart_set
from services.chart_customization import parse_chart_customization, apply_month_range_filter
from services.prompt_guidance import classify_prompt_intent, system_prompt_for, EDGE_CASE_INSTRUCTIONS
from services.data_profile_service import build_data_profile, validate_profile
# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint('chatbot', __name__)

TABULAR_TYPES = ('csv', 'xlsx', 'xls')

NO_FAKE_CODE_INSTRUCTION = (
    "You do not have a code execution environment. NEVER write example code and "
    "present it as though you ran it, NEVER say things like 'let's simulate code' "
    "or 'we will now load the file and compute...', and NEVER fabricate console/notebook "
    "output. If the user asked for a chart, a real chart image is generated separately "
    "and shown alongside your message — do not describe writing plotting code for it. "
    "Answer directly in plain prose and markdown tables only, using the real data/figures "
    "already given to you in this conversation."
)


def _parse_bool_field(value):
    """
    Parse a boolean form/JSON field into True/False/None.
    None means "not sent" — the caller should fall back to text-parsed
    customization or a hardcoded default, same precedence pattern used
    for chart_type/chart_palette/chart_style overrides.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('true', '1', 'on', 'yes'):
        return True
    if text in ('false', '0', 'off', 'no'):
        return False
    return None


def _resolve_bool(explicit_override, text_parsed, default):
    """Explicit UI override > natural-language mention > hardcoded default."""
    if explicit_override is not None:
        return explicit_override
    if text_parsed is not None:
        return text_parsed
    return default


def _latest_tabular_upload(user_id):
    return UploadedFile.query.filter(
        UploadedFile.user_id == user_id,
        UploadedFile.file_type.in_(TABULAR_TYPES)
    ).order_by(UploadedFile.upload_date.desc()).first()


def _build_chart(
    chart_service,
    labels,
    values,
    chart_type,
    title,
    xlabel,
    ylabel,
    palette=None,
    style=None,
    show_data_labels=True,
    show_variance=False,
    show_trend=False,
    noise_variance=None,
):
    """Render one chart and return its base64 data URI."""
    try:
        return chart_service.render_base64(
            labels,
            values,
            chart_type=chart_type,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            palette=palette,
            style=style,
            show_data_labels=show_data_labels,
            show_variance=show_variance,
            show_trend=show_trend,
            noise_variance=noise_variance,
        )
    except Exception:
        logger.exception("Chart generation failed: %s", chart_type)
        return None


def _resolve_chart_set(chart_type_override, labels, values, is_time_series, group_label=None):
    """
    NEW: replaces the old _resolve_chart_type(). Still returns a single
    (chart_type, source) pair for backward compatibility with every
    existing call site below, but now goes through recommend_chart_set()
    so a ranked list of alternatives is available too (feature: multiple
    recommended chart types from one prompt). Explicit override still
    always wins and skips the recommender entirely, same as before.
    """
    if chart_type_override:
        logger.info(f"Chart type '{chart_type_override}' used from explicit override.")
        return chart_type_override, 'override', []

    chart_set = recommend_chart_set(
        labels, values,
        is_time_series=is_time_series,
        ai_service=ai_service,
        group_label=group_label,
    )
    alternatives = [
        {'chart_type': a.chart_type, 'reason': a.reason}
        for a in chart_set.alternatives
    ]
    return chart_set.best.chart_type, chart_set.best.source, alternatives



def _profile_context(profile):
    """Serialize the validated data profile for the answer model."""
    try:
        return json.dumps(
            profile.to_ai_context(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except Exception:
        return str(profile.to_ai_context())


def _chart_variance(labels, values):
    """Return descriptive statistics for the real plotted values."""
    numeric = []
    for value in values or []:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue

    if not numeric:
        return None

    mean = statistics.fmean(numeric)
    variance = statistics.pvariance(numeric) if len(numeric) > 1 else 0.0
    std = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0

    changes = []
    for i in range(1, len(numeric)):
        previous = numeric[i - 1]
        if previous == 0:
            continue
        changes.append({
            'from': str(labels[i - 1]),
            'to': str(labels[i]),
            'pct': round((numeric[i] - previous) / abs(previous) * 100, 2),
        })

    return {
        'count': len(numeric),
        'mean': round(mean, 4),
        'std_dev': round(std, 4),
        'variance': round(variance, 4),
        'min': round(min(numeric), 4),
        'max': round(max(numeric), 4),
        'range': round(max(numeric) - min(numeric), 4),
        'coefficient_of_variation_pct': (
            round(std / abs(mean) * 100, 2) if mean else None
        ),
        'period_change_pct': changes,
        'noise_policy': (
            'Charts use the original uploaded values. '
            'No artificial random noise is added.'
        ),
    }


def _render_chart_options(
    chart_service,
    labels,
    values,
    best_type,
    alternatives,
    title,
    xlabel,
    ylabel,
    palette=None,
    style=None,
    show_data_labels=True,
    show_variance=False,
    show_trend=False,
    noise_variance=None,
):
    """Render the recommended chart and all available alternatives."""
    picks = [
        {
            'chart_type': best_type,
            'reason': 'Recommended visualization',
        }
    ]

    seen = {best_type}
    for alternative in alternatives or []:
        chart_type = alternative.get('chart_type')
        if chart_type and chart_type not in seen:
            picks.append(alternative)
            seen.add(chart_type)

    rendered = []
    for index, pick in enumerate(picks, start=1):
        chart_type = pick['chart_type']
        image = _build_chart(
            chart_service,
            labels,
            values,
            chart_type,
            title if index == 1 else f'{title} — {chart_type.capitalize()}',
            xlabel,
            ylabel,
            palette,
            style,
            show_data_labels=show_data_labels,
            show_variance=show_variance,
            show_trend=show_trend,
            noise_variance=noise_variance,
        )
        if image:
            rendered.append({
                'option': index,
                'chart_type': chart_type,
                'reason': pick.get('reason', ''),
                'image': image,
            })

    return rendered



@chatbot_bp.route('/chat')
@login_required
def chat():
    conversations = Conversation.query.filter_by(
        user_id=current_user.id
    ).order_by(Conversation.updated_at.desc()).all()
    return render_template('chatbot.html', conversations=conversations)


@chatbot_bp.route('/chat/conversations')
@login_required
def list_conversations():
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
    conversation = Conversation(user_id=current_user.id, title='New Conversation')
    db.session.add(conversation)
    db.session.commit()
    return jsonify({'success': True, 'conversation_id': conversation.id, 'title': conversation.title})


@chatbot_bp.route('/chat/send', methods=['POST'])
@login_required
def send_message():
    result_response = None

    if request.content_type and 'multipart/form-data' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        chart_type_override = request.form.get('chart_type')
        chart_palette_override = request.form.get('chart_palette')
        chart_style_override = request.form.get('chart_style')
        chart_noise_variance_override = request.form.get('chart_noise_variance', type=float)
        chart_title_override = request.form.get('chart_title')
        chart_xlabel_override = request.form.get('chart_xlabel')
        chart_ylabel_override = request.form.get('chart_ylabel')
        chart_month_start_override = request.form.get('chart_month_start', type=int)
        chart_month_end_override = request.form.get('chart_month_end', type=int)
        uploaded_files = [f for f in request.files.getlist('files') if f and f.filename]
    elif request.content_type and 'application/x-www-form-urlencoded' in request.content_type:
        user_message = request.form.get('message', '').strip()
        conversation_id = request.form.get('conversation_id')
        chart_type_override = request.form.get('chart_type')
        chart_palette_override = request.form.get('chart_palette')
        chart_style_override = request.form.get('chart_style')
        chart_noise_variance_override = request.form.get('chart_noise_variance', type=float)
        chart_title_override = request.form.get('chart_title')
        chart_xlabel_override = request.form.get('chart_xlabel')
        chart_ylabel_override = request.form.get('chart_ylabel')
        chart_month_start_override = request.form.get('chart_month_start', type=int)
        chart_month_end_override = request.form.get('chart_month_end', type=int)
        uploaded_files = []
    else:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        chart_type_override = data.get('chart_type')
        chart_palette_override = data.get('chart_palette')
        chart_style_override = data.get('chart_style')
        chart_noise_variance_override = data.get('chart_noise_variance')
        chart_title_override = data.get('chart_title')
        chart_xlabel_override = data.get('chart_xlabel')
        chart_ylabel_override = data.get('chart_ylabel')
        chart_month_start_override = data.get('chart_month_start')
        chart_month_end_override = data.get('chart_month_end')
        uploaded_files = []

    if not user_message and not uploaded_files:
        return jsonify({'error': 'Message cannot be empty'}), 400

    if not user_message and uploaded_files:
        user_message = 'Please analyze the uploaded file(s) and provide a summary.'

    # --- NEW: parse natural-language chart customization from the message
    # (date range / title / axis renaming — palette & style are already
    # covered by the dropdown overrides above).
    chart_custom = parse_chart_customization(user_message)

    # Explicit dropdown selections always win (same precedence as the
    # existing title/axis-label overrides above); natural-language
    # mentions in the typed message ("pie chart", "ey theme", "make it
    # blue") are the fallback when the user didn't touch the dropdowns.
    chart_type_override = chart_type_override or chart_custom.chart_type
    chart_palette_override = chart_palette_override or chart_custom.palette
    chart_noise_variance = (
        chart_noise_variance_override
        if chart_noise_variance_override not in (None, '')
        else chart_custom.noise_variance
    )
    if chart_title_override:
        chart_custom.title = chart_title_override
    if chart_xlabel_override:
        chart_custom.xlabel = chart_xlabel_override
    if chart_ylabel_override:
        chart_custom.ylabel = chart_ylabel_override
    if chart_month_start_override and chart_month_end_override:
        chart_custom.date_range = (int(chart_month_start_override), int(chart_month_end_override))

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

    history = ChatHistory.query.filter_by(
        conversation_id=conversation.id
    ).order_by(ChatHistory.timestamp.desc()).limit(10).all()

    history_dicts = [{'prompt': h.prompt, 'response': h.response} for h in reversed(history)]

    prompt_text = user_message
    file_context = []

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

            file_size = os.path.getsize(filepath)
            extracted_text = extract_text(filepath, file_type)

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

    attached_non_tabular = bool(uploaded_files) and tabular_filepath is None

    if not tabular_filepath and not attached_non_tabular:
        fallback = _latest_tabular_upload(current_user.id)
        if fallback:
            fallback_path = os.path.join(current_app.config['UPLOAD_FOLDER'], fallback.stored_filename)
            if os.path.exists(fallback_path):
                tabular_filepath = fallback_path
                tabular_file_type = fallback.file_type
                tabular_filename = fallback.filename

    chart_image = None
    chart_type_source = None
    chart_alternatives = []
    chart_choices = []
    variance_summary = None
    computed_context = None
    profile_warnings = []
    profile_ai_context = None

    if tabular_filepath:
        df = load_dataframe(tabular_filepath, tabular_file_type)
        if df is not None:
            fiscal_start = current_app.config.get('FISCAL_YEAR_START_MONTH', 4)

            # --- NEW: structured data profile + validation gate, built
            # before any deterministic-path or AI work touches this
            # dataframe. This doesn't replace the alias/generic column
            # detection below — it's an additional sanity check that
            # surfaces issues (e.g. no numeric column at all) early.
            profile = build_data_profile(df)
            profile_ai_context = _profile_context(profile)
            is_valid, profile_issues = validate_profile(profile)
            if not is_valid:
                profile_warnings = profile_issues
                logger.warning(f"Data profile validation issues for {tabular_filename}: {profile_issues}")

            if is_unique_value_query(user_message):
                unique_col = find_column_for_question(df.columns, user_message)

                if not unique_col and any(word in user_message.lower()
                                          for word in ('timing', 'timings', 'time slot', 'test time')):
                    time_candidates = [col for col in df.columns if 'time' in str(col).lower()]
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

            month_col = find_column(df.columns, MONTH_COLUMN_ALIASES) \
                or find_date_column_generic(df, user_message)
            value_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
                or find_numeric_column_generic(df, user_message, exclude=month_col)
            group_col = detect_group_column(df.columns, user_message) \
                or find_group_column_generic(df, user_message, exclude={month_col, value_col})

            # --- NEW: apply a natural-language date-range filter, if the
            # user asked for one and we have a date column to filter on.
            # Applied to a COPY so downstream deterministic paths (unique
            # values, exact counts above) already ran against the full
            # dataset — only the chart/aggregate paths below see the
            # filtered window.
            if chart_custom.date_range and month_col:
                df = apply_month_range_filter(df, month_col, chart_custom.date_range)

            # An EXPLICIT column mention (e.g. "...by region", "...per
            # product") is a much stronger, deterministic signal than the
            # loose financial-keyword heuristic below — it must win even
            # when the question also contains a word like "revenue" or
            # "amount" that would otherwise misroute this into the
            # monthly/time-series branch. Without this, "total revenue by
            # region" was being treated as a time-series request (because
            # of the word "revenue"), producing no usable result and
            # leaving the AI with no real numbers to answer from.
            explicit_group_match = detect_group_column(df.columns, user_message) is not None

            message_lower = user_message.lower()
            time_series_request = any(keyword in message_lower for keyword in (
                'sales', 'revenue', 'income', 'profit', 'expense',
                'amount', 'financial', 'month', 'monthly', 'quarter',
                'q1', 'q2', 'q3', 'q4', 'qtr', 'date', 'period'
            )) and not explicit_group_match

            if computed_context is None and group_col and not time_series_request:
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
                        labels, values = category_aggregate(df, group_col, numeric_col, operation)
                        ylabel = numeric_col

                if labels:
                    computed_context = build_ai_context(
                        categories=(labels, values), category_label=group_col
                    )

                    if user_wants_chart:
                        title = chart_custom.title or f'{tabular_filename} — {group_col} Breakdown'
                        xlabel = chart_custom.xlabel or group_col
                        ylabel = chart_custom.ylabel or ylabel
                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source, chart_alternatives = _resolve_chart_set(
                            chart_type_override, labels, values,
                            is_time_series=False, group_label=group_col,
                        )
                        chart_choices = _render_chart_options(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            chart_alternatives,
                            title,
                            xlabel,
                            ylabel,
                            chart_palette_override,
                            chart_style_override,
                            show_data_labels=(chart_custom.show_data_labels is not False),
                            show_variance=bool(chart_custom.show_variance),
                            show_trend=bool(chart_custom.show_trend),
                            noise_variance=chart_noise_variance,
                        )
                        chart_image = (
                            chart_choices[0]['image'] if chart_choices else None
                        )
                        variance_summary = _chart_variance(labels, values)

            elif computed_context is None and month_col and value_col:
                monthly = monthly_totals(df, month_col, value_col)
                quarterly = quarterly_totals(df, month_col, value_col, fiscal_start)

                if monthly:
                    computed_context = build_ai_context(
                        monthly=monthly, quarterly=quarterly, fiscal_start_month=fiscal_start,
                    )

                    if user_wants_chart:
                        use_quarterly = wants_quarterly_breakdown(user_message) and quarterly
                        if use_quarterly:
                            labels, values = list(quarterly.keys()), list(quarterly.values())
                            default_title = f'{tabular_filename} — Quarterly {value_col}'
                        else:
                            labels = [name for _, name, _ in monthly]
                            values = [total for _, _, total in monthly]
                            default_title = f'{tabular_filename} — Monthly {value_col}'

                        title = chart_custom.title or default_title
                        xlabel = chart_custom.xlabel or 'Period'
                        ylabel = chart_custom.ylabel or value_col

                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source, chart_alternatives = _resolve_chart_set(
                            chart_type_override, labels, values,
                            is_time_series=True, group_label=value_col,
                        )
                        chart_choices = _render_chart_options(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            chart_alternatives,
                            title,
                            xlabel,
                            ylabel,
                            chart_palette_override,
                            chart_style_override,
                            show_data_labels=(chart_custom.show_data_labels is not False),
                            show_variance=bool(chart_custom.show_variance),
                            show_trend=bool(chart_custom.show_trend),
                            noise_variance=chart_noise_variance,
                        )
                        chart_image = (
                            chart_choices[0]['image'] if chart_choices else None
                        )
                        variance_summary = _chart_variance(labels, values)

            elif computed_context is None and group_col:
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
                        labels, values = category_aggregate(df, group_col, numeric_col, operation)
                        ylabel = numeric_col

                if labels:
                    computed_context = build_ai_context(
                        categories=(labels, values), category_label=group_col,
                    )

                    if user_wants_chart:
                        title = chart_custom.title or f'{tabular_filename} — {group_col} Breakdown'
                        xlabel = chart_custom.xlabel or group_col
                        ylabel = chart_custom.ylabel or ylabel
                        chart_service = ChartService.from_app_config(current_app.config)
                        resolved_chart_type, chart_type_source, chart_alternatives = _resolve_chart_set(
                            chart_type_override, labels, values,
                            is_time_series=False, group_label=group_col,
                        )
                        chart_choices = _render_chart_options(
                            chart_service,
                            labels,
                            values,
                            resolved_chart_type,
                            chart_alternatives,
                            title,
                            xlabel,
                            ylabel,
                            chart_palette_override,
                            chart_style_override,
                            show_data_labels=(chart_custom.show_data_labels is not False),
                            show_variance=bool(chart_custom.show_variance),
                            show_trend=bool(chart_custom.show_trend),
                            noise_variance=chart_noise_variance,
                        )
                        chart_image = (
                            chart_choices[0]['image'] if chart_choices else None
                        )
                        variance_summary = _chart_variance(labels, values)

    if profile_ai_context:
        prompt_text += (
            "\n\nSTRUCTURED DATA PROFILE (AUTHORITATIVE SCHEMA):\n"
            + profile_ai_context
            + "\nUse only columns present in this profile. "
              "Do not invent columns or metadata."
        )

    if variance_summary:
        prompt_text += (
            "\n\nREAL-DATA VARIANCE SUMMARY (DESCRIPTIVE ONLY):\n"
            + json.dumps(variance_summary, ensure_ascii=False, indent=2)
            + "\nDo not alter the underlying values or add synthetic noise."
        )

    if computed_context:
        prompt_text = f"{prompt_text}\n\n{computed_context}"

    # --- NEW: system prompt now goes through classify_prompt_intent() /
    # system_prompt_for() instead of a single hardcoded string, so vague
    # prompts, chart-requests-with-no-file, and unsupported-column
    # references each get a targeted instruction. The data-analyst /
    # no-fake-code instructions are still layered in whenever files or
    # computed figures are involved, same as before.
    available_columns = None
    if tabular_filepath:
        try:
            # df is already loaded above when tabular_filepath is set; this
            # only re-reads if something upstream failed before df was bound.
            available_columns = list(df.columns) if 'df' in locals() and df is not None else None
        except Exception:
            available_columns = None

    intent = classify_prompt_intent(
        user_message,
        has_file=bool(file_context or tabular_filepath),
        available_columns=available_columns,
    )
    base_system_prompt = system_prompt_for(intent)

    system_prompt = None
    if file_context or computed_context:
        system_prompt = (
            "You are an expert financial and data analyst AI assistant. The user has "
            "uploaded file(s) for analysis. Analyze the provided data carefully and answer "
            "the user's question accurately and directly.\n\n"
            f"{NO_FAKE_CODE_INSTRUCTION}\n\n"
            f"{base_system_prompt}\n\n"
            "When working with tabular data (CSV, Excel), state calculations, identify "
            "trends, and provide specific numbers from the data. If a block of "
            "'Pre-computed figures' or 'AUTHORITATIVE DETERMINISTIC RESULT' is provided, "
            "treat those numbers as authoritative — quote them directly and do not "
            "recalculate or estimate your own totals. Format your responses with markdown "
            "tables when showing data comparisons. Be precise with numbers and cite specific "
            "rows/columns/periods/categories from the data."
            f"{EDGE_CASE_INSTRUCTIONS}"
        )
    elif intent.category != 'general_chat':
        # Vague / chart-no-data / unsupported-column cases still apply
        # even with no file involved this turn.
        system_prompt = base_system_prompt

    if result_response is not None:
        result = {'response': result_response, 'response_time': 0.0}
    else:
        result = ai_service.get_chat_response(
            prompt_text, conversation_history=history_dicts, system_prompt=system_prompt
        )

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

    if len(history) == 0:
        conversation.title = user_message[:50] + ('...' if len(user_message) > 50 else '')

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
        'chart_type_source': chart_type_source,
        'chart_alternatives': chart_alternatives,
        'chart_choices': chart_choices,
        'chart_data': {'labels': labels, 'values': values} if 'labels' in locals() and 'values' in locals() and chart_choices else None,
        'variance_summary': variance_summary,
        'data_profile_warnings': profile_warnings,
        'prompt_intent': intent.category,
    })


@chatbot_bp.route('/chat/chart-choice', methods=['POST'])
@login_required
def choose_chart():
    """Regenerate a selected chart with user-controlled visual settings."""
    data = request.get_json(silent=True) or {}
    chart_type = str(data.get('chart_type', '')).lower().strip()
    allowed_types = {'bar', 'barh', 'line', 'area', 'pie'}
    if chart_type not in allowed_types:
        return jsonify({'error': 'Unsupported chart type'}), 400

    try:
        labels = list(data.get('labels') or [])
        values = [float(v) for v in (data.get('values') or [])]
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid chart data'}), 400

    if not labels or len(labels) != len(values):
        return jsonify({'error': 'Chart labels and values must have equal non-empty lengths'}), 400
    if len(labels) > current_app.config.get('CHART_MAX_POINTS', 60):
        return jsonify({'error': 'Too many chart points'}), 400

    palette = str(data.get('palette') or current_app.config.get('CHART_PALETTE', 'ey'))
    style = str(data.get('style') or current_app.config.get('CHART_STYLE', 'darkgrid'))
    title = str(data.get('title') or 'Chart')[:200]
    xlabel = str(data.get('xlabel') or '')[:100]
    ylabel = str(data.get('ylabel') or '')[:100]
    show_data_labels = bool(data.get('show_data_labels', True))
    show_variance = bool(data.get('show_variance', False))
    show_trend = bool(data.get('show_trend', False))
    noise_variance = data.get('noise_variance')
    noise_variance = float(noise_variance) if noise_variance not in (None, '') else None

    service = ChartService.from_app_config(current_app.config)
    try:
        image = service.render_base64(
            labels, values,
            chart_type=chart_type,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            palette=palette,
            style=style,
            show_data_labels=show_data_labels,
            show_variance=show_variance,
            show_trend=show_trend,
            noise_variance=noise_variance,
        )
    except Exception:
        logger.exception('Selected chart regeneration failed')
        return jsonify({'error': 'Unable to regenerate chart'}), 500

    message_id = data.get('message_id')
    if message_id:
        try:
            message = ChatHistory.query.filter_by(
                id=int(message_id), user_id=current_user.id
            ).first()
            if message:
                message.chart_image = image
                db.session.commit()
        except (TypeError, ValueError):
            pass

    variance = _chart_variance(labels, values)
    return jsonify({
        'success': True,
        'image': image,
        'chart_type': chart_type,
        'variance_summary': variance,
        'settings': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'palette': palette,
            'style': style,
            'show_data_labels': show_data_labels,
            'show_variance': show_variance,
            'show_trend': show_trend,
        },
    })


@chatbot_bp.route('/chat/history/<int:conversation_id>')
@login_required
def get_conversation_history(conversation_id):
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