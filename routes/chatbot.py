"""
Chatbot Routes
==============
Handles the AI chatbot interface, conversation management,
message sending/receiving, chat history operations, and
sales-chart / financial-insight generation from uploaded files.
All endpoints require authentication.
"""

import os
import re
import logging
import json
import statistics
import pandas as pd
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
    find_columns_multi, resolve_column_ambiguity,
    MONTH_COLUMN_ALIASES, VALUE_COLUMN_ALIASES,
)

# --- NEW: feature modules -----------------------------------------------
from services.chart_recommender import recommend_chart_set
from services.chart_customization import (
    parse_chart_customization, apply_month_range_filter, has_customization,
    detect_comparison_columns, parse_comparison_colors, wants_category_on_y_axis,
)
from services.prompt_guidance import classify_prompt_intent, system_prompt_for, EDGE_CASE_INSTRUCTIONS
from services.data_profile_service import build_data_profile, validate_profile
from services.generic_chart_planner import (
    plan_chart, plan_named_metrics, plan_from_ai, fallback_plan, render_plan_base64,
    wants_visual, detect_intent, mentions_any_column, is_time_axis_word, ANALYTIC_INTENTS,
    has_ai_cue, has_time_dimension, fy_availability, mentions_fiscal_year, ChartNotSupported,
)
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
    "already given to you in this conversation. "
    "FINANCIAL YEAR RULE: 'financial year', 'fiscal year', 'FY', 'F.Y.', 'FYE', 'FYTD' and "
    "'fiscal quarter' ALWAYS mean April to March (FY25 = April 2024 to March 2025; Q1 = Apr-Jun, "
    "Q2 = Jul-Sep, Q3 = Oct-Dec, Q4 = Jan-Mar). Never treat them as a calendar year. "
    "NEVER embed images, markdown image syntax (![...](...)), image links, or made-up chart "
    "file names in your answer — the server attaches the real chart itself. If no chart was "
    "attached this turn, say so plainly instead of pretending one exists."
)


_MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')


def _strip_markdown_images(text):
    """
    The language model cannot create images, but it sometimes writes a
    markdown image ("![Quarter-Level Metric Overview](...)") anyway, which
    the browser renders as a broken-image icon next to the REAL chart.
    Remove any such tag, and the now-empty "The chart" style heading that
    often precedes it.
    """
    if not text:
        return text
    cleaned = _MD_IMAGE_RE.sub('', text)
    cleaned = re.sub(r'(?im)^\s*(?:#{1,6}\s*|\*\*)?the chart(?:\*\*)?\s*:?\s*$', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# Column names that only describe WHEN (not WHAT) — a question that merely
# mentions "quarterly" must not be treated as "group by the Quarter column".
_TIME_COLUMN_NAMES = {'quarter', 'qtr', 'month', 'year', 'date', 'period', 'day', 'week', 'month number', 'time'}


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


# Generic words that don't literally match a column name but clearly
# mean "the date/period axis" or "the numeric measure axis" — used when
# resolving an explicit "x axis should be <phrase>" instruction to a real
# dataframe column below.
_AXIS_DATE_SYNONYMS = {'period', 'date', 'time', 'month', 'timeline', 'year', 'quarter'}
_AXIS_VALUE_SYNONYMS = {'amount', 'value', 'revenue', 'sales', 'total', 'figure', 'number', 'count'}


def _find_year_like_column(df):
    """
    Find a column that looks like a discrete fiscal/calendar year field
    (e.g. 'Year' with values 2021-2025), as distinct from a full date/
    month column. Used to group a multi-metric comparison ("EBITDA vs
    Net Profit by year") by year even when the underlying rows are
    monthly and there's no pre-aggregated yearly total in the file.
    """
    for col in df.columns:
        name = str(col).strip().lower()
        if name == 'year' or name.endswith('_year') or name.endswith(' year') or name.startswith('year_'):
            try:
                if 0 < df[col].nunique(dropna=True) <= 50:
                    return col
            except Exception:
                continue
    return None


def _merge_series_by_label(pairs_per_column):
    """
    Merge several {column: [(label, value), ...]} aggregations that were
    computed independently (one groupby call per comparison metric) into
    one shared label axis plus one values list per series, so they can
    be plotted as grouped bars / multiple lines against the same
    categories. A label missing from one series' own aggregation (e.g. a
    year with no data for that metric) is filled with 0.0 rather than
    dropped, so all series stay the same length as `labels`.
    """
    labels = []
    seen = set()
    for pairs in pairs_per_column.values():
        for label, _ in pairs:
            if label not in seen:
                seen.add(label)
                labels.append(label)

    series = []
    for col, pairs in pairs_per_column.items():
        lookup = dict(pairs)
        series.append({'label': str(col), 'values': [float(lookup.get(l, 0.0)) for l in labels]})
    return labels, series


def _build_comparison_chart_data(df, comparison_cols, month_col, group_col):
    """
    Decide how to group a multi-metric comparison request and build the
    shared labels + per-metric series for it.

    Resolution order:
      1. A year-like column (see _find_year_like_column) -> sum each
         metric per year. Matches "compare EBITDA and Net Profit for
         each year" even when the file only has monthly rows.
      2. A date/month column -> each metric's calendar-month total
         (same aggregation monthly_totals() already uses elsewhere in
         this app), for an "X vs Y over time" request.
      3. A categorical group column (region, product, etc.) -> sum each
         metric per category.
      4. Otherwise, no reliable grouping is possible; caller should fall
         back to the existing single-metric chart logic.

    Returns (labels, series, group_label, is_time_series) or
    (None, None, None, None) when no grouping could be resolved.
    """
    year_col = _find_year_like_column(df)
    if year_col:
        pairs_per_col = {}
        for col in comparison_cols:
            lbls, vals = category_sums(df, year_col, col)
            pairs_per_col[col] = list(zip(lbls, vals))
        labels, series = _merge_series_by_label(pairs_per_col)
        if labels:
            return labels, series, year_col, False

    if month_col:
        _fsm = current_app.config.get('FISCAL_YEAR_START_MONTH', 4)
        pairs_per_col = {}
        for col in comparison_cols:
            monthly = monthly_totals(df, month_col, col, fiscal_start_month=_fsm)
            pairs_per_col[col] = [(name, total) for _, name, total in monthly]
        labels, series = _merge_series_by_label(pairs_per_col)
        if labels:
            return labels, series, month_col, True

    if group_col:
        pairs_per_col = {}
        for col in comparison_cols:
            lbls, vals = category_sums(df, group_col, col)
            pairs_per_col[col] = list(zip(lbls, vals))
        labels, series = _merge_series_by_label(pairs_per_col)
        if labels:
            return labels, series, group_col, False

    return None, None, None, None


def _resolve_axis_field(field_text, df, month_col, value_col):
    """
    Resolve a free-text axis instruction ("amount", "period", "region")
    to an actual dataframe column plus what kind of column it is.

    Returns (column_name_or_None, kind) where kind is one of
    'numeric' | 'date' | 'categorical' | None.

    Tries, in order: (1) an exact/substring match against real column
    names (handles the common case where the user just typed the real
    column name, any casing/spacing), (2) generic synonyms ("period"/
    "date" -> whichever date column was already detected elsewhere,
    "amount"/"revenue" -> whichever numeric measure column was already
    detected). This is deliberately conservative — if nothing resolves,
    the caller just falls back to the normal auto-detected chart.
    """
    if not field_text:
        return None, None
    norm_field = re.sub(r'[^a-z0-9]', '', field_text.strip().lower())
    if not norm_field:
        return None, None

    best_col = None
    for col in df.columns:
        norm_col = re.sub(r'[^a-z0-9]', '', str(col).lower())
        if not norm_col:
            continue
        if norm_col == norm_field or norm_field in norm_col or norm_col in norm_field:
            best_col = col
            break

    if best_col is not None:
        try:
            is_numeric = str(df[best_col].dtype).startswith(('int', 'float'))
        except Exception:
            is_numeric = False
        if is_numeric:
            return best_col, 'numeric'
        return best_col, 'categorical'

    field_lower = field_text.strip().lower()
    if field_lower in _AXIS_DATE_SYNONYMS and month_col:
        return month_col, 'date'
    if field_lower in _AXIS_VALUE_SYNONYMS and value_col:
        return value_col, 'numeric'
    return None, None


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
    background_color=None,
    series_color=None,
    color_list=None,
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
            background_color=background_color,
            series_color=series_color,
            color_list=color_list,
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
    background_color=None,
    series_color=None,
    color_list=None,
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
            background_color=background_color,
            series_color=series_color,
            color_list=color_list,
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
    column_ambiguity_notes = []

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
            # A message counts as "wants a chart" if it either uses an
            # explicit chart word ("chart", "plot", "trend", ...) OR it
            # carries a graph customization/specification of its own
            # (a chart type, color, axis rename/mapping, date range,
            # label/variance/trend toggle, noise %). This is what makes
            # a pure follow-up like "put the title as Q1 Sales and color
            # the bars red" generate/regenerate a graph on its own,
            # instead of requiring the user to also type "chart".
            user_wants_chart = (
                wants_chart(user_message, chart_keywords)
                or has_customization(chart_custom)
                or wants_visual(user_message)
            )

            month_col = find_column(df.columns, MONTH_COLUMN_ALIASES) \
                or find_date_column_generic(df, user_message)
            value_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
                or find_numeric_column_generic(df, user_message, exclude=month_col)
            group_col = detect_group_column(df.columns, user_message) \
                or find_group_column_generic(df, user_message, exclude={month_col, value_col})

            # --- Financial-dataset validation: a sheet can legitimately
            # have MORE than one date column ("Order Date" vs "Ship
            # Date") or value column ("Amount" vs "Discount Amount").
            # find_column() above only ever returns the FIRST alias
            # match, so a second candidate would be silently ignored —
            # re-check for that here and, when there genuinely is more
            # than one candidate, resolve it contextually (does the
            # user's own wording name one directly? failing that, ask
            # the AI service to pick using the column names + the
            # question) rather than keep guessing "whichever came
            # first". column_ambiguity_notes collects anything that had
            # to be resolved this way so the answer can be transparent
            # about the pick instead of presenting a guess as certain.
            month_candidates = find_columns_multi(df.columns, MONTH_COLUMN_ALIASES)
            if len(month_candidates) > 1:
                resolved_month, month_was_ambiguous, month_alternatives = resolve_column_ambiguity(
                    month_candidates, user_message, ai_service=ai_service, kind='date',
                )
                if resolved_month:
                    month_col = resolved_month
                if month_was_ambiguous and month_alternatives:
                    column_ambiguity_notes.append(
                        f"Multiple date columns were present ({', '.join(month_candidates)}); "
                        f"used '{month_col}' as the most contextually relevant one for this question."
                    )

            value_candidates = find_columns_multi(df.columns, VALUE_COLUMN_ALIASES)
            if len(value_candidates) > 1:
                resolved_value, value_was_ambiguous, value_alternatives = resolve_column_ambiguity(
                    value_candidates, user_message, ai_service=ai_service, kind='value',
                )
                if resolved_value:
                    value_col = resolved_value
                if value_was_ambiguous and value_alternatives:
                    column_ambiguity_notes.append(
                        f"Multiple value columns were present ({', '.join(value_candidates)}); "
                        f"used '{value_col}' as the most contextually relevant one for this question."
                    )

            # Guard: a column can only ever be matched by NAME (e.g. the
            # user literally typed "amount" while asking for an axis/
            # column), so it's possible to match a continuous NUMERIC
            # column here even though it's meaningless as a grouping
            # category — e.g. "Amount" with ~1,900 distinct values across
            # 2,000 rows isn't a category, it's a measure. Grouping by it
            # produces a near-one-bar-per-row chart with no real
            # aggregation. Reject such a match and let the normal
            # time-series/monthly fallback below handle it instead.
            if group_col is not None:
                try:
                    col_series = df[group_col]
                    is_numeric_col = str(col_series.dtype).startswith(('int', 'float'))
                    n_unique = col_series.nunique(dropna=True)
                    distinct_ratio = n_unique / max(len(col_series), 1)
                    if is_numeric_col and (n_unique > 40 or distinct_ratio > 0.5):
                        group_col = None
                except Exception:
                    pass

            # --- NEW: apply a natural-language date-range filter, if the
            # user asked for one and we have a date column to filter on.
            # Applied to a COPY so downstream deterministic paths (unique
            # values, exact counts above) already ran against the full
            # dataset — only the chart/aggregate paths below see the
            # filtered window.
            # (Skipped for financial-year wording: "Q2 FY25" / "H1 FY25" / "FY25"
            # are resolved to exact Apr-Mar date windows by the chart planner,
            # and a calendar-month pre-filter would silently strip the data.)
            if mentions_fiscal_year(user_message):
                chart_custom.date_range = None   # the planner scopes FY windows exactly
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

            # --- Multi-metric comparison charts ("compare EBITDA and Net
            # Profit for each year", "relationship between marketing
            # spending and revenue"). This must run BEFORE the single-
            # metric explicit-axis/time-series branches below, since
            # those only ever resolve ONE numeric column — a "compare A
            # and B" request would otherwise silently keep just A (or,
            # if neither alias list matched, produce no chart at all).
            if computed_context is None and user_wants_chart:
                comparison_cols = detect_comparison_columns(df, user_message)
                # Exclude whatever column would end up being the GROUPING
                # axis (year/date/category) from the metrics list — a
                # word like "year" in "for each year" can incidentally
                # substring-match a numeric 'Year' column, which must
                # never be treated as one of the metrics being compared.
                _exclude_from_metrics = {
                    c for c in (_find_year_like_column(df), month_col, group_col) if c
                }
                comparison_cols = [c for c in comparison_cols if c not in _exclude_from_metrics]
                if len(comparison_cols) >= 2:
                    comp_labels, comp_series, comp_group_col, comp_is_time_series = (
                        _build_comparison_chart_data(df, comparison_cols, month_col, group_col)
                    )
                    if comp_labels and comp_series:
                        comp_colors = parse_comparison_colors(user_message, comparison_cols)
                        for s in comp_series:
                            if s['label'] in comp_colors:
                                s['color'] = comp_colors[s['label']]

                        if chart_type_override in ('bar', 'barh', 'line'):
                            comp_chart_type = chart_type_override
                        elif chart_custom.chart_type in ('bar', 'barh', 'line'):
                            comp_chart_type = chart_custom.chart_type
                        elif wants_category_on_y_axis(user_message, comp_group_col):
                            comp_chart_type = 'barh'
                        elif comp_is_time_series:
                            comp_chart_type = 'line'
                        else:
                            comp_chart_type = 'bar'

                        comp_title = chart_custom.title or (
                            f"{tabular_filename} — {' vs '.join(comparison_cols)} by {comp_group_col}"
                        )
                        comp_xlabel = chart_custom.xlabel or comp_group_col
                        comp_ylabel = chart_custom.ylabel or 'Amount ($)'

                        chart_service = ChartService.from_app_config(current_app.config)
                        try:
                            chart_image = chart_service.render_comparison_base64(
                                comp_labels, comp_series,
                                chart_type=comp_chart_type,
                                title=comp_title,
                                xlabel=comp_xlabel,
                                ylabel=comp_ylabel,
                                background_color=chart_custom.background_color,
                            )
                            chart_type_source = 'comparison'
                            computed_context = (
                                "AUTHORITATIVE DETERMINISTIC RESULT — DO NOT RECALCULATE:\n"
                                f"Grouped by: {comp_group_col}\n"
                                f"Categories: {', '.join(str(l) for l in comp_labels)}\n"
                                + "\n".join(
                                    f"{s['label']}: {', '.join(f'{v:,.2f}' for v in s['values'])}"
                                    for s in comp_series
                                )
                            )
                            result_response = (
                                f"Here's the {' vs '.join(comparison_cols)} comparison by {comp_group_col}, "
                                f"generated directly from the uploaded data."
                            )
                        except Exception:
                            logger.exception("Comparison chart generation failed for columns: %s", comparison_cols)

            # --- NEW: generic / business-language chart planner --------------
            # Answers questions that name NO column and NO chart type:
            #   1. rule-based planner (13 analytical intents: trend, costs vs
            #      income, efficiency, before/after, customers, ...),
            #   2. named-metric planner ("plot Marketing Expense over time",
            #      "x axis period, y axis revenue") — chronological, not
            #      collapsed into 12 month buckets,
            #   3. AI planner for anything else: the model only PICKS real
            #      columns / aggregation / chart type as JSON; every column is
            #      validated and all numbers are computed in pandas.
            # An explicit category column ("... by region") still goes to the
            # original branches below, unchanged.
            _group_hit = detect_group_column(df.columns, user_message)
            _user_named_category = (
                _group_hit is not None
                and re.sub(r'\s+', ' ', str(_group_hit).strip().lower()) not in _TIME_COLUMN_NAMES
            )
            if _user_named_category:
                # detect_group_column() also returns numeric MEASURE columns
                # ("Revenue", "Marketing Expense") — those are metrics to plot,
                # not a category to group by.
                try:
                    _gs = df[_group_hit]
                    if pd.api.types.is_numeric_dtype(_gs) and (
                        _gs.nunique(dropna=True) > 12
                        or _gs.nunique(dropna=True) / max(len(_gs), 1) > 0.3
                    ):
                        _user_named_category = False
                except Exception:
                    pass
            # An axis instruction only counts as "explicit" when BOTH axes resolve
            # to real columns (the parser also mis-reads phrases like "the
            # relationship between income and spending" as axis fields).
            _axis_x_ok = _resolve_axis_field(chart_custom.x_axis_field, df, month_col, value_col)[0] is not None
            _axis_y_ok = _resolve_axis_field(chart_custom.y_axis_field, df, month_col, value_col)[0] is not None
            _explicit_axes = bool(chart_custom.x_axis_field and chart_custom.y_axis_field and _axis_x_ok and _axis_y_ok)
            _axes_time = any(
                is_time_axis_word(t) for t in (chart_custom.x_axis_field, chart_custom.y_axis_field)
            )
            if _axes_time:
                _explicit_axes = True
            _planner_intent = detect_intent(user_message)
            _generic_question = not mentions_any_column(df, user_message)

            # A specific financial year was named (e.g. "FY26", "FY24 vs FY25")
            # -> check FIRST whether the data actually covers it. FY is 1 Apr -
            # 31 Mar throughout this app (services/insight_service.fiscal_quarter's
            # default). If the requested year(s) simply aren't in the file, say
            # so plainly instead of drawing a chart from whatever data exists.
            _fy_check = fy_availability(df, user_message) if mentions_fiscal_year(user_message) else None
            if _fy_check and computed_context is None and user_wants_chart:
                _missing_fy = [l for l, ok in _fy_check['available'].items() if not ok]
                if _missing_fy and len(_missing_fy) == len(_fy_check['available']):
                    span = (
                        f"The uploaded data only covers {_fy_check['data_min'].strftime('%b %Y')} to "
                        f"{_fy_check['data_max'].strftime('%b %Y')}."
                        if _fy_check['data_min'] is not None else
                        "The uploaded data has no usable date column at all."
                    )
                    computed_context = (
                        "NO CHART COULD BE GENERATED: the user asked about "
                        f"{', '.join(_missing_fy)}, but that period is not in the uploaded data. {span}\n"
                        "Tell the user plainly that this financial year isn't in their data and state "
                        "the date range the data DOES cover. Do not invent figures or a chart."
                    )
            # Column-free, or analytical wording the rules don't model
            # ("which product performs best", "relationship between income and
            # spending") -> the AI planner may choose the columns.
            _ai_eligible = (
                _planner_intent is None
                and (_generic_question or has_ai_cue(user_message))
                and not _explicit_axes
            )

            if (
                computed_context is None
                and user_wants_chart
                and (not _user_named_category or _ai_eligible)
                and (not _explicit_axes or _axes_time)
            ):
                planner_plan = None
                try:
                    if _axes_time:
                        planner_plan = plan_named_metrics(
                            df, user_message,
                            axis_hints=(chart_custom.x_axis_field or '', chart_custom.y_axis_field or ''),
                        )
                    else:
                        if _planner_intent is not None and not _user_named_category:
                            planner_sheets = None
                            if _planner_intent == 'granularity_compare' and tabular_file_type in ('xlsx', 'xls'):
                                try:
                                    planner_sheets = pd.read_excel(tabular_filepath, sheet_name=None)
                                except Exception:
                                    planner_sheets = None
                            planner_plan = plan_chart(
                                df, user_message, sheets=planner_sheets,
                                allow_generic_time=not _user_named_category,
                            )
                        if not planner_plan and not _user_named_category:
                            planner_plan = plan_named_metrics(df, user_message)
                        # Question the rules did not recognise (or an analytical intent
                        # they could not satisfy): let the AI choose the columns
                        # (validated + computed deterministically).
                        if not planner_plan and (
                            (_planner_intent in ANALYTIC_INTENTS and not _user_named_category)
                            or _ai_eligible
                        ):
                            planner_plan = plan_from_ai(df, user_message, ai_service)
                except ChartNotSupported as exc:
                    computed_context = (
                        f"NO CHART COULD BE GENERATED: {exc}\n"
                        f"Available columns: {', '.join(str(c) for c in df.columns)}.\n"
                        "Tell the user plainly, in your own words, that this isn't something the "
                        "uploaded data can show, using the reason above, and suggest what they could "
                        "ask instead. Do not invent data or a chart."
                    )
                except Exception:
                    logger.exception("Generic chart planner failed")

                if planner_plan:
                    try:
                        hint = planner_plan.get('chart_type_hint')
                        use_legacy_renderer = bool(
                            planner_plan.get('legacy_compatible')
                            and (
                                chart_type_override or chart_palette_override or chart_style_override
                                or has_customization(chart_custom)
                                or hint in ('pie', 'area', 'barh')
                            )
                        )
                        if use_legacy_renderer:
                            # Single-series plan + a specific type/colour/title was
                            # requested: keep the existing renderer so every
                            # customization and the alternatives panel still work.
                            labels = planner_plan['labels']
                            values = planner_plan['series'][0]['values']
                            chart_service = ChartService.from_app_config(current_app.config)
                            resolved_chart_type, chart_type_source, chart_alternatives = _resolve_chart_set(
                                chart_type_override or hint, labels, values,
                                is_time_series=bool(planner_plan.get('is_time_series')),
                                group_label=planner_plan.get('ylabel'),
                            )
                            chart_choices = _render_chart_options(
                                chart_service, labels, values, resolved_chart_type, chart_alternatives,
                                chart_custom.title or planner_plan['title'],
                                chart_custom.xlabel or planner_plan.get('xlabel', ''),
                                chart_custom.ylabel or planner_plan.get('ylabel', ''),
                                chart_palette_override, chart_style_override,
                                # dense charts (many periods) are unreadable with a
                                # label on every point unless the user asked for them
                                show_data_labels=(
                                    chart_custom.show_data_labels
                                    if chart_custom.show_data_labels is not None
                                    else len(labels) <= 16
                                ),
                                show_variance=bool(chart_custom.show_variance),
                                show_trend=bool(chart_custom.show_trend),
                                noise_variance=chart_noise_variance,
                                background_color=chart_custom.background_color,
                                series_color=chart_custom.series_color,
                                color_list=chart_custom.color_list,
                            )
                            chart_image = chart_choices[0]['image'] if chart_choices else None
                        else:
                            chart_image = render_plan_base64(
                                planner_plan,
                                background_color=chart_custom.background_color,
                                title=chart_custom.title,
                                xlabel=chart_custom.xlabel,
                                ylabel=chart_custom.ylabel,
                                series_color=chart_custom.series_color,
                            )
                            chart_type_source = f"planner:{planner_plan['intent']}"
                            chart_choices = []
                    except Exception:
                        logger.exception("Rendering planner chart failed (%s)", planner_plan.get('intent'))
                        chart_image = None

                    if chart_image:
                        computed_context = (
                            planner_plan['facts']
                            + "\nDescribe what this chart shows in 3-6 sentences using ONLY the figures above; "
                              "state any assumption listed above. Do not embed images or links — "
                              "the chart is attached automatically."
                        )
                        if planner_plan.get('reason'):
                            computed_context += f"\nChart design note: {planner_plan['reason']}"
                        variance_summary = None
                _no_time_dim = _planner_intent == 'generic_time' and not has_time_dimension(df)
                if (not chart_image and not _explicit_axes and not _user_named_category
                        and (_planner_intent in ANALYTIC_INTENTS or _no_time_dim)):
                    # The file cannot support this question (e.g. "workforce" with
                    # no headcount column, or "costs vs income" with no cost
                    # columns). Do NOT fall through to the old "count rows of the
                    # first category" chart — that produced the bogus
                    # "Quarter: 15/15/15/15" chart.
                    computed_context = (
                        "NO CHART COULD BE GENERATED for this request because the uploaded "
                        "data does not contain the columns/time dimension needed "
                        f"(request type: {_planner_intent}"
                        f"{'; the file has no date/time column' if _no_time_dim else ''}).\n"
                        f"Available columns: {', '.join(str(c) for c in df.columns)}.\n"
                        "Tell the user plainly which kind of column is missing and suggest "
                        "what they could ask instead. Do not invent data or images."
                    )

            message_lower = user_message.lower()
            time_series_request = any(keyword in message_lower for keyword in (
                'sales', 'revenue', 'income', 'profit', 'expense',
                'amount', 'financial', 'month', 'monthly', 'quarter',
                'q1', 'q2', 'q3', 'q4', 'qtr', 'date', 'period',
                'over time', 'trend', 'timeline', 'yearly', 'annual', 'growth'
            )) and not explicit_group_match

            axis_x_col, axis_x_kind = _resolve_axis_field(chart_custom.x_axis_field, df, month_col, value_col)
            axis_y_col, axis_y_kind = _resolve_axis_field(chart_custom.y_axis_field, df, month_col, value_col)

            # Figure out which resolved axis (if any) is the numeric
            # measure and which is the category/date to group by — the
            # user may name either one on x or y, so this has to work
            # both ways round ("x=amount,y=period" AND "x=period,y=amount").
            explicit_measure_col = explicit_category_col = explicit_category_kind = None
            explicit_measure_on_x = None
            if axis_x_kind == 'numeric' and axis_y_col:
                explicit_measure_col, explicit_category_col = axis_x_col, axis_y_col
                explicit_category_kind, explicit_measure_on_x = axis_y_kind, True
            elif axis_y_kind == 'numeric' and axis_x_col:
                explicit_measure_col, explicit_category_col = axis_y_col, axis_x_col
                explicit_category_kind, explicit_measure_on_x = axis_x_kind, False

            # Guard: "x axis period, y axis revenue" can resolve BOTH axes to the
            # same column (e.g. the measure column), which used to crash with
            # "TypeError: arg must be a list, tuple, 1-d array, or Series".
            if explicit_measure_col is not None and explicit_measure_col == explicit_category_col:
                explicit_measure_col = explicit_category_col = None
                explicit_category_kind = explicit_measure_on_x = None

            if computed_context is None and explicit_measure_col and explicit_category_col and user_wants_chart:
                # An explicit "x axis should be X, y axis should be Y"
                # instruction is the strongest possible signal — honor it
                # directly instead of falling through to the auto-detected
                # group/time-series branches below. Still aggregate
                # properly (monthly totals for a date column, sum-by-
                # category otherwise) rather than plotting raw per-row
                # values, which is what produced a one-bar-per-row chart
                # before this branch existed.
                if explicit_category_kind == 'date':
                    monthly = monthly_totals(
                        df, explicit_category_col, explicit_measure_col,
                        fiscal_start_month=current_app.config.get('FISCAL_YEAR_START_MONTH', 4),
                    )
                    labels = [name for _, name, _ in monthly]
                    values = [total for _, _, total in monthly]
                else:
                    labels, values = category_aggregate(df, explicit_category_col, explicit_measure_col, 'sum')

                if labels:
                    computed_context = build_ai_context(
                        categories=(labels, values), category_label=explicit_category_col,
                    )
                    default_title = f'{tabular_filename} — {explicit_measure_col} by {explicit_category_col}'
                    title = chart_custom.title or default_title
                    # NOTE: chart_service.render_png_bytes() internally
                    # swaps labels for chart_type='barh' (it always does
                    # ax.set_xlabel(ylabel)/set_ylabel(xlabel) for barh so
                    # the category names land on the correct axis) — so
                    # callers must ALWAYS pass xlabel=category,
                    # ylabel=measure regardless of visual orientation.
                    # explicit_measure_on_x only decides chart_type below,
                    # never label assignment, or the labels end up
                    # double-swapped and backwards.
                    xlabel = chart_custom.xlabel or explicit_category_col
                    ylabel = chart_custom.ylabel or explicit_measure_col
                    # Orientation follows the user's explicit axis request:
                    # measure on x -> horizontal bar (or keep a line/area
                    # override if they also asked for one); measure on y
                    # with a date category -> a normal line trend.
                    if chart_custom.chart_type:
                        resolved_chart_type = chart_custom.chart_type
                    elif explicit_measure_on_x:
                        resolved_chart_type = 'barh'
                    elif explicit_category_kind == 'date':
                        resolved_chart_type = 'line'
                    else:
                        resolved_chart_type = 'bar'
                    chart_service = ChartService.from_app_config(current_app.config)
                    _, chart_type_source, chart_alternatives = _resolve_chart_set(
                        resolved_chart_type, labels, values,
                        is_time_series=(explicit_category_kind == 'date'), group_label=explicit_category_col,
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
                        background_color=chart_custom.background_color,
                        series_color=chart_custom.series_color,
                        color_list=chart_custom.color_list,
                    )
                    chart_image = chart_choices[0]['image'] if chart_choices else None
                    variance_summary = _chart_variance(labels, values)

            elif computed_context is None and group_col and not time_series_request:
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
                            background_color=chart_custom.background_color,
                            series_color=chart_custom.series_color,
                            color_list=chart_custom.color_list,
                        )
                        chart_image = (
                            chart_choices[0]['image'] if chart_choices else None
                        )
                        variance_summary = _chart_variance(labels, values)

            elif computed_context is None and month_col and value_col:
                monthly = monthly_totals(df, month_col, value_col, fiscal_start_month=fiscal_start)
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
                            background_color=chart_custom.background_color,
                            series_color=chart_custom.series_color,
                            color_list=chart_custom.color_list,
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
                            background_color=chart_custom.background_color,
                            series_color=chart_custom.series_color,
                            color_list=chart_custom.color_list,
                        )
                        chart_image = (
                            chart_choices[0]['image'] if chart_choices else None
                        )
                        variance_summary = _chart_variance(labels, values)

            # --- NEW: never end a chart request with no chart at all. If the
            # user clearly asked for a chart (chart word / customization) but
            # none of the branches above produced one, draw the most sensible
            # best-effort chart from the real data (trend of the main measure,
            # or the main measure by its main category) and say what was assumed.
            _chart_blocked = str(computed_context or '').startswith('NO CHART COULD BE GENERATED')
            if user_wants_chart and not chart_image and result_response is None and not _chart_blocked:
                try:
                    try:
                        fb_plan = plan_from_ai(df, user_message, ai_service)
                    except ChartNotSupported:
                        fb_plan = None
                    fb_plan = fb_plan or fallback_plan(df, user_message)
                    if fb_plan:
                        fb_image = render_plan_base64(
                            fb_plan,
                            background_color=chart_custom.background_color,
                            title=chart_custom.title,
                            xlabel=chart_custom.xlabel,
                            ylabel=chart_custom.ylabel,
                            series_color=chart_custom.series_color,
                        )
                        chart_image = fb_image
                        chart_type_source = 'planner:fallback'
                        chart_choices = []
                        fb_context = (
                            fb_plan['facts']
                            + "\nDescribe what this chart shows using ONLY the figures above and "
                              "state the assumption. Do not embed images or links — the chart is "
                              "attached automatically."
                        )
                        computed_context = (
                            f"{computed_context}\n\n{fb_context}" if computed_context else fb_context
                        )
                except Exception:
                    logger.exception("Fallback chart failed")

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

    # Tell the model the truth about the chart so it never invents one.
    if chart_image:
        prompt_text += (
            "\n\nCHART STATUS: a real chart image HAS been generated by the server and is shown "
            "to the user below your answer. Explain what it shows. Do not embed any image."
        )
    elif tabular_filepath and (wants_chart(user_message, current_app.config.get('CHART_TRIGGER_KEYWORDS'))
                               or wants_visual(user_message)):
        prompt_text += (
            "\n\nCHART STATUS: NO chart could be generated for this request. Say so plainly, "
            "explain why in one sentence (e.g. the needed column is missing), and suggest a "
            "specific alternative. Never embed images, image links, or describe a chart as if it existed."
        )

    if column_ambiguity_notes:
        prompt_text += (
            "\n\nCOLUMN SELECTION NOTES (mention this to the user briefly if relevant):\n"
            + "\n".join(f"- {note}" for note in column_ambiguity_notes)
        )

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

    # The model can't draw images; strip any fake markdown image it wrote.
    result['response'] = _strip_markdown_images(result.get('response'))

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
    allowed_types = {'bar', 'barh', 'line', 'area', 'pie', 'histogram'}
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
    background_color = data.get('background_color') or None
    series_color = data.get('series_color') or None
    color_list = data.get('color_list') or None
    if color_list is not None and not isinstance(color_list, list):
        color_list = None

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
            background_color=background_color,
            series_color=series_color,
            color_list=color_list,
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
