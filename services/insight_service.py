"""
Insight Service Module
=======================
Deterministic financial/data computations over uploaded tabular data
(CSV/XLSX). Numbers are computed here in Python — never left to the
language model to "do the math" — so totals are guaranteed accurate and
the model is never put in a position where it has to fake running code.

Two chart paths are supported:
  1. Month/sales time-series data -> monthly_totals() / quarterly_totals()
  2. Any other categorical column the user names in their message
     (e.g. "visualize students per venue") -> category_counts() /
     category_sums(), via detect_group_column().

CHANGELOG (this revision):
  - Added suggest_chart_type(): decides bar/barh/line/area/pie for a given
    dataset using a deterministic heuristic, with an AI call (via the
    passed-in ai_service) used to make the final pick on every chart
    request. The heuristic always runs first and is used as a guaranteed
    fallback if the AI call fails or returns something unusable, so chart
    generation is never blocked by an AI/network failure.
"""

import logging

import pandas as pd
import re

logger = logging.getLogger(__name__)

# Column-name aliases used to auto-detect month/value columns for
# time-series (month/quarter) charts specifically.
MONTH_COLUMN_ALIASES = ['month', 'period', 'date', 'mon']
VALUE_COLUMN_ALIASES = ['sales', 'revenue', 'amount', 'total', 'value', 'income']

MONTH_NAME_TO_NUM = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9, 'oct': 10,
    'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
}

MONTH_NUM_TO_SHORT = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

CHART_KEYWORDS_DEFAULT = [
    'graph', 'chart', 'plot', 'visualize', 'visualise', 'trend',
    'bar chart', 'line chart', 'pie chart', 'draw', 'diagram', 'picture',
    'breakdown', 'distribution',
]
QUARTER_KEYWORDS = ['quarter', 'q1', 'q2', 'q3', 'q4', 'qtr']

# Minimum column-name length considered when scanning the user's message
# for a mentioned column — avoids short/generic names (e.g. "id") matching
# by accident.
_GROUP_COLUMN_MIN_LENGTH = 3

# Chart types the renderer (services/chart_service.py) supports. Kept here
# too so suggest_chart_type() has a single source of truth to validate
# against without importing chart_service (avoids a circular import risk).
VALID_CHART_TYPES = ('bar', 'barh', 'line', 'area', 'pie')



def suggest_chart_type(
    labels,
    values,
    is_time_series=False,
    ai_service=None,
    group_label=None,
):
    """
    Choose a chart type deterministically, with an optional AI refinement.

    The heuristic always provides a safe fallback. AI is only allowed to
    choose from VALID_CHART_TYPES and cannot change the underlying data.
    """
    labels = list(labels or [])
    values = list(values or [])
    count = len(labels)

    # Guaranteed deterministic fallback.
    if count == 0:
        heuristic = "bar"
    elif is_time_series:
        heuristic = "line"
    elif count > 12:
        heuristic = "barh"
    elif count <= 6:
        heuristic = "bar"
    else:
        heuristic = "bar"

    if ai_service is None:
        return heuristic, "heuristic"

    try:
        prompt = (
            "Choose the best chart type for this dataset. "
            f"Chart types allowed: {', '.join(VALID_CHART_TYPES)}. "
            f"Time series: {bool(is_time_series)}. "
            f"Number of categories/points: {count}. "
            f"Group/metric: {group_label or 'unknown'}. "
            "Reply with ONLY one chart type."
        )
        response = ai_service.get_chat_response(
            prompt,
            conversation_history=[],
            system_prompt=(
                "You are a chart-selection assistant. "
                "Return exactly one value from: bar, barh, line, area, pie."
            ),
        )
        candidate = str(response or "").strip().lower()
        candidate = re.sub(r"[^a-z]", "", candidate)
        if candidate in VALID_CHART_TYPES:
            # Pie is only sensible for a small categorical breakdown.
            if candidate == "pie" and (is_time_series or count > 8):
                return heuristic, "heuristic"
            if candidate in ("line", "area") and not is_time_series:
                return heuristic, "heuristic"
            return candidate, "ai"
    except Exception:
        logger.exception("AI chart-type suggestion failed; using heuristic.")

    return heuristic, "heuristic"

def find_column(columns, aliases):
    """Return the first column whose lowercased name contains any alias."""
    for col in columns:
        low = str(col).strip().lower()
        if any(alias in low for alias in aliases):
            return col
    return None


def find_numeric_column_generic(df, message=None, exclude=None):
    """
    Generic (domain-agnostic) numeric/currency/stock column resolver.

    Use this as a FALLBACK when find_column(df.columns, VALUE_COLUMN_ALIASES)
    returns None — i.e. the sheet's numeric column isn't named 'sales',
    'revenue', 'amount', etc. Works on any spreadsheet by classifying
    columns via schema_service instead of matching against an English
    word list, so it generalizes to student marksheets, inventory sheets,
    stock data, or anything else.

    Returns:
        str or None
    """
    from services.schema_service import classify_dataframe, resolve_numeric_column
    schema = classify_dataframe(df)
    return resolve_numeric_column(df, schema, message=message, exclude=exclude)


def find_date_column_generic(df, message=None):
    """
    Generic (domain-agnostic) date/time column resolver. Fallback for when
    find_column(df.columns, MONTH_COLUMN_ALIASES) finds nothing because the
    date column isn't named 'month'/'period'/'date'.

    Returns:
        str or None
    """
    from services.schema_service import classify_dataframe, resolve_date_column
    schema = classify_dataframe(df)
    return resolve_date_column(df, schema, message=message)


def find_group_column_generic(df, message=None, exclude=None):
    """
    Generic (domain-agnostic) categorical column resolver. Fallback for
    when detect_group_column() finds no name-based match at all (e.g. the
    question is vague — "break this down for me" — with no column
    reference).

    Returns:
        str or None
    """
    from services.schema_service import classify_dataframe, resolve_group_column
    schema = classify_dataframe(df)
    return resolve_group_column(df, schema, message=message, exclude=exclude)


def _normalize_text(value):
    """Normalize punctuation/spacing so 'Test-time (hh:mm AM/PM)' can
    match a question such as 'students per test-time'."""
    text = str(value or '').lower()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def detect_group_column(columns, message):
    """Detect the categorical column the user is asking about.

    Handles exact column mentions, punctuation variants, and common natural
    language forms such as:
      - 'how many students per campus'
      - 'students in each test-time'
      - 'count by venue'
      - 'number of students for each batch'

    This function is deliberately deterministic; the LLM is never asked to
    guess which spreadsheet column should be counted.
    """
    if not message:
        return None

    msg = _normalize_text(message)
    if not msg:
        return None

    # Strongest match: normalized column name appears in the normalized
    # question. This fixes names such as "Test-time (hh:mm AM/PM)".
    candidates = []
    for col in columns:
        normalized_col = _normalize_text(col)
        if len(normalized_col) < _GROUP_COLUMN_MIN_LENGTH:
            continue

        score = 0
        if normalized_col in msg:
            score = 100 + len(normalized_col)
        else:
            # Also match the meaningful part before a parenthetical
            # description, e.g. "test-time" -> "test time".
            words = normalized_col.split()
            meaningful = ' '.join(words[:3])
            if len(meaningful) >= _GROUP_COLUMN_MIN_LENGTH and meaningful in msg:
                score = 80 + len(meaningful)

        if score:
            candidates.append((score, len(normalized_col), col))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    # Natural-language fallback: identify the phrase after "each/per/by".
    # Only accept a column when at least one significant column word appears
    # in that phrase, preventing arbitrary columns from being selected.
    for phrase in re.findall(r'\b(?:each|per|by|for each)\s+([a-z0-9][a-z0-9 ]{1,50})', msg):
        phrase = phrase.strip()
        for col in sorted(columns, key=lambda c: -len(_normalize_text(c))):
            normalized_col = _normalize_text(col)
            col_words = [w for w in normalized_col.split() if len(w) >= 3]
            if col_words and any(w in phrase.split() for w in col_words):
                return col

    # Common conversational aliases for spreadsheets. Only use these when
    # the workbook contains exactly one matching column.
    aliases = {
        'campus': ['campus', 'college', 'institute'],
        'venue': ['venue', 'location', 'centre', 'center'],
        'time': ['time', 'test time', 'slot'],
        'batch': ['batch', 'group'],
        'department': ['department', 'dept'],
        'category': ['category', 'type'],
    }
    for alias, terms in aliases.items():
        if any(re.search(rf'\b{re.escape(term)}\b', msg) for term in terms):
            matches = []
            for col in columns:
                normalized_col = _normalize_text(col)
                if any(term in normalized_col for term in terms):
                    matches.append(col)
            if len(matches) == 1:
                return matches[0]

    return None


def month_to_num(value):
    """Best-effort parse of a month cell (name, number, or date) into 1-12."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        m = int(value)
        return m if 1 <= m <= 12 else None
    text = str(value).strip().lower()
    if text in MONTH_NAME_TO_NUM:
        return MONTH_NAME_TO_NUM[text]
    parsed = pd.to_datetime(text, errors='coerce')
    if pd.notna(parsed):
        return parsed.month
    return None


def fiscal_quarter(month_num, fiscal_start_month=4):
    """
    Map a calendar month (1-12) to a quarter label, honoring a configurable
    fiscal-year start month (4 = Indian-style Q1=Apr-Jun, the default here;
    1 = calendar Q1=Jan-Mar).
    """
    offset = (month_num - fiscal_start_month) % 12
    return f'Q{offset // 3 + 1}'


def load_dataframe(filepath, file_type):
    """Load a CSV/XLSX/XLS file as a DataFrame. Returns None on failure."""
    try:
        if file_type == 'csv':
            return pd.read_csv(filepath)
        if file_type in ('xlsx', 'xls'):
            return pd.read_excel(filepath)
    except Exception as e:
        logger.error(f"Could not load {filepath} for insight generation: {e}")
    return None


def monthly_totals(df, month_col, value_col):
    """Return a list of (month_num, month_label, total) sorted by month."""
    work = df[[month_col, value_col]].copy()
    work['_month_num'] = work[month_col].apply(month_to_num)
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
    work = work.dropna(subset=['_month_num', value_col])
    if work.empty:
        return []
    grouped = work.groupby('_month_num')[value_col].sum().sort_index()
    return [(int(m), MONTH_NUM_TO_SHORT[int(m)], float(v)) for m, v in grouped.items()]


def quarterly_totals(df, month_col, value_col, fiscal_start_month=4):
    """Return an ordered dict of quarter label -> total (Q1..Q4, only quarters present)."""
    monthly = monthly_totals(df, month_col, value_col)
    quarters = {}
    for month_num, _, total in monthly:
        q = fiscal_quarter(month_num, fiscal_start_month)
        quarters[q] = quarters.get(q, 0) + total
    ordered = [f'Q{i}' for i in range(1, 5) if f'Q{i}' in quarters]
    return {q: quarters[q] for q in ordered}


def category_counts(df, group_col):
    """
    Count occurrences of each distinct value in a categorical column —
    e.g. "number of students assigned to each venue" is exactly a row
    count per distinct Venue value.
    """
    series = df[group_col].dropna().astype(str).str.strip()
    series = series[series != '']
    counts = series.value_counts().sort_index()
    return list(counts.index), [int(v) for v in counts.values]


def category_sums(df, group_col, value_col):
    """Sum a numeric column grouped by a categorical column."""
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work.dropna(subset=[value_col])
    work = work[work[group_col] != '']
    if work.empty:
        return [], []
    grouped = work.groupby(group_col)[value_col].sum().sort_index()
    return list(grouped.index), [float(v) for v in grouped.values]


def category_aggregate(df, group_col, value_col, operation='sum'):
    """Aggregate a numeric column by category deterministically."""
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work.dropna(subset=[value_col])
    work = work[work[group_col] != '']
    if work.empty:
        return [], []

    grouped = work.groupby(group_col)[value_col]
    if operation == 'mean':
        result = grouped.mean()
    elif operation == 'min':
        result = grouped.min()
    elif operation == 'max':
        result = grouped.max()
    elif operation == 'count':
        result = grouped.count()
    else:
        result = grouped.sum()

    result = result.sort_index()
    return list(result.index), [float(v) for v in result.values]


def requested_aggregation(message):
    """
    Return the deterministic aggregation requested by a natural-language
    question: 'mean', 'min', 'max', 'sum', or 'count' (default).

    Uses whole-word regex matching (not bare substring checks) so this
    doesn't false-positive on unrelated words containing 'sum' (e.g.
    'summary', 'consumption', 'assume') and so 'total'/'sum' are
    recognized generically — for ANY metric name ('total cost', 'total
    marks', 'total attendance'), not just a fixed list of finance phrases
    like 'total sales'/'total revenue'. This is what makes aggregation
    detection work the same way regardless of the spreadsheet's domain.
    """
    text = _normalize_text(message)

    def has_word(*words):
        return any(re.search(rf'\b{re.escape(w)}\b', text) for w in words)

    if has_word('average', 'avg', 'mean'):
        return 'mean'
    if has_word('minimum', 'lowest', 'smallest', 'min'):
        return 'min'
    if has_word('maximum', 'highest', 'largest', 'max'):
        return 'max'

    # 'total number of X' / 'count of X' / 'how many' mean COUNT, not sum
    # — check this before the generic 'total' trigger below, since
    # 'total' alone is ambiguous ("total students" = count of rows,
    # "total marks" = sum of a numeric column).
    if re.search(r'\btotal number\b', text) or has_word('how many', 'count', 'number of'):
        return 'count'

    if has_word('sum', 'total', 'sales by', 'revenue by', 'amount by'):
        return 'sum'

    return 'count'


def wants_chart(message, keywords=None):
    """Cheap, deterministic intent check — no AI round trip needed for this."""
    text = (message or '').lower()
    return any(kw in text for kw in (keywords or CHART_KEYWORDS_DEFAULT))


def wants_quarterly_breakdown(message):
    text = (message or '').lower()
    return any(kw in text for kw in QUARTER_KEYWORDS)


def is_unique_value_query(message):
    """Detect requests to list distinct/unique values from a column."""
    text = (message or '').lower()
    return any(term in text for term in (
        'different', 'distinct', 'unique', 'list all', 'list the',
        'what are the', 'which are the', 'which timings', 'all timings',
        'what timings', 'what time slots', 'time slots'
    ))


def unique_column_values(df, column):
    """Return distinct values from the complete DataFrame, preserving first-seen order."""
    values = []
    seen = set()
    for raw in df[column].tolist():
        if pd.isna(raw):
            continue
        normalized = _normalize_time(raw) if 'time' in str(column).lower() else str(raw).strip()
        if normalized is None or normalized == '':
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            values.append(normalized)
    return values


def is_count_query(message):
    """Return True for deterministic row/category counting questions."""
    text = (message or '').lower()
    count_terms = (
        'how many', 'number of', 'count of', 'count ', 'total number',
        'how much students', 'how many students'
    )
    return any(term in text for term in count_terms)


def _normalized_text(value):
    """Normalize labels such as '09:00 AM', '9:00 AM', and '9 AM'."""
    if pd.isna(value):
        return ''
    text = str(value).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('.', '')
    return text


def _normalize_time(value):
    """Explicitly normalize Excel time strings without dateutil inference."""
    if pd.isna(value):
        return None

    if hasattr(value, "hour") and hasattr(value, "minute") and not isinstance(value, str):
        try:
            hour = int(value.hour)
            minute = int(value.minute)
            suffix = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            return f"{display_hour}:{minute:02d} {suffix}"
        except Exception:
            pass

    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(".", ":")
    text = re.sub(r"\s*:\s*", ":", text)

    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)

    if 1 <= hour <= 12 and 0 <= minute <= 59:
        return f"{hour}:{minute:02d} {suffix}"
    return None


def find_column_for_question(columns, message):
    """Prefer a column whose normalized name/keywords clearly match the question."""
    q = _normalized_text(message)
    normalized = {col: _normalized_text(col) for col in columns}

    # Strong semantic matches first.
    if 'test' in q and 'time' in q:
        for col, name in normalized.items():
            if 'test' in name and 'time' in name:
                return col

    for keyword in ('time', 'campus', 'venue', 'department', 'college', 'location'):
        if keyword in q:
            candidates = [col for col, name in normalized.items() if keyword in name]
            if candidates:
                return sorted(candidates, key=lambda c: len(str(c)))[0]

    # Exact/substring mention of a column name.
    for col in sorted(columns, key=lambda c: -len(str(c))):
        name = normalized[col]
        if len(name) >= 3 and name in q:
            return col

    return None


def find_time_column_by_value(df, requested_time):
    """
    Find a column whose VALUES normalize-match requested_time, without
    requiring the column name itself to appear in the question.

    This exists because find_column_for_question() is name-based and
    requires the literal word "time" (or "test"+"time") somewhere in the
    question. Natural phrasing like "how many students have a test at
    9 AM" never says the word "time" at all — it only states a time value
    — so name-based matching silently fails and the deterministic count
    path never fires, leaving the AI with no real data (which previously
    caused it to fabricate an answer). Scanning by value instead of by
    column name closes that gap.

    Args:
        df: pandas DataFrame to search.
        requested_time: canonical 'H:MM AM/PM' string, e.g. from
            extract_time_from_question().

    Returns:
        The first matching column name, or None if requested_time is falsy
        or no column's values match it.
    """
    if not requested_time:
        return None
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        for raw in series.tolist():
            if _normalize_time(raw) == requested_time:
                return col
    return None


def exact_category_count(df, group_col, requested_value=None, time_value=False):
    """
    Deterministically count rows for an exact requested category/value.
    Returns (matched_label, count, all_counts).
    """
    if time_value and requested_value:
        wanted = _normalize_time(requested_value)
        if wanted:
            normalized = df[group_col].map(_normalize_time)
            mask = normalized == wanted
            return requested_value, int(mask.sum()), None

    series = df[group_col].dropna().astype(str).str.strip()
    series = series[series != '']
    counts = series.value_counts()
    if requested_value:
        wanted = _normalized_text(requested_value)
        for label, count in counts.items():
            if _normalized_text(label) == wanted:
                return label, int(count), None
        # Try containment only after exact normalized matching.
        matches = [(label, count) for label, count in counts.items()
                   if wanted and wanted in _normalized_text(label)]
        if len(matches) == 1:
            return matches[0][0], int(matches[0][1]), None
        return requested_value, 0, None

    counts = counts.sort_index()
    return None, None, (list(counts.index), [int(v) for v in counts.values])


def extract_time_from_question(message):
    """Extract 9 AM / 9:00 AM / 09:00 AM from a natural-language question."""
    text = (message or '').lower()
    m = re.search(r'\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b', text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = m.group(3).replace('.', '').upper()
    if 1 <= hour <= 12 and 0 <= minute <= 59:
        return f'{hour}:{minute:02d} {suffix}'
    return None


def _safe_float(value):
    """Convert spreadsheet numeric values safely for display."""
    if value is None:
        return None
    if isinstance(value, str):
        value = (value.replace(',', '').replace('$', '').replace('₹', '')
                      .replace('€', '').replace('£', '').strip())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_ai_context(value_match=None, monthly=None, quarterly=None,
                     categories=None, category_label=None,
                     fiscal_start_month=4):
    """Build authoritative AI context without crashing on string numerics."""
    lines = [
        "Pre-computed figures (authoritative — do not recalculate, estimate, "
        "or write/simulate code to reproduce these):"
    ]

    if fiscal_start_month != 1 and (monthly or quarterly):
        try:
            start_name = MONTH_NUM_TO_SHORT[int(fiscal_start_month)]
            lines.append(
                f"Fiscal year starts in {start_name}; Q1 begins in that month."
            )
        except (KeyError, TypeError, ValueError):
            pass

    if monthly:
        lines.append("Monthly totals:")
        for item in monthly:
            try:
                if len(item) >= 3:
                    _, name, total = item
                else:
                    name, total = item
                numeric = _safe_float(total)
                if numeric is not None:
                    lines.append(f"  - {name}: {numeric:,.2f}")
                else:
                    lines.append(f"  - {name}: {total}")
            except (TypeError, ValueError, IndexError):
                lines.append(f"  - {item}")

    if quarterly:
        lines.append("Quarterly totals:")
        items = quarterly.items() if isinstance(quarterly, dict) else quarterly
        for item in items:
            try:
                q, total = item
                numeric = _safe_float(total)
                if numeric is not None:
                    lines.append(f"  - {q}: {numeric:,.2f}")
                else:
                    lines.append(f"  - {q}: {total}")
            except (TypeError, ValueError, IndexError):
                lines.append(f"  - {item}")

    if categories:
        try:
            labels, values = categories
            lines.append(f"{category_label or 'Category'} breakdown:")
            for label, value in zip(labels, values):
                numeric = _safe_float(value)
                if numeric is not None:
                    lines.append(f"  - {label}: {numeric:,.2f}")
                else:
                    lines.append(f"  - {label}: {value}")
        except (TypeError, ValueError):
            pass

    if value_match:
        if isinstance(value_match, dict):
            lines.extend(f"{k}: {v}" for k, v in value_match.items())
        else:
            lines.append(str(value_match))

    return "\n".join(lines)

def build_no_match_context(columns):
    return (
        "A real tabular dataset is available, but no calculation could be "
        "confidently matched to the user's wording. Do not invent a value. "
        f"Available columns: {', '.join(map(str, columns))}"
    )


def detect_value_query(df, message):
    if df is None or df.empty:
        return None
    text = _normalize_text(message)
    columns = list(df.columns)

    # Prefer the longest exact normalized column-name mention.
    mentioned = None
    for col in sorted(columns, key=lambda c: len(_normalize_text(c)), reverse=True):
        norm = _normalize_text(col)
        if len(norm) >= 3 and norm in text:
            mentioned = col
            break

    if mentioned is not None:
        series = df[mentioned]
        # Exact value lookup.
        for raw in series.dropna().tolist():
            value = str(raw).strip()
            if len(value) >= 2 and _normalize_text(value) in text:
                mask = series.astype(str).str.strip().str.casefold() == value.casefold()
                return {
                    "Column": str(mentioned),
                    "Requested value": value,
                    "Count": f"{int(mask.sum()):,}",
                }

    if is_count_query(message):
        group = detect_group_column(df.columns, message)
        if group:
            counts = df[group].dropna().astype(str).str.strip().value_counts()
            if len(counts):
                return {
                    "Column": str(group),
                    "Aggregation": "count",
                    "Total rows": f"{int(counts.sum()):,}",
                    "Top categories": "; ".join(
                        f"{k}: {int(v):,}" for k, v in counts.head(10).items()
                    ),
                }
    return None


def looks_like_data_question(message):
    text = (message or "").lower()
    terms = (
        "data", "dataset", "excel", "spreadsheet", "column", "row", "value",
        "total", "sum", "average", "mean", "highest", "lowest", "maximum",
        "minimum", "count", "how many", "sales", "revenue", "cost", "price",
        "marks", "students", "campus", "venue", "batch", "trend", "chart",
        "graph", "plot", "visualize", "breakdown", "distribution", "quarter",
        "month", "year",
    )
    return any(x in text for x in terms)


def looks_like_clarifying_reply(response):
    text = (response or "").lower()
    return any(x in text for x in (
        "which column", "please specify", "could you specify",
        "what column", "which metric", "which category",
    ))


def analyze_file_query(filepath, file_type, question, ai_service=None, chart_config=None):
    """
    One-shot helper for routes/charts.py's standalone `/api/charts/generate`
    endpoint: load a CSV/Excel file, deterministically figure out what the
    question is asking for (time-series vs. categorical breakdown vs. a
    single numeric column), compute it with the same pandas logic
    /chat/send uses, render it as a chart, and return everything the route
    needs in one dict.

    Args:
        filepath: path to the CSV/XLSX/XLS file on disk.
        file_type: 'csv', 'xlsx', or 'xls'.
        question: natural-language chart request, e.g. "monthly revenue trend".
        ai_service: optional AIService instance, passed through to chart
            type selection for an AI-assisted pick (falls back to the
            deterministic heuristic if omitted or if the call fails).
        chart_config: optional dict of ChartService kwargs (style, palette,
            context, dpi, figsize) to override the defaults.

    Returns:
        dict: {
            'chart': base64 PNG data URI,
            'insights': human-readable text summary of the computed figures,
            'metric_column': the numeric column that was charted (or None),
            'group': the column used for the x-axis/grouping (or None),
            'labels': list of x-axis labels,
            'values': list of numeric values,
        }

    Raises:
        ValueError: if the file can't be read or no chartable data could
            be found/computed for the question.
    """
    df = load_dataframe(filepath, file_type)
    if df is None or df.empty:
        raise ValueError("Could not read any data from this file.")

    from services.chart_service import ChartService

    month_col = find_column(df.columns, MONTH_COLUMN_ALIASES) \
        or find_date_column_generic(df, question)
    value_col = find_column(df.columns, VALUE_COLUMN_ALIASES) \
        or find_numeric_column_generic(df, question, exclude=month_col)
    group_col = detect_group_column(df.columns, question) \
        or find_group_column_generic(df, question, exclude={month_col, value_col})

    is_time_series = False
    metric_column = value_col
    group = None
    labels, values = [], []

    if month_col and value_col:
        fiscal_start = 4
        monthly = monthly_totals(df, month_col, value_col)
        if not monthly:
            raise ValueError(
                f"No valid data found in '{month_col}'/'{value_col}' to chart."
            )
        if wants_quarterly_breakdown(question):
            quarterly = quarterly_totals(df, month_col, value_col, fiscal_start)
            labels, values = list(quarterly.keys()), list(quarterly.values())
        else:
            labels = [name for _, name, _ in monthly]
            values = [total for _, _, total in monthly]
        is_time_series = True
        group = month_col

    elif group_col:
        operation = requested_aggregation(question)
        if operation == 'count' or not value_col or value_col == group_col:
            labels, values = category_counts(df, group_col)
            metric_column = 'Row count'
        else:
            labels, values = category_aggregate(df, group_col, value_col, operation)
        group = group_col

    elif value_col:
        series = pd.to_numeric(df[value_col], errors='coerce').dropna()
        labels = [str(i + 1) for i in range(min(60, len(series)))]
        values = [float(v) for v in series.iloc[:60]]

    else:
        raise ValueError(
            "Could not identify a numeric or categorical column in this "
            "file to build a chart from."
        )

    if not labels:
        raise ValueError("No chartable data could be computed from this file.")

    chart_type, _source = suggest_chart_type(
        labels, values, is_time_series=is_time_series,
        ai_service=ai_service, group_label=group,
    )

    chart_service = ChartService(**chart_config) if chart_config else ChartService()
    title = f"{metric_column or 'Value'} by {group or 'Category'}"
    chart_image = chart_service.render_base64(
        labels, values,
        chart_type=chart_type,
        title=title,
        xlabel=group or '',
        ylabel=metric_column or '',
    )

    insights = build_ai_context(
        categories=(labels, values),
        category_label=group or metric_column,
    )

    return {
        'chart': chart_image,
        'insights': insights,
        'metric_column': metric_column,
        'group': group,
        'labels': labels,
        'values': values,
    }