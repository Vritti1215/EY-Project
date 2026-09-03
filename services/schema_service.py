"""
Schema Detection Service
=========================
Classifies every column in an uploaded spreadsheet into a data type —
Number, Currency, Date/Time, Text/Categorical, Boolean, Stock, Geography
— WITHOUT relying on the column being named a specific English word.

Why this exists:
    The original insight_service.py detects "the sales column" or "the
    month column" by checking whether the column NAME contains a word
    like 'sales', 'revenue', 'month'. That only works for files that
    happen to use that exact vocabulary. A students sheet ('Marks',
    'Attendance'), an inventory sheet ('Quantity', 'Unit Price'), or a
    stock sheet ('Close', 'Volume') would all fail that check even
    though the DATA is perfectly usable.

    This module instead looks at what's actually IN each column — its
    dtype, its value patterns, its cardinality — to classify it, the same
    way Excel itself infers a column's type. Once every column has a
    type, insight_service's query logic can ask "give me the numeric
    column most relevant to this question" instead of "give me the
    column literally named 'sales'".

This module is additive: it doesn't replace MONTH_COLUMN_ALIASES /
VALUE_COLUMN_ALIASES, it gives you a fallback ladder — try the specific
alias match first (fast, no ambiguity when it hits), then fall back to
generic type-based detection when no alias matches. See
`resolve_numeric_column()` and `resolve_date_column()` below.
"""

import logging
import re
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference lists for Geography / Stock detection
# ---------------------------------------------------------------------------

_COUNTRIES = {
    'india', 'united states', 'usa', 'uk', 'united kingdom', 'china', 'japan',
    'germany', 'france', 'canada', 'australia', 'brazil', 'russia', 'italy',
    'spain', 'mexico', 'indonesia', 'south korea', 'saudi arabia', 'turkey',
    'netherlands', 'switzerland', 'singapore', 'uae', 'united arab emirates',
    'south africa', 'pakistan', 'bangladesh', 'nigeria', 'egypt', 'thailand',
    'vietnam', 'philippines', 'malaysia', 'poland', 'sweden', 'norway',
}
_INDIAN_STATES = {
    'maharashtra', 'delhi', 'karnataka', 'tamil nadu', 'uttar pradesh',
    'gujarat', 'rajasthan', 'west bengal', 'telangana', 'kerala', 'punjab',
    'haryana', 'bihar', 'madhya pradesh', 'andhra pradesh', 'odisha',
    'assam', 'jharkhand', 'chhattisgarh', 'uttarakhand', 'goa',
}
_GEO_COLUMN_NAME_HINTS = (
    'country', 'countries', 'state', 'city', 'cities', 'region', 'location',
    'campus', 'venue', 'address', 'zone', 'territory', 'province', 'nation',
)
_GEO_VALUES = _COUNTRIES | _INDIAN_STATES

_STOCK_PRICE_HINTS = (
    'open', 'close', 'high', 'low', 'adj close', 'adj_close',
    'share price', 'ltp', 'last price',
)
_STOCK_IDENTIFIER_HINTS = ('ticker', 'symbol', 'stock name', 'scrip')
_STOCK_VOLUME_HINTS = ('volume', 'qty traded', 'shares traded')
_CURRENCY_SYMBOLS = ('₹', '$', '€', '£', '¥')
_CURRENCY_COLUMN_NAME_HINTS = (
    'price', 'cost', 'amount', 'revenue', 'sales', 'income', 'expense',
    'fee', 'salary', 'wage', 'budget', 'total', 'value', 'payment',
    'invoice', 'fare', 'charge', 'rate',
)

COLUMN_TYPES = (
    'number', 'currency', 'datetime', 'boolean', 'stock', 'geography', 'text'
)


def _sample(series, n=50):
    """Return up to n non-null values as strings, for pattern sniffing."""
    return series.dropna().astype(str).str.strip().head(n).tolist()


def _looks_like_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return True
    vals = {v.strip().lower() for v in _sample(series, 200)}
    if not vals or len(vals) > 4:
        return False
    boolean_pairs = [
        {'yes', 'no'}, {'true', 'false'}, {'pass', 'fail'},
        {'y', 'n'}, {'1', '0'}, {'active', 'inactive'},
        {'present', 'absent'},
    ]
    return any(vals <= pair for pair in boolean_pairs)


def _looks_like_currency(series, col_name):
    name = str(col_name).lower()
    sample = _sample(series, 50)
    has_symbol = any(any(sym in v for sym in _CURRENCY_SYMBOLS) for v in sample)
    name_hint = any(hint in name for hint in _CURRENCY_COLUMN_NAME_HINTS)
    is_numericish = pd.api.types.is_numeric_dtype(series) or has_symbol
    return is_numericish and (has_symbol or name_hint)


def _looks_like_geography(series, col_name):
    name = str(col_name).lower()
    if any(hint in name for hint in _GEO_COLUMN_NAME_HINTS):
        return True
    sample = {v.lower() for v in _sample(series, 100)}
    if not sample:
        return False
    matches = sum(1 for v in sample if v in _GEO_VALUES)
    return matches / len(sample) >= 0.5


def _looks_like_stock(series, col_name, all_columns):
    """
    True only for NUMERIC price/volume columns from stock-style data
    (Open, Close, High, Low, Volume, LTP, etc.). Ticker/Symbol columns are
    identifiers, not numeric metrics — those are intentionally NOT
    classified as 'stock' here, so they fall through to 'text' and become
    a group-by candidate instead of a (nonsensical) aggregation target.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    name = str(col_name).lower()
    return any(hint in name for hint in _STOCK_PRICE_HINTS + _STOCK_VOLUME_HINTS)


def _looks_like_datetime(series):
    """
    True only for columns that are genuinely date/time-like.

    IMPORTANT: pd.to_datetime() will happily "successfully" parse plain
    integers (interpreting them as epoch timestamps, or 4-digit numbers
    as years) and even short numeric-looking strings. Without guarding
    against that, purely numeric columns like exam marks (88, 92, 75) or
    costs (1200, 300) get misclassified as datetime, which then makes
    resolve_numeric_column() unable to find them at all. So: numeric
    dtypes are NEVER treated as datetime candidates here (a real date
    column is either already datetime64, or text/object containing
    actual date-formatted strings).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ''].head(30)
    if sample.empty:
        return False

    # Require values to actually look date-like (contain a date separator
    # or a month name) before even attempting to parse — this avoids
    # treating short alphanumeric codes/strings as dates just because
    # to_datetime can coerce something out of them.
    date_pattern = re.compile(
        r'(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|'
        r'(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)',
        re.IGNORECASE
    )
    looks_date_shaped = sample.str.contains(date_pattern, regex=True, na=False)
    if looks_date_shaped.mean() < 0.6:
        return False

    parsed = pd.to_datetime(sample, errors='coerce', format='mixed')
    success_rate = parsed.notna().sum() / len(sample)
    return success_rate >= 0.8


def classify_column(df, col_name):
    """
    Classify a single column into one of COLUMN_TYPES.

    Returns:
        dict: {
            'name': col_name,
            'type': one of COLUMN_TYPES,
            'unique_ratio': fraction of non-null values that are unique
                (useful for deciding if a text column is a good group-by
                candidate vs. a free-text/ID column),
            'nunique': number of distinct non-null values,
        }
    """
    series = df[col_name]
    non_null = series.dropna()
    nunique = non_null.nunique()
    unique_ratio = (nunique / len(non_null)) if len(non_null) else 0.0

    col_type = 'text'  # default/fallback

    if _looks_like_boolean(series):
        col_type = 'boolean'
    elif _looks_like_datetime(series):
        col_type = 'datetime'
    elif _looks_like_stock(series, col_name, df.columns):
        col_type = 'stock'
    elif _looks_like_currency(series, col_name):
        col_type = 'currency'
    elif pd.api.types.is_numeric_dtype(series):
        col_type = 'number'
    elif _looks_like_geography(series, col_name):
        col_type = 'geography'
    else:
        col_type = 'text'

    return {
        'name': col_name,
        'type': col_type,
        'unique_ratio': round(float(unique_ratio), 3),
        'nunique': int(nunique),
    }


def classify_dataframe(df):
    """
    Classify every column in a DataFrame.

    Returns:
        list[dict]: one classification dict per column, in column order.
    """
    return [classify_column(df, col) for col in df.columns]


def schema_summary_text(schema):
    """
    Render a classified schema as a short text block — useful both for
    logging/debugging and as extra AI context (e.g. so the model knows
    which columns are numeric vs categorical when phrasing an answer).
    """
    lines = ['Detected column types:']
    for col in schema:
        lines.append(
            f"  - {col['name']}: {col['type']} "
            f"({col['nunique']} distinct values)"
        )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Generic column resolution — the fallback ladder used by insight_service
# ---------------------------------------------------------------------------

def resolve_numeric_column(df, schema, message=None, exclude=None):
    """
    Find the best numeric-ish column (Number, Currency, or Stock-price-like)
    to use as the "value" column for aggregation, WITHOUT requiring it to
    be named 'sales'/'revenue'/'amount'/etc.

    Resolution order:
      1. If `message` mentions a column name directly, use that (if it's
         numeric-typed).
      2. If exactly one Currency-typed column exists, use it (currency
         columns are almost always the intended metric — e.g. 'Fees',
         'Cost', 'Price' — even with unfamiliar names).
      3. If exactly one Number-typed column exists (excluding obvious ID
         columns — very high unique_ratio with no repeats, like a row
         index or roll number), use it.
      4. If multiple numeric columns exist, prefer one whose name is NOT
         an identifier-like word ('id', 'roll', 'number', 'index', 'code')
         and has a meaningful spread of values (not just 1..N).
      5. Otherwise, return None — caller should fall back to row counting.

    Args:
        df: pandas DataFrame.
        schema: output of classify_dataframe(df).
        message: optional natural-language question, checked for an exact
            column-name mention first.
        exclude: optional column name to never return (e.g. the group-by
            column already chosen for this query).

    Returns:
        str or None: the resolved column name.
    """
    exclude = exclude or set()
    if isinstance(exclude, str):
        exclude = {exclude}

    numeric_types = {'number', 'currency', 'stock'}
    candidates = [c for c in schema if c['type'] in numeric_types and c['name'] not in exclude]

    if not candidates:
        return None

    # 1. Direct mention in the question.
    if message:
        msg_lower = message.lower()
        for c in candidates:
            if str(c['name']).lower() in msg_lower:
                return c['name']

    # 2. Prefer currency columns — almost always the intended financial metric.
    currency_candidates = [c for c in candidates if c['type'] == 'currency']
    if len(currency_candidates) == 1:
        return currency_candidates[0]['name']

    # 3/4. Filter out obvious identifier columns (roll no, id, index, code)
    # — these are numeric but not meaningful to sum/average.
    id_like_terms = ('id', 'roll', 'index', 'code', 'no.', 'number', 'sr', 'serial')

    def is_id_like(col):
        name = str(col['name']).lower()
        # A column is ID-like if its name strongly suggests an identifier
        # AND its values are (almost) all unique — a real "Amount" column
        # named "Fee Number" but with repeated/varying values should NOT
        # be excluded just because of the word "number".
        name_hits = any(term in name for term in id_like_terms)
        return name_hits and col['unique_ratio'] > 0.9

    non_id_candidates = [c for c in candidates if not is_id_like(c)]
    pool = non_id_candidates or candidates  # never end up with zero options

    if len(pool) == 1:
        return pool[0]['name']

    # 5. Multiple plausible numeric columns and no way to disambiguate from
    # the question — prefer currency > stock > number, then the one with
    # the most distinct values (usually the "real" metric vs. a rating/
    # flag-like numeric column).
    type_priority = {'currency': 0, 'stock': 1, 'number': 2}
    pool.sort(key=lambda c: (type_priority.get(c['type'], 9), -c['nunique']))
    return pool[0]['name']


def resolve_date_column(df, schema, message=None):
    """
    Find the best Date/Time column, without requiring it to be named
    'month'/'period'/'date'.

    Returns:
        str or None: the resolved column name.
    """
    date_candidates = [c for c in schema if c['type'] == 'datetime']
    if not date_candidates:
        return None

    if message:
        msg_lower = message.lower()
        for c in date_candidates:
            if str(c['name']).lower() in msg_lower:
                return c['name']

    if len(date_candidates) == 1:
        return date_candidates[0]['name']

    # Multiple date columns (e.g. 'Order Date' and 'Ship Date') — prefer
    # one whose name suggests it's the primary/transactional date.
    preferred_terms = ('date', 'month', 'period', 'day')
    scored = sorted(
        date_candidates,
        key=lambda c: 0 if any(t in str(c['name']).lower() for t in preferred_terms) else 1
    )
    return scored[0]['name']


def resolve_group_column(df, schema, message=None, exclude=None):
    """
    Find the best categorical (Text, Boolean, or Geography) column to
    group/count by, when insight_service's message-based detect_group_column()
    doesn't find a match by name.

    This is a fallback of last resort: if the question gives no usable
    hint at all, it picks the categorical column with the most reasonable
    number of distinct groups (not a free-text/ID field with near-100%
    unique values, and not a column with only 1 distinct value).
    """
    exclude = exclude or set()
    if isinstance(exclude, str):
        exclude = {exclude}

    categorical_types = {'text', 'boolean', 'geography'}
    candidates = [
        c for c in schema
        if c['type'] in categorical_types
        and c['name'] not in exclude
        and c['nunique'] > 1
        and c['unique_ratio'] < 0.9  # exclude free-text/ID-like columns
    ]
    if not candidates:
        return None

    if message:
        msg_lower = message.lower()
        for c in candidates:
            if str(c['name']).lower() in msg_lower:
                return c['name']

    # Prefer a moderate number of distinct groups (2-20 is "chartable";
    # too many groups makes for an unreadable chart).
    def chartability_score(c):
        n = c['nunique']
        if 2 <= n <= 20:
            return 0
        return 1

    candidates.sort(key=lambda c: (chartability_score(c), c['unique_ratio']))
    return candidates[0]['name']
