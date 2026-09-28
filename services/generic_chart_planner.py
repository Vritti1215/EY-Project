"""
Generic Chart Planner
=====================
Turns *generic, business-language* chart requests into a real, data-grounded
chart — without needing the user to name a column or a chart type.

Examples this module is built for::

    "Create a graph showing how the data changes over time."
    "Show how the company's customer base has changed over time."
    "Highlight periods where costs appear to be increasing faster than income."
    "Visualise the relationship between the company's income and profitability."
    "Compare the company's financial performance before and after major changes."
    "Highlight the differences between monthly, quarterly, and annual performance."

Why this exists
---------------
The older chart branches in routes/chatbot.py only work when the question
contains a literal column name ("revenue by region") or a small keyword list
("sales", "quarter", ...). For a generic question they fall through to
"count the rows of the first categorical column" — which is how a
"changes over time" request ended up as a bar chart of "Quarter: 15, 15, 15,
15". They also collapse every year into 12 month buckets, so five years of
monthly data became a single meaningless Jan..Dec line.

How it works (deterministic — the AI never computes a number)
------------------------------------------------------------
    question  ->  intent (keyword rules)
              ->  business concepts mapped to REAL columns
                  ("income" -> Revenue, "customer base" -> Customers,
                   "workforce" -> Employees, "profitability" -> Net Profit ...)
              ->  a chronological period table (Year/Quarter/Month aware)
              ->  pandas aggregation (sum for flows, mean for headcounts,
                  ratios recomputed after aggregation)
              ->  matplotlib render in the EY palette

Public API
----------
    detect_intent(question)                    -> str | None
    plan_chart(df, question, sheets=None, ...) -> dict | None
    fallback_plan(df, question)                -> dict | None   (best effort)
    render_plan_base64(plan, ...)              -> "data:image/png;base64,..."
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re

import matplotlib

matplotlib.use("Agg")  # headless — same as chart_service.py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChartNotSupported(Exception):
    """Raised by plan_from_ai when the model determines the uploaded data cannot answer the question at all (as opposed to the call simply failing)."""

# EY brand-aligned colours (same values as chart_service.BRAND_PALETTES['ey'],
# plus a few high-contrast accents for multi-series charts).
DARK = "#2E2E38"
YELLOW = "#FFE600"
GREY = "#9DA0A6"
CYAN = "#06B6D4"
ORANGE = "#F97316"
GREEN = "#16A34A"
RED = "#DC2626"
SERIES_COLORS = [DARK, CYAN, ORANGE, GREEN, "#7C3AED", GREY]

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
MONTH_SHORT = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# 1. Business concepts -> real columns
# ---------------------------------------------------------------------------
# Each concept lists column-name fragments in PRIORITY order. The first
# numeric column containing a fragment (and none of the global exclusions)
# wins. 'agg' says how the column behaves when months are rolled into
# quarters/years: money flows are summed, headcount-style stocks are averaged.
_GLOBAL_EXCLUDE = ("%", "margin", "growth", "ratio", "rate", "per ", "index",
                    "number", "id")

CONCEPTS = {
    "revenue": dict(
        label="Revenue", agg="sum",
        patterns=["revenue", "net sales", "total sales", "sales", "turnover",
                  "total income", "income", "amount"],
        exclude=["cost", "expense", "tax", "profit", "cash", "before", "interest"],
    ),
    "net_profit": dict(
        label="Net Profit", agg="sum",
        patterns=["net profit", "net income", "profit after tax", "pat",
                  "net earnings", "profit"],
        exclude=["gross", "before", "operating", "margin", "loss"],
    ),
    "gross_profit": dict(
        label="Gross Profit", agg="sum",
        patterns=["gross profit", "gross income", "gross margin value"],
        exclude=[],
    ),
    "ebitda": dict(
        label="EBITDA", agg="sum",
        patterns=["ebitda", "operating profit", "operating income", "ebit"],
        exclude=[],
    ),
    "customers": dict(
        label="Customers", agg="mean",
        patterns=["customer", "client", "subscriber", "member", "user"],
        exclude=["acquisition cost", "cost", "revenue", "satisfaction"],
    ),
    "employees": dict(
        label="Employees", agg="mean",
        patterns=["employee", "headcount", "head count", "staff", "workforce", "fte"],
        exclude=["cost", "expense", "salary", "turnover"],
    ),
    "cash_flow": dict(
        label="Operating Cash Flow", agg="sum",
        patterns=["operating cash flow", "cash flow"],
        exclude=["free"],
    ),
}

_COST_INCLUDE = ("cogs", "cost of", "operating expense", "opex", "expense", "cost")
_COST_EXCLUDE = ("interest", "tax", "depreciation", "amortization", "amortisation",
                 "capital", "capex", "%", "margin", "per ", "acquisition")


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _numeric_columns(df: pd.DataFrame) -> list:
    """Columns that are (or are mostly) numeric — strings like '1,234' count."""
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().sum() > 0:
                cols.append(c)
            continue
        cleaned = pd.to_numeric(
            s.astype(str).str.replace(r"[,$€£₹\s]", "", regex=True), errors="coerce"
        )
        if len(s) and cleaned.notna().mean() >= 0.8:
            cols.append(c)
    return cols


def _as_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,$€£₹\s]", "", regex=True), errors="coerce"
    )


def _is_time_like_name(name) -> bool:
    n = _norm(name)
    return n in {"year", "month", "month number", "quarter", "qtr", "day", "week"} or \
        n.endswith(" year") or n.endswith("_year")


def find_concept_column(df: pd.DataFrame, concept: str):
    """Best real column for a business concept, or None."""
    spec = CONCEPTS[concept]
    numeric = [c for c in _numeric_columns(df) if not _is_time_like_name(c)]
    for pattern in spec["patterns"]:
        for c in numeric:
            n = _norm(c)
            if pattern not in n:
                continue
            if any(g in n for g in _GLOBAL_EXCLUDE):
                continue
            if any(e in n for e in spec["exclude"]):
                continue
            return c
    return None


def find_cost_columns(df: pd.DataFrame) -> list:
    """
    Operating-cost columns (COGS, Operating Expenses, R&D, Marketing ...).
    Interest, tax, depreciation and capex are NOT operating costs, so they
    are left out. A single explicit 'Total Cost/Expenses' column wins.
    """
    numeric = [c for c in _numeric_columns(df) if not _is_time_like_name(c)]
    for c in numeric:
        n = _norm(c)
        if n.startswith("total") and any(k in n for k in ("cost", "expense")) \
                and not any(e in n for e in _COST_EXCLUDE):
            return [c]
    found = []
    for c in numeric:
        n = _norm(c)
        if any(k in n for k in _COST_INCLUDE) and not any(e in n for e in _COST_EXCLUDE):
            found.append(c)
    return found


# ---------------------------------------------------------------------------
# 1b. Financial-year references
# ---------------------------------------------------------------------------
# RULE (whole app): "financial year", "fiscal year", "FY", "F.Y.", "FYE",
# "FYTD", "fiscal quarter", "financial half" ... ALWAYS mean April -> March.
#   FY25 / FY2025 / FY24-25 / F.Y. 2024-25 / financial year 2025
#       -> 1 Apr 2024 - 31 Mar 2025      (named after the year it ENDS in)
#   Q1 FY25 = Apr-Jun 2024, Q2 = Jul-Sep, Q3 = Oct-Dec, Q4 = Jan-Mar 2025
#   H1 FY25 = Apr-Sep 2024, H2 = Oct 2024-Mar 2025
#   "current / this / latest FY" = the FY containing the newest data,
#   "last / previous FY" = the one before it, "FYTD" = current FY so far.
_FY_TOKEN = (r"(?:fy|f\.\s?y\.?|fye|(?:financial|fiscal|finance)\s+(?:year|yr)|fin\.?\s*yr)")
_FY_NUM_RE = re.compile(
    r"(?:(?<![a-z])(?P<sub>[qh])\s?(?P<subn>[1-4])\s*[-,]?\s*(?:of\s+)?)?"
    r"(?<![a-z])" + _FY_TOKEN + r"[\s'\u2019\-\.:]*(?P<a>\d{2,4})"
    r"(?:\s*[-/\u2013]\s*'?(?P<b>\d{2,4}))?(?!\d)"
    r"(?!\s*(?:months?|mths?|quarters?|weeks?|days?))",
    re.I,
)
_FY_REL_RE = re.compile(
    r"(?<![a-z])(?P<rel>current|this|latest|recent|last|previous|prior|preceding)\s+"
    r"(?:fy|f\.\s?y\.?|(?:financial|fiscal)\s+(?:year|yr))(?![a-z])", re.I)
_FYTD_RE = re.compile(
    r"(?<![a-z])fytd(?![a-z])|(?<![a-z])(?:fy|(?:financial|fiscal)\s+(?:year|yr))\s*(?:to[- ]date|ytd)", re.I)
_FY_MENTION = re.compile(
    r"(?<![a-z])(?:fy|f\.\s?y\.?|fye|fytd)(?![a-z])"
    r"|(?:financial|fiscal|finance)\s+(?:year|yr|quarter|qtr|half|period)"
    r"|fin\.?\s*yr|fiscal", re.I)
_NEWER = {"current", "this", "latest", "recent"}


def _fy_end_year(text) -> int:
    y = int(text)
    return 2000 + y if y < 100 else y


def _fy_of(ts) -> int:
    """End-year of the Apr-Mar financial year containing timestamp `ts`."""
    return int(ts.year + (1 if ts.month >= 4 else 0))


def _fy_range(end_year: int):
    return pd.Timestamp(end_year - 1, 4, 1), pd.Timestamp(end_year, 3, 31)


def _fy_label(end_year: int) -> str:
    return f"FY{str(end_year)[-2:]}"


def _sub_range(end_year: int, kind: str, n: int):
    start, _ = _fy_range(end_year)
    if kind.lower() == "q":
        s0 = start + pd.DateOffset(months=3 * (n - 1))
        return s0, s0 + pd.DateOffset(months=3) - pd.Timedelta(days=1), f"Q{n} {_fy_label(end_year)}"
    if n > 2:
        return None
    s0 = start + pd.DateOffset(months=6 * (n - 1))
    return s0, s0 + pd.DateOffset(months=6) - pd.Timedelta(days=1), f"H{n} {_fy_label(end_year)}"


def detect_fiscal_years(question: str, ref_date=None):
    """
    Every financial year / FY quarter / FY half named in `question`, as
    [(label, start_timestamp, end_timestamp), ...] sorted chronologically.
    Always April -> March. `ref_date` (the newest date in the data) lets
    relative wording ("current FY", "last financial year", "FYTD") resolve;
    without it those are ignored.
    """
    q = question or ""
    seen, out = set(), []

    def add(label, start, end):
        if label not in seen:
            seen.add(label)
            out.append((label, start, end))

    for m in _FY_NUM_RE.finditer(q):
        a, b = m.group("a"), m.group("b")
        end_year = _fy_end_year(b) if b else _fy_end_year(a)
        if not (1990 <= end_year <= 2100):
            continue
        if m.group("sub"):
            sub = _sub_range(end_year, m.group("sub"), int(m.group("subn")))
            if sub:
                add(sub[2], sub[0], sub[1])
                continue
        start, end = _fy_range(end_year)
        add(_fy_label(end_year), start, end)

    if ref_date is not None:
        cur = _fy_of(pd.Timestamp(ref_date))
        for m in _FY_REL_RE.finditer(q):
            end_year = cur if m.group("rel").lower() in _NEWER else cur - 1
            start, end = _fy_range(end_year)
            add(_fy_label(end_year), start, end)
        if _FYTD_RE.search(q):
            start, _ = _fy_range(cur)
            add(f"{_fy_label(cur)} YTD", start, pd.Timestamp(ref_date))

    out.sort(key=lambda t: (t[1], t[2]))
    return out


def _fy_mention_count(question: str) -> int:
    """Number of distinct FY periods named, counting relative wording
    ("this FY vs last FY") even before the data is known."""
    q = question or ""
    n = len({(m.group("a"), m.group("b"), m.group("sub"), m.group("subn")) for m in _FY_NUM_RE.finditer(q)})
    rels = {("new" if m.group("rel").lower() in _NEWER else "old") for m in _FY_REL_RE.finditer(q)}
    return n + len(rels)


def mentions_fiscal_year(question: str) -> bool:
    """True for ANY financial-year wording: FY25, F.Y., financial/fiscal year,
    fiscal quarter, FYTD, 'FY wise' ... All of it means April -> March."""
    return bool(_FY_MENTION.search(question or ""))


def fy_availability(df: pd.DataFrame, question: str):
    """
    None when no FY is named in `question`. Otherwise a dict describing, for
    every named FY, whether the uploaded data actually has rows in that
    Apr-Mar window — so the caller can tell the user plainly when the data
    they asked for ("FY26") simply isn't in the file, instead of drawing a
    chart from whatever period happens to be available.
    """
    frame = _frame_or_none_raw(df)
    ref = frame["_date"].max() if frame is not None and not frame.empty else None
    fys = detect_fiscal_years(question, ref_date=ref)
    if not fys:
        return None
    if frame is None or frame.empty:
        return {"fys": fys, "data_min": None, "data_max": None,
                "available": {label: False for label, _, _ in fys}}
    dmin, dmax = frame["_date"].min(), frame["_date"].max()
    available = {label: bool(((frame["_date"] >= start) & (frame["_date"] <= end)).any())
                for label, start, end in fys}
    return {"fys": fys, "data_min": dmin, "data_max": dmax, "available": available}


# ---------------------------------------------------------------------------
# 2. Chronological period table
# ---------------------------------------------------------------------------
def _month_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = int(value)
        return v if 1 <= v <= 12 else None
    text = str(value).strip().lower()
    if text in MONTHS:
        return MONTHS[text]
    if text[:3] in MONTHS and text.isalpha():
        return MONTHS[text[:3]]
    if text.isdigit() and 1 <= int(text) <= 12:
        return int(text)
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    for kwargs in ({"format": "mixed", "dayfirst": True}, {"dayfirst": True}, {}):
        try:
            parsed = pd.to_datetime(series, errors="coerce", **kwargs)
            if parsed.notna().mean() >= 0.6:
                return parsed
        except Exception:  # noqa: BLE001 - older pandas / odd values
            continue
    return pd.Series(pd.NaT, index=series.index)


def _find_year_column(df):
    for c in df.columns:
        n = _norm(c)
        if n == "year" or n.endswith(" year") or n.endswith("_year") or n == "fiscal year" or n == "fy":
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.8 and s.dropna().between(1900, 2100).all():
                return c
    return None


def _find_quarter_column(df):
    for c in df.columns:
        n = _norm(c)
        if n in ("quarter", "qtr", "fiscal quarter") or n.endswith(" quarter"):
            vals = df[c].astype(str).str.upper().str.extract(r"Q?\s*([1-4])")[0]
            if vals.notna().mean() > 0.8:
                return c
    return None


def build_period_frame(df: pd.DataFrame):
    """
    Return a copy of `df` with a `_date` column (first day of the month /
    quarter / year the row belongs to) or None when the data has no usable
    time dimension.

    Priority: a real date column -> Year + Month -> Year + Quarter -> Year.
    """
    work = df.copy()
    date_col = None

    # 1. real date column (prefer names that say so)
    ranked = sorted(
        work.columns,
        key=lambda c: (0 if any(k in _norm(c) for k in ("date", "period", "time", "day")) else 1),
    )
    for c in ranked:
        s = work[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_col = c
            break
        if pd.api.types.is_numeric_dtype(s) or _norm(c) in ("month", "quarter"):
            continue
        looks_datey = any(k in _norm(c) for k in ("date", "period", "time", "day", "month"))
        if looks_datey or s.dtype == object:
            parsed = _parse_dates(s)
            if parsed.notna().mean() >= 0.6 and parsed.dropna().nunique() >= 3:
                date_col = c
                break
    if date_col is not None:
        work["_date"] = _parse_dates(work[date_col])
        work = work.dropna(subset=["_date"])
        work["_date"] = work["_date"].dt.to_period("M").dt.to_timestamp()
        return (work, date_col) if len(work) else None

    # 2. Year (+ Month | Quarter)
    year_col = _find_year_column(work)
    if year_col is not None:
        year = pd.to_numeric(work[year_col], errors="coerce")
        month_col = next((c for c in work.columns
                          if _norm(c) in ("month", "month name", "month number", "mon")), None)
        quarter_col = _find_quarter_column(work)
        month = None
        if month_col is not None:
            month = work[month_col].map(_month_number)
        if month is None or month.isna().mean() > 0.2:
            if quarter_col is not None:
                q = work[quarter_col].astype(str).str.upper().str.extract(r"Q?\s*([1-4])")[0].astype(float)
                month = (q - 1) * 3 + 1
            else:
                month = pd.Series(1, index=work.index)
        work["_date"] = pd.to_datetime(
            {"year": year, "month": month.fillna(1).astype(int), "day": 1}, errors="coerce"
        )
        work = work.dropna(subset=["_date"])
        return (work, year_col) if len(work) else None

    return None


def _native_granularity(dates: pd.Series) -> str:
    uniq = np.sort(dates.dropna().unique())
    if len(uniq) < 2:
        return "Y"
    gaps = np.diff(uniq).astype("timedelta64[D]").astype(float)
    med = float(np.median(gaps))
    if med < 45:
        return "M"
    if med < 130:
        return "Q"
    return "Y"


_GRAN_RANK = {"M": 0, "Q": 1, "Y": 2}


def choose_granularity(dates: pd.Series, question: str = "", prefer: str | None = None) -> str:
    q = (question or "").lower()
    native = _native_granularity(dates)
    fiscal = mentions_fiscal_year(q)
    want = prefer
    if re.search(r"\b(monthly|month[- ]by[- ]month|each month|per month|by month|month[- ]wise)\b", q):
        want = "M"
    elif re.search(r"\b(quarterly|each quarter|per quarter|by quarter|quarter[- ]wise)\b|"
                   r"(?:fiscal|financial) quarters?|fy quarters?", q):
        want = "Q"
    elif re.search(r"\b(annual|annually|yearly|year[- ]over[- ]year|yoy|each year|per year|by year|year[- ]wise)\b|"
                   r"(?<![a-z])(?:fy|(?:financial|fiscal) years?)[- ]?wise|"
                   r"\b(?:by|per|each|across|between)\s+(?:fy|(?:financial|fiscal) years?)\b", q):
        want = "Y"
    if want is None:
        months = dates.dt.to_period("M").nunique()
        want = "M" if months <= 24 else ("Q" if months <= 96 else "Y")
    # never finer than the data itself
    if _GRAN_RANK[want] < _GRAN_RANK[native]:
        want = native
    # a yearly view of a single (financial) year is useless
    n_years = (dates.dt.year + (dates.dt.month >= 4).astype(int)).nunique() if fiscal else dates.dt.year.nunique()
    if want == "Y" and n_years < 2:
        want = "Q" if _GRAN_RANK["Q"] >= _GRAN_RANK[native] else native
    return want


def _period_keys(frame: pd.DataFrame, gran: str, quarter_col=None, fiscal=False):
    d = frame["_date"]
    if gran == "M":
        sort = d.dt.year * 100 + d.dt.month
        label = d.dt.strftime("%b %Y")
    elif fiscal and gran == "Q":
        fy = d.dt.year + (d.dt.month >= 4).astype(int)
        fq = ((d.dt.month - 4) % 12) // 3 + 1          # Apr-Jun = Q1 ... Jan-Mar = Q4
        sort = fy * 10 + fq
        label = "FY" + (fy % 100).astype(str).str.zfill(2) + " Q" + fq.astype(str)
    elif fiscal and gran == "Y":
        fy = d.dt.year + (d.dt.month >= 4).astype(int)  # FY25 = Apr 2024 - Mar 2025
        sort = fy
        label = "FY" + (fy % 100).astype(str).str.zfill(2)
    elif gran == "Q":
        if quarter_col is not None and _find_year_column(frame) is not None:
            yc = _find_year_column(frame)
            yr = pd.to_numeric(frame[yc], errors="coerce").astype("Int64")
            qn = frame[quarter_col].astype(str).str.upper().str.extract(r"Q?\s*([1-4])")[0].astype("Int64")
        else:
            yr = d.dt.year.astype("Int64")
            qn = ((d.dt.month - 1) // 3 + 1).astype("Int64")
        sort = yr * 10 + qn
        label = yr.astype(str) + " Q" + qn.astype(str)
    else:
        sort = d.dt.year
        label = d.dt.year.astype(str)
    return sort, label


def aggregate(frame: pd.DataFrame, columns: dict, gran: str) -> pd.DataFrame:
    """
    Aggregate `columns` ({name: 'sum'|'mean'|'last'}) into chronological
    periods. Returns a DataFrame indexed by period label, in time order.
    When the frame is flagged fiscal, quarters/years are Apr-Mar financial
    quarters/years, and a period the data only partly covers is labelled
    e.g. "FY26 (9 mo)" so it is never mistaken for a full year.
    """
    fiscal = bool(getattr(frame, "attrs", {}).get("fiscal"))
    f = frame.copy()
    quarter_col = _find_quarter_column(f)
    f["_sort"], f["_label"] = _period_keys(f, gran, quarter_col, fiscal)
    for c in columns:
        f[c] = _as_number(f[c])
    grouped = f.groupby(["_sort", "_label"], sort=True).agg(columns)
    if fiscal and gran in ("Y", "Q"):
        n_months = f.groupby(["_sort", "_label"])["_date"].nunique()
        full = 12 if gran == "Y" else 3
        grouped["_label2"] = [
            (lab if n_months[(srt, lab)] >= full else f"{lab} ({int(n_months[(srt, lab)])} mo)")
            for srt, lab in grouped.index
        ]
        grouped = grouped.reset_index().sort_values("_sort")
        grouped = grouped.drop(columns=["_sort", "_label"]).rename(columns={"_label2": "_label"}).set_index("_label")
        return grouped
    grouped = grouped.reset_index().sort_values("_sort")
    grouped = grouped.drop(columns="_sort").set_index("_label")
    return grouped


def _lag_for(gran: str) -> int:
    return {"M": 12, "Q": 4, "Y": 1}[gran]


# ---------------------------------------------------------------------------
# 3. Intent detection
# ---------------------------------------------------------------------------
_OPERATING = r"operating (?:performance|profit|result)|operational performance"
_FINANCIAL = r"(?:overall )?financial performance|bottom line|overall performance"

# Order matters: most specific first.
_INTENT_RULES = [
    ("fy_compare", lambda q: _fy_mention_count(q) >= 2),
    ("granularity_compare",
     lambda q: re.search(r"monthly", q) and re.search(r"quarterly", q)
     and re.search(r"annual|yearly", q)),
    ("before_after",
     lambda q: re.search(r"before and after|before/after|before vs|pre[- ]and[- ]post|"
                         r"(?:before|after) (?:the )?(?:major |big |key )?(?:change|shift|event)", q)),
    ("anomalies",
     lambda q: re.search(r"unusual|abnormal|anomal|outlier|exceptional|"
                         r"unusually (?:high|low)|spike|abrupt", q)),
    ("costs_vs_income",
     lambda q: re.search(r"cost|expense|spend", q)
     and re.search(r"faster|more quickly|outpac|grow(?:ing)? (?:faster|more)|"
                   r"increas|rising|higher than|than (?:income|revenue|sales)", q)
     and re.search(r"income|revenue|sales|turnover", q)),
    ("operating_vs_overall",
     lambda q: re.search(_OPERATING, q) and re.search(_FINANCIAL + r"|overall|net", q)),
    ("growth_compare",
     lambda q: (re.search(r"business (?:growth|activity|performance|scale)|operational growth|"
                          r"customer.*(?:employee|workforce)|(?:workforce|employee).*customer", q)
                and re.search(r"financial|revenue|profit|income", q))
     or re.search(r"growth.{0,40}(?:accompan|translat|improv)|(?:accompan|translat).{0,40}growth", q)),
    ("profit_vs_scale",
     lambda q: re.search(r"(?:more|less) profitable.{0,20}(?:grow|larger|scale|bigger)|"
                         r"profitab\w+.{0,30}(?:as it grows|as the company grows|with (?:growth|scale|size))", q)),
    ("relationship",
     lambda q: re.search(r"relationship|correlat|(?:link|connection) between|versus|\bvs\b", q)
     and re.search(r"income|revenue|sales|turnover", q) and re.search(r"profit", q)),
    ("largest_improvements",
     lambda q: re.search(r"improv|biggest (?:gain|jump|increase)|largest (?:gain|jump|increase)|"
                         r"best (?:period|quarter|month)", q)
     and re.search(r"period|quarter|month|year|performance", q)),
    ("efficiency",
     lambda q: re.search(r"efficien", q)),
    ("customers_workforce",
     lambda q: re.search(r"customer|client|subscriber|workforce|employee|headcount|staff|head count", q)),
    ("profitability",
     lambda q: re.search(r"profitab", q)),
    # "revenue by financial year", "FY wise profit", "compare across fiscal years"
    ("fy_overview",
     lambda q: mentions_fiscal_year(q) and _fy_mention_count(q) == 0
     and re.search(r"wise|\bby\b|\bper\b|each|across|compar|between|growth|trend|over|perform|summary|overview|yoy|show|chart|graph|plot", q)),
]

_GENERIC_TIME = re.compile(
    r"over time|changes? over|change over|trend|across (?:different |all |the )?(?:reporting )?periods?|"
    r"time series|timeline|through time|history|how .{0,30}(?:has|have|did|does|is|are) .{0,30}chang"
    r"|data changes?|grow(?:th|n)|evolv|\bfy\s?\d|financial year|fiscal year"
)
_VISUAL_VERBS = re.compile(r"graph|chart|plot|visuali[sz]e|draw|diagram|show|highlight|illustrate|display|picture|compare|comparing")


def detect_intent(question: str):
    q = (question or "").lower()
    if not q.strip():
        return None
    for name, rule in _INTENT_RULES:
        try:
            if rule(q):
                return name
        except Exception:  # noqa: BLE001
            continue
    if _GENERIC_TIME.search(q):
        return "generic_time"
    return None


STRONG_INTENTS = {
    "fy_compare", "fy_overview", "granularity_compare", "before_after", "anomalies", "costs_vs_income",
    "operating_vs_overall", "growth_compare", "profit_vs_scale", "relationship",
    "largest_improvements", "efficiency",
}
ANALYTIC_INTENTS = STRONG_INTENTS | {"customers_workforce", "profitability"}


_VISUAL_NOUNS = re.compile(r"\bvisual\w*|\binfographic\b|\bgraphic\b")
_AI_CUES = re.compile(
    r"relationship|correlat|\bversus\b|\bvs\b|top \d+|bottom \d+|\bbest\b|\bworst\b|highest|lowest|"
    r"\bmost\b|\bleast\b|rank|distribution|share of|proportion|percentage|breakdown|pattern|seasonal|"
    r"perform\w*|overview|contribut\w*|drivers?|split|composition|spread"
)


def has_ai_cue(question: str) -> bool:
    """Analytical wording (ranking, relationship, distribution, ...) the rules
    do not model — worth sending to the AI planner even when a column is named."""
    return bool(_AI_CUES.search((question or "").lower()))


def has_time_dimension(df: pd.DataFrame) -> bool:
    """A real time axis needs at least two distinct periods (a 'Time' column of
    clock times such as '9.00 AM' is not a time dimension)."""
    built = build_period_frame(df)
    return built is not None and built[0]["_date"].nunique() >= 2


def wants_visual(question: str) -> bool:
    """True when the question is one of the analytical prompts this planner
    answers with a chart, even if it never says 'chart' or 'graph'
    (e.g. "Highlight periods where costs are increasing faster than income")."""
    intent = detect_intent(question)
    q = (question or "").lower()
    if intent is None:
        return bool(_VISUAL_NOUNS.search(q))
    if intent in STRONG_INTENTS:
        return True
    # weaker intents (customers, profitability, plain "over time") need an
    # explicit visual verb AND a time/period phrase, so an ordinary
    # question such as "how many customers are in Delhi?" stays a text answer.
    has_time = bool(_GENERIC_TIME.search(q) or re.search(r"across|periods?|by (?:month|quarter|year)", q))
    return bool(_VISUAL_VERBS.search(q)) and has_time


# ---------------------------------------------------------------------------
# 4. Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_num(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.{decimals}f}B"
    if a >= 1e6:
        return f"{v / 1e6:.{decimals}f}M"
    if a >= 1e4:
        return f"{v / 1e3:.{decimals}f}K"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def _pct(v, decimals=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{decimals}f}%"


def _safe_pct_change(cur, prev):
    if prev is None or cur is None or prev == 0 or pd.isna(prev) or pd.isna(cur):
        return float("nan")
    return (cur - prev) / abs(prev) * 100.0


GRAN_NAME = {"M": "Month", "Q": "Quarter", "Y": "Year"}
GRAN_ADJ = {"M": "monthly", "Q": "quarterly", "Y": "annual"}


def _series_dict(label, values, color=None, fmt="num"):
    return {"label": label, "values": [None if pd.isna(v) else float(v) for v in values],
            "color": color, "fmt": fmt}


def _base_plan(intent, kind, title, labels, series, **extra):
    plan = {
        "intent": intent, "kind": kind, "title": title, "labels": list(labels),
        "series": series, "xlabel": "", "ylabel": "", "y_fmt": "num",
        "facts": "", "reason": "", "assumptions": [],
    }
    plan.update(extra)
    return plan


def _facts_header(plan_title, assumptions):
    lines = ["AUTHORITATIVE DETERMINISTIC RESULT — DO NOT RECALCULATE:",
             f"Chart generated by the server: {plan_title}"]
    for a in assumptions:
        lines.append(f"Assumption: {a}")
    return lines


# ---------------------------------------------------------------------------
# 5. The planner
# ---------------------------------------------------------------------------
def _primary_metric(df, question, prefer=("net_profit", "ebitda", "revenue")):
    """Pick the single metric that best represents 'performance'."""
    q = (question or "").lower()
    order = list(prefer)
    if re.search(r"revenue|sales|income|turnover|top[- ]line", q):
        order = ["revenue"] + [c for c in order if c != "revenue"]
    elif re.search(r"profit", q):
        order = ["net_profit"] + [c for c in order if c != "net_profit"]
    elif re.search(r"ebitda|operating", q):
        order = ["ebitda"] + [c for c in order if c != "ebitda"]
    for concept in order + ["revenue", "net_profit", "ebitda", "gross_profit", "cash_flow"]:
        col = find_concept_column(df, concept)
        if col is not None:
            return concept, col
    # Domain-agnostic fallback: first sensible numeric column.
    numeric = [c for c in _numeric_columns(df) if not _is_time_like_name(c)]
    numeric = [c for c in numeric if df[c].nunique(dropna=True) > 3]
    if numeric:
        return "generic", numeric[0]
    return None, None


def _metric_agg(concept, col):
    if concept in CONCEPTS:
        return CONCEPTS[concept]["agg"]
    n = _norm(col)
    if any(k in n for k in ("avg", "average", "price", "score", "rating", "mean", "%")):
        return "mean"
    return "sum"


def _label_for(concept, col, agg):
    base = str(col)
    return f"{base} (avg)" if agg == "mean" else base


def plan_chart(df: pd.DataFrame, question: str, sheets: dict | None = None,
               allow_generic_time: bool = True):
    """
    Build a chart plan for `question` from `df` (or None when the question is
    not one this planner recognises / the data cannot support it).
    """
    intent = detect_intent(question)
    if intent is None or df is None or df.empty:
        return None
    if intent == "generic_time" and not allow_generic_time:
        return None

    handler = {
        "fy_compare": _plan_fy_compare,
        "fy_overview": _plan_fy_overview,
        "granularity_compare": _plan_granularity_compare,
        "before_after": _plan_before_after,
        "anomalies": _plan_anomalies,
        "costs_vs_income": _plan_costs_vs_income,
        "operating_vs_overall": _plan_operating_vs_overall,
        "growth_compare": _plan_growth_compare,
        "profit_vs_scale": _plan_profit_vs_scale,
        "relationship": _plan_relationship,
        "largest_improvements": _plan_largest_improvements,
        "efficiency": _plan_efficiency,
        "customers_workforce": _plan_customers_workforce,
        "profitability": _plan_profitability,
        "generic_time": _plan_generic_time,
    }[intent]

    try:
        plan = handler(df, question, sheets or {})
    except Exception:  # noqa: BLE001
        logger.exception("Generic chart planner failed for intent %s", intent)
        return None
    return plan


# -- helpers shared by handlers ---------------------------------------------
def _frame_or_none_raw(df):
    built = build_period_frame(df)
    return built[0] if built else None


def _frame_or_none(df, question=None):
    """
    Chronological frame for `df`. When the question uses ANY financial-year
    wording the frame is flagged `attrs["fiscal"]` so aggregate() buckets by
    Apr-Mar financial years / fiscal quarters. When it names exactly one FY
    (or FY quarter/half, "current FY", FYTD) the frame is also scoped to that
    window and flagged `attrs["fy_scoped"]`. Two or more named FYs are handled
    by the fy_compare intent. Returns None when there is no time dimension or
    a single named FY has zero matching rows.
    """
    frame = _frame_or_none_raw(df)
    if frame is None:
        return None
    if question and mentions_fiscal_year(question):
        fys = detect_fiscal_years(question, ref_date=frame["_date"].max())
        if len(fys) == 1:
            label, start, end = fys[0]
            frame = frame[(frame["_date"] >= start) & (frame["_date"] <= end)]
            if frame.empty:
                return None
            frame = frame.copy()
            frame.attrs["fy_scoped"] = True
            frame.attrs["fy_label"] = label
        else:
            frame = frame.copy()
        frame.attrs["fiscal"] = True
    return frame


def _fy_suffix(frame) -> str:
    lab = getattr(frame, "attrs", {}).get("fy_label")
    return f" — {lab} (Apr-Mar)" if lab else ""


def _year_count(frame) -> int:
    d = frame["_date"]
    if getattr(frame, "attrs", {}).get("fiscal"):
        return int((d.dt.year + (d.dt.month >= 4).astype(int)).nunique())
    return int(d.dt.year.nunique())


def _first_last_facts(labels, values, name):
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return []
    first, last = vals[0], vals[-1]
    peak_i = int(np.nanargmax(values))
    low_i = int(np.nanargmin(values))
    return [
        f"{name}: {labels[0]} = {_fmt_num(first, 2)} -> {labels[-1]} = {_fmt_num(last, 2)} "
        f"({_pct(_safe_pct_change(last, first))} overall).",
        f"{name}: peak {_fmt_num(values[peak_i], 2)} in {labels[peak_i]}; "
        f"lowest {_fmt_num(values[low_i], 2)} in {labels[low_i]}.",
    ]


# -- generic "how does the data change over time" ---------------------------
def _plan_generic_time(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    # A measure was named ("plot Marketing Expense over time") -> use THAT one.
    if _named_metric_columns(df, question):
        named = plan_named_metrics(df, question)
        if named:
            return named
    months = frame["_date"].dt.to_period("M").nunique()
    # Single-year data is handled well by the original monthly/quarterly
    # branch (which also honours the FISCAL_YEAR_START_MONTH setting) — but a
    # request scoped to ONE named financial year must be plotted here, since
    # the legacy branch would ignore the FY and use every year in the file.
    if months <= 12 and not frame.attrs.get("fy_scoped"):
        return None
    concept, col = _primary_metric(df, question, prefer=("revenue", "net_profit", "ebitda"))
    if col is None:
        return None
    gran = choose_granularity(frame["_date"], question)
    agg = _metric_agg(concept, col)
    table = aggregate(frame, {col: agg}, gran)
    labels, values = list(table.index), table[col].tolist()
    label = _label_for(concept, col, agg)

    title = f"{label} over time ({GRAN_ADJ[gran]})" + _fy_suffix(frame)
    assumptions = [
        f"No specific measure was named, so '{col}' was used as the main measure "
        f"and the time axis is the '{_date_source(df, frame)}' data grouped {GRAN_ADJ[gran]}."
    ]
    plan = _base_plan(
        "generic_time", "line", title, labels, [_series_dict(label, values, DARK)],
        xlabel=GRAN_NAME[gran], ylabel=label, show_trend=True, annotate_extremes=True,
        assumptions=assumptions, legacy_compatible=True, is_time_series=True,
        reason=f"Line chart of {label} across {len(labels)} {GRAN_ADJ[gran]} periods.",
    )
    other = [c for c in _numeric_columns(df) if c != col and not _is_time_like_name(c)][:8]
    facts = _facts_header(title, assumptions)
    facts += [f"{GRAN_NAME[gran]}-by-{GRAN_NAME[gran].lower()} values:"]
    facts += [f"  - {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, values)]
    facts += _first_last_facts(labels, values, label)
    if other:
        facts.append("Other measures available to chart: " + ", ".join(map(str, other)) + ".")
    plan["facts"] = "\n".join(facts)
    return plan


def _date_source(df, frame):
    built = build_period_frame(df)
    return built[1] if built else "date"


# -- customers / workforce ---------------------------------------------------
def _plan_customers_workforce(df, question, sheets):
    q = question.lower()
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    wants_c = re.search(r"customer|client|subscriber", q)
    wants_e = re.search(r"workforce|employee|headcount|head count|staff", q)
    picks = []
    if wants_c:
        c = find_concept_column(df, "customers")
        if c is not None:
            picks.append(("customers", c))
    if wants_e:
        e = find_concept_column(df, "employees")
        if e is not None:
            picks.append(("employees", e))
    if not picks:
        return None  # the data simply has no such column -> let caller explain
    gran = choose_granularity(frame["_date"], question)
    cols = {c: "mean" for _, c in picks}
    table = aggregate(frame, cols, gran)
    labels = list(table.index)

    if len(picks) == 1:
        concept, col = picks[0]
        values = table[col].tolist()
        label = f"{col} (avg per {GRAN_NAME[gran].lower()})"
        title = f"{col} over time ({GRAN_ADJ[gran]} average)"
        plan = _base_plan(
            "customers_workforce", "line", title, labels,
            [_series_dict(label, values, DARK)],
            xlabel=GRAN_NAME[gran], ylabel=label, show_trend=True, annotate_extremes=True,
            legacy_compatible=True, is_time_series=True,
            reason=f"{col} is a headcount-style measure, so each period shows its average level.",
        )
        facts = _facts_header(title, [f"'{col}' is a stock-style measure; periods show the average."])
        facts += [f"  - {l}: {_fmt_num(v, 1)}" for l, v in zip(labels, values)]
        facts += _first_last_facts(labels, values, col)
        plan["facts"] = "\n".join(facts)
        return plan

    # both -> indexed so different magnitudes are comparable
    series, facts_lines = [], []
    for i, (_, col) in enumerate(picks):
        vals = table[col].tolist()
        base = vals[0] if vals and vals[0] else None
        idx = [v / base * 100 if base else float("nan") for v in vals]
        series.append(_series_dict(col, idx, SERIES_COLORS[i]))
        facts_lines += _first_last_facts(labels, vals, col)
    title = "Customers vs workforce growth (index, first period = 100)"
    plan = _base_plan(
        "customers_workforce", "line", title, labels, series,
        xlabel=GRAN_NAME[gran], ylabel="Index (first period = 100)", y_fmt="index",
        reason="Both measures are indexed to 100 so their different sizes are comparable.",
    )
    plan["facts"] = "\n".join(_facts_header(title, ["Values are indexed to the first period = 100."]) + facts_lines)
    return plan


# -- business growth vs financial growth -------------------------------------
def _plan_growth_compare(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    wanted = []
    for concept in ("customers", "employees", "revenue", "net_profit"):
        col = find_concept_column(df, concept)
        if col is not None:
            wanted.append((concept, col))
    business = [w for w in wanted if w[0] in ("customers", "employees")]
    financial = [w for w in wanted if w[0] in ("revenue", "net_profit")]
    if not financial:
        return None
    if not business:
        # No operational metrics in the file: compare the financial measures' growth.
        wanted = financial
    if len(wanted) < 2:
        return None

    months = frame["_date"].dt.to_period("M").nunique()
    gran = "Y" if _year_count(frame) >= 3 else choose_granularity(frame["_date"], question)
    if months <= 12:
        gran = choose_granularity(frame["_date"], question)
    cols = {col: _metric_agg(concept, col) for concept, col in wanted}
    table = aggregate(frame, cols, gran)
    labels = list(table.index)

    series, lines = [], []
    for i, (concept, col) in enumerate(wanted):
        vals = table[col].tolist()
        base = next((v for v in vals if v), None)
        idx = [v / base * 100 if base else float("nan") for v in vals]
        kind = "business" if concept in ("customers", "employees") else "financial"
        series.append(_series_dict(f"{col} ({kind})", idx, SERIES_COLORS[i]))
        lines.append(f"{col} ({kind}): {_fmt_num(vals[0], 2)} -> {_fmt_num(vals[-1], 2)} "
                     f"= index {idx[-1]:.1f} (first {GRAN_NAME[gran].lower()} = 100, "
                     f"{_pct(idx[-1] - 100)} total).")
    title = f"Business growth vs financial growth ({GRAN_ADJ[gran]}, index first = 100)"
    assumptions = ["Growth is shown as an index (first period = 100) so customers/employees and "
                   "money amounts can share one axis."]
    plan = _base_plan(
        "growth_compare", "line", title, labels, series,
        xlabel=GRAN_NAME[gran], ylabel="Index (first period = 100)", y_fmt="index",
        assumptions=assumptions,
        reason="Indexed growth lines: if financial lines rise faster than business lines, growth is translating into results.",
    )
    facts = _facts_header(title, assumptions) + lines
    if len(series) >= 2:
        facts.append("If a financial line ends above the business lines, financial results grew faster than activity.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- Revenue bars + margin line ---------------------------------------------
def _margin_table(frame, df, gran, want=("net",)):
    """Aggregate revenue + profit columns, then derive margins AFTER aggregation."""
    rev = find_concept_column(df, "revenue")
    net = find_concept_column(df, "net_profit")
    ebt = find_concept_column(df, "ebitda")
    gp = find_concept_column(df, "gross_profit")
    cols = {c: "sum" for c in (rev, net, ebt, gp) if c is not None}
    if rev is None or not cols:
        return None, {}
    table = aggregate(frame, cols, gran)
    margins = {}
    if net is not None:
        margins["Net margin %"] = table[net] / table[rev] * 100
    if ebt is not None:
        margins["EBITDA margin %"] = table[ebt] / table[rev] * 100
    if gp is not None:
        margins["Gross margin %"] = table[gp] / table[rev] * 100
    return table, {"rev": rev, "net": net, "ebitda": ebt, "gp": gp, "margins": margins}


def _plan_profit_vs_scale(df, question, sheets):
    return _combo_revenue_margin(df, question, "profit_vs_scale")


def _plan_profitability(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    net = find_concept_column(df, "net_profit")
    rev = find_concept_column(df, "revenue")
    if net is not None and rev is not None:
        return _combo_revenue_margin(df, question, "profitability", bar_metric="net")
    return None


def _combo_revenue_margin(df, question, intent, bar_metric="revenue"):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    gran = choose_granularity(frame["_date"], question)
    table, info = _margin_table(frame, df, gran)
    if table is None or "Net margin %" not in info["margins"]:
        return None
    labels = list(table.index)
    margin = info["margins"]["Net margin %"]
    if bar_metric == "net":
        bar_col = info["net"]
    else:
        bar_col = info["rev"]
    bars = table[bar_col].tolist()
    m_vals = margin.tolist()
    title = f"{bar_col} vs Net margin % ({GRAN_ADJ[gran]})"
    plan = _base_plan(
        intent, "combo", title, labels,
        [_series_dict(bar_col, bars, DARK), _series_dict("Net margin %", m_vals, YELLOW, fmt="pct")],
        xlabel=GRAN_NAME[gran], ylabel=str(bar_col), y2label="Net margin %",
        reason=("Bars show the size of the business; the line shows how much of each unit "
                "of revenue is kept as profit. A rising line as bars grow = more profitable at scale."),
    )
    valid = [(l, m) for l, m in zip(labels, m_vals) if not pd.isna(m)]
    facts = _facts_header(title, ["Net margin % = Net Profit / Revenue for each period."])
    facts += [f"  - {l}: {bar_col} {_fmt_num(b, 2)}, Net margin {m:.2f}%"
              for l, b, m in zip(labels, bars, m_vals) if not pd.isna(m)]
    if len(valid) >= 3:
        x = np.array(bars, dtype=float)
        y = np.array(m_vals, dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            facts.append(f"Correlation between {bar_col} and Net margin %: r = {r:.2f}.")
        facts.append(f"Net margin moved from {valid[0][1]:.2f}% ({valid[0][0]}) to "
                     f"{valid[-1][1]:.2f}% ({valid[-1][0]}); average {np.nanmean(m_vals):.2f}%.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- relationship (scatter) --------------------------------------------------
def _plan_relationship(df, question, sheets):
    rev = find_concept_column(df, "revenue")
    prof = find_concept_column(df, "net_profit") or find_concept_column(df, "ebitda")
    if rev is None or prof is None:
        return None
    frame = _frame_or_none(df, question)
    work = frame if frame is not None else df
    x = _as_number(work[rev])
    y = _as_number(work[prof])
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return None
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    if frame is not None:
        labels = work.loc[ok[ok].index, "_date"].dt.strftime("%b %Y").tolist()
    else:
        labels = [str(i + 1) for i in range(len(x))]
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")

    title = f"{prof} vs {rev}"
    plan = _base_plan(
        "relationship", "scatter", title, labels, [],
        x=x.tolist(), y=y.tolist(), xlabel=str(rev), ylabel=str(prof),
        slope=float(slope), intercept=float(intercept), r=r,
        reason="Each dot is one period; the line is the best-fit trend.",
    )
    facts = _facts_header(title, [f"'Income' was interpreted as '{rev}' and 'profitability' as '{prof}'."])
    facts.append(f"{len(x)} periods plotted. Correlation r = {r:.2f}.")
    facts.append(f"Best-fit slope: each extra 1.00 of {rev} goes with {slope:+.3f} of {prof}.")
    margin_note = ""
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(x != 0, y / x * 100, np.nan)
    if not np.all(np.isnan(m)) and len(m) >= 6:
        half = len(m) // 2
        margin_note = (f"Average {prof}/{rev} margin: {np.nanmean(m[:half]):.2f}% in the first half of the data "
                       f"vs {np.nanmean(m[half:]):.2f}% in the second half.")
        facts.append(margin_note)
    plan["facts"] = "\n".join(facts)
    return plan


# -- costs vs income ---------------------------------------------------------
def _plan_costs_vs_income(df, question, sheets):
    frame = _frame_or_none(df, question)
    rev = find_concept_column(df, "revenue")
    costs = find_cost_columns(df)
    if frame is None or rev is None or not costs:
        return None
    months = frame["_date"].dt.to_period("M").nunique()
    quarters = frame["_date"].dt.to_period("Q").nunique()
    gran = choose_granularity(frame["_date"], question, prefer="Q" if quarters >= 8 else "Y")
    if gran == "M" and months > 36:
        gran = "Q"
    cols = {rev: "sum", **{c: "sum" for c in costs}}
    table = aggregate(frame, cols, gran)
    total_cost = table[costs].sum(axis=1)
    revenue = table[rev]
    lag = _lag_for(gran) if len(table) > _lag_for(gran) + 1 else 1
    lag_name = {12: "same month last year", 4: "same quarter last year", 1: "previous year"}.get(lag, "previous period")
    if lag == 1 and gran != "Y":
        lag_name = "previous period"

    labels, rev_g, cost_g = [], [], []
    for i in range(lag, len(table)):
        labels.append(table.index[i])
        rev_g.append(_safe_pct_change(revenue.iloc[i], revenue.iloc[i - lag]))
        cost_g.append(_safe_pct_change(total_cost.iloc[i], total_cost.iloc[i - lag]))
    if len(labels) < 2:
        return None
    flagged = [i for i, (r, c) in enumerate(zip(rev_g, cost_g))
               if not pd.isna(r) and not pd.isna(c) and c > r]

    title = f"Cost growth vs {rev} growth (vs {lag_name})"
    cost_names = ", ".join(map(str, costs))
    assumptions = [f"Total operating cost = {cost_names} (interest, tax, depreciation and capex excluded).",
                   f"Growth is measured against the {lag_name} to remove seasonality."]
    plan = _base_plan(
        "costs_vs_income", "growth_bars", title, labels,
        [_series_dict(f"{rev} growth %", rev_g, DARK, fmt="pct"),
         _series_dict("Total cost growth %", cost_g, ORANGE, fmt="pct")],
        xlabel=GRAN_NAME[gran], ylabel="Growth %", y_fmt="pct", flagged=flagged,
        assumptions=assumptions,
        reason="Shaded periods are where total costs grew faster than revenue.",
    )
    facts = _facts_header(title, assumptions)
    facts.append(f"{len(flagged)} of {len(labels)} periods had cost growth above revenue growth.")
    for i in flagged:
        facts.append(f"  - {labels[i]}: cost {cost_g[i]:+.1f}% vs revenue {rev_g[i]:+.1f}% "
                     f"(gap {cost_g[i] - rev_g[i]:+.1f} pts).")
    facts.append(f"Overall {table.index[0]} -> {table.index[-1]}: revenue {_pct(_safe_pct_change(revenue.iloc[-1], revenue.iloc[0]))}, "
                 f"total cost {_pct(_safe_pct_change(total_cost.iloc[-1], total_cost.iloc[0]))}.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- efficiency --------------------------------------------------------------
def _plan_efficiency(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    gran = choose_granularity(frame["_date"], question)
    table, info = _margin_table(frame, df, gran)
    if table is None or not info["margins"]:
        return None
    wanted = [k for k in ("Gross margin %", "EBITDA margin %", "Net margin %") if k in info["margins"]]
    labels = list(table.index)
    series = [_series_dict(k, info["margins"][k].tolist(), SERIES_COLORS[i], fmt="pct")
              for i, k in enumerate(wanted)]
    title = f"Financial efficiency — profit margins over time ({GRAN_ADJ[gran]})"
    assumptions = ["Efficiency is measured as profit margins: how much of each unit of revenue is kept as profit."]
    plan = _base_plan(
        "efficiency", "line", title, labels, series,
        xlabel=GRAN_NAME[gran], ylabel="Margin (% of revenue)", y_fmt="pct", assumptions=assumptions,
        reason="Rising margins mean the company is converting more of its revenue into profit.",
    )
    facts = _facts_header(title, assumptions)
    for k in wanted:
        vals = info["margins"][k].tolist()
        facts.append(f"{k}: {labels[0]} {vals[0]:.2f}% -> {labels[-1]} {vals[-1]:.2f}%; "
                     f"average {np.nanmean(vals):.2f}%; best {np.nanmax(vals):.2f}% ({labels[int(np.nanargmax(vals))]}), "
                     f"worst {np.nanmin(vals):.2f}% ({labels[int(np.nanargmin(vals))]}).")
    plan["facts"] = "\n".join(facts)
    return plan


# -- operating vs overall performance ---------------------------------------
def _plan_operating_vs_overall(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    op = find_concept_column(df, "ebitda") or find_concept_column(df, "gross_profit")
    overall = find_concept_column(df, "net_profit")
    if op is None or overall is None:
        return None
    gran = choose_granularity(frame["_date"], question)
    table = aggregate(frame, {op: "sum", overall: "sum"}, gran)
    labels = list(table.index)
    title = f"Operating performance ({op}) vs overall financial performance ({overall})"
    plan = _base_plan(
        "operating_vs_overall", "line", title, labels,
        [_series_dict(str(op), table[op].tolist(), DARK), _series_dict(str(overall), table[overall].tolist(), YELLOW)],
        xlabel=GRAN_NAME[gran], ylabel="Amount",
        reason=f"'{op}' captures operating results; '{overall}' is what is left after depreciation, interest and tax.",
    )
    conv = (table[overall] / table[op] * 100).replace([np.inf, -np.inf], np.nan)
    facts = _facts_header(title, [f"'Operating performance' = {op}; 'overall financial performance' = {overall}."])
    facts += _first_last_facts(labels, table[op].tolist(), str(op))
    facts += _first_last_facts(labels, table[overall].tolist(), str(overall))
    facts.append(f"On average {overall} was {np.nanmean(conv):.1f}% of {op}.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- largest improvements ----------------------------------------------------
def _plan_largest_improvements(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    concept, col = _primary_metric(df, question)
    if col is None:
        return None
    gran = choose_granularity(frame["_date"], question, prefer="Q")
    agg = _metric_agg(concept, col)
    table = aggregate(frame, {col: agg}, gran)
    vals = table[col].tolist()
    labels, changes = [], []
    for i in range(1, len(vals)):
        labels.append(f"{table.index[i]}")
        changes.append(_safe_pct_change(vals[i], vals[i - 1]))
    if len(labels) < 2:
        return None
    order = [i for i in np.argsort(np.nan_to_num(changes, nan=-1e18))[::-1]][:3]
    top = [i for i in order if not pd.isna(changes[i]) and changes[i] > 0]

    title = f"{col}: period-over-period change ({GRAN_ADJ[gran]})"
    plan = _base_plan(
        "largest_improvements", "change_bars", title, labels,
        [_series_dict(f"{col} change %", changes, DARK, fmt="pct")],
        xlabel=GRAN_NAME[gran], ylabel=f"Change vs previous {GRAN_NAME[gran].lower()} (%)", y_fmt="pct",
        highlight=top,
        assumptions=[f"'Performance' was measured with '{col}'."],
        reason="Yellow bars mark the three biggest improvements; grey bars are declines.",
    )
    facts = _facts_header(title, plan["assumptions"])
    ranked = sorted([(c, l, vals[i + 1], vals[i]) for i, (c, l) in enumerate(zip(changes, labels)) if not pd.isna(c)],
                    reverse=True)
    facts.append("Largest improvements:")
    for c, l, cur, prev in ranked[:5]:
        facts.append(f"  - {l}: {c:+.1f}% ({_fmt_num(prev, 2)} -> {_fmt_num(cur, 2)})")
    facts.append("Largest declines:")
    for c, l, cur, prev in ranked[::-1][:3]:
        facts.append(f"  - {l}: {c:+.1f}% ({_fmt_num(prev, 2)} -> {_fmt_num(cur, 2)})")
    plan["facts"] = "\n".join(facts)
    return plan


# -- anomalies ---------------------------------------------------------------
def _plan_anomalies(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    q = question.lower()
    gran = choose_granularity(frame["_date"], question, prefer="M")
    if gran == "M" and frame["_date"].dt.to_period("M").nunique() > 72:
        gran = "Q"
    table, info = _margin_table(frame, df, gran)
    use_margin = table is not None and "Net margin %" in info["margins"] \
        and re.search(r"profitab|margin|profit", q)
    if use_margin:
        series_vals = info["margins"]["Net margin %"]
        name, y_fmt = "Net margin %", "pct"
        assumptions = ["'Profitability' was measured as Net margin % (Net Profit / Revenue)."]
    else:
        concept, col = _primary_metric(df, question)
        if col is None:
            return None
        agg = _metric_agg(concept, col)
        t2 = aggregate(frame, {col: agg}, gran)
        series_vals, name, y_fmt = t2[col], str(col), "num"
        assumptions = [f"No specific measure named; '{col}' was used."]
        table = t2
    labels = list(series_vals.index)
    vals = series_vals.to_numpy(dtype=float)
    mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
    threshold = 1.5
    z = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
    high = [i for i in range(len(vals)) if z[i] >= threshold]
    low = [i for i in range(len(vals)) if z[i] <= -threshold]
    if not high and not low and sd > 0:  # nothing extreme: still show the top/bottom point
        high, low = [int(np.nanargmax(vals))], [int(np.nanargmin(vals))]
        assumptions.append("No period exceeded 1.5 standard deviations, so only the single highest and lowest are marked.")

    title = f"Unusually high / low {name} ({GRAN_ADJ[gran]})"
    plan = _base_plan(
        "anomalies", "line", title, labels, [_series_dict(name, vals.tolist(), DARK, fmt=y_fmt)],
        xlabel=GRAN_NAME[gran], ylabel=name, y_fmt=y_fmt,
        band=(mu - threshold * sd, mu + threshold * sd), mean_line=mu,
        markers=[{"i": i, "kind": "high"} for i in high] + [{"i": i, "kind": "low"} for i in low],
        assumptions=assumptions,
        reason="Grey band = average ± 1.5 standard deviations; points outside it are flagged.",
    )
    facts = _facts_header(title, assumptions)
    fmt = (lambda v: f"{v:.2f}%") if y_fmt == "pct" else (lambda v: _fmt_num(v, 2))
    facts.append(f"Average {fmt(mu)}, standard deviation {fmt(sd) if y_fmt != 'pct' else f'{sd:.2f} pts'}.")
    facts.append("Unusually high: " + (", ".join(f"{labels[i]} ({fmt(vals[i])})" for i in high) or "none"))
    facts.append("Unusually low: " + (", ".join(f"{labels[i]} ({fmt(vals[i])})" for i in low) or "none"))
    plan["facts"] = "\n".join(facts)
    return plan


# -- before / after ----------------------------------------------------------
def _plan_before_after(df, question, sheets):
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    concept, col = _primary_metric(df, question)
    if col is None:
        return None
    gran = choose_granularity(frame["_date"], question, prefer="Q")
    agg = _metric_agg(concept, col)
    table = aggregate(frame, {col: agg}, gran)
    labels, vals = list(table.index), table[col].to_numpy(dtype=float)
    if len(vals) < 6:
        return None

    # 1) explicit year in the question ("after 2023")
    split, source = None, None
    m = re.search(r"\b(19|20)\d{2}\b", question)
    if m:
        year_label = m.group(0)
        for i, l in enumerate(labels):
            if str(l).startswith(year_label) and 2 <= i <= len(labels) - 2:
                split, source = i, f"the start of {year_label} (named in the question)"
                break
    # 2) otherwise the strongest level shift
    if split is None:
        best, best_gap = None, -1.0
        pooled = np.nanstd(vals) or 1.0
        min_seg = max(3, len(vals) // 4)
        for i in range(min_seg, len(vals) - min_seg + 1):
            gap = abs(np.nanmean(vals[i:]) - np.nanmean(vals[:i])) / pooled
            if gap > best_gap:
                best, best_gap = i, gap
        split = best
        source = "the biggest shift in average level found in the data (no specific event was named)"
    before, after = vals[:split], vals[split:]
    b_mean, a_mean = float(np.nanmean(before)), float(np.nanmean(after))
    change = _safe_pct_change(a_mean, b_mean)

    title = f"{col}: before vs after {labels[split]}"
    assumptions = [f"'Major change' = {source}: the split is at {labels[split]}."]
    plan = _base_plan(
        "before_after", "line", title, labels, [_series_dict(str(col), vals.tolist(), DARK)],
        xlabel=GRAN_NAME[gran], ylabel=str(col), split=split, before_mean=b_mean, after_mean=a_mean,
        assumptions=assumptions,
        reason="Dashed line = the split point; horizontal segments = the average before and after it.",
    )
    facts = _facts_header(title, assumptions)
    facts.append(f"Before ({labels[0]}–{labels[split - 1]}, {len(before)} periods): average {_fmt_num(b_mean, 2)}.")
    facts.append(f"After ({labels[split]}–{labels[-1]}, {len(after)} periods): average {_fmt_num(a_mean, 2)}.")
    facts.append(f"Average changed by {_pct(change)}.")
    facts.append("The user did not name a specific event; invite them to name a date/year to re-split the chart.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- financial-year comparison ("compare FY24 and FY25", "FY23 vs FY25") ----
def _plan_fy_compare(df, question, sheets):
    frame = _frame_or_none_raw(df)
    if frame is None:
        return None
    fys = detect_fiscal_years(question, ref_date=frame["_date"].max())
    if len(fys) < 2:
        return None
    named = _named_metric_columns(df, question)
    concept, col = (None, named[0]) if named else _primary_metric(df, question, prefer=("revenue", "net_profit", "ebitda"))
    if col is None:
        return None
    agg = _metric_agg(concept, col)
    labels, values, missing, partial = [], [], [], []
    for label, start, end in fys:
        seg = frame[(frame["_date"] >= start) & (frame["_date"] <= end)]
        if seg.empty:
            missing.append(label)
            continue
        n_mo = int(seg["_date"].dt.to_period("M").nunique())
        expected = int(round(((end.year - start.year) * 12 + end.month - start.month) + 1))
        if isinstance(label, str) and label.endswith("YTD"):
            expected = n_mo
        partial.append((label, n_mo, expected)) if n_mo < expected else None
        labels.append(f"{label} ({n_mo} mo)" if n_mo < expected else label)
        values.append(float(_as_number(seg[col]).agg(agg)))
    if len(labels) < 1:
        return None
    if len(labels) < 2:
        # Only one of the named years actually has data — say so rather than
        # drawing a one-bar "comparison".
        return None

    title = f"{col}: " + " vs ".join(labels)
    plan = _base_plan(
        "fy_compare", "bars", title, labels, [_series_dict(str(col), values, DARK)],
        xlabel="Financial year (Apr-Mar)", ylabel=str(col),
        reason="Each bar totals the measure over that financial year (1 Apr - 31 Mar).",
    )
    assumptions = [f"'{l}' = {s.strftime('%b %Y')} - {e.strftime('%b %Y')} (Apr-Mar financial year)."
                   for l, s, e in fys if any(str(x).startswith(l) for x in labels)]
    for lab, n_mo, expected in partial:
        assumptions.append(f"{lab} only has {n_mo} of {expected} months in the data, so its total is not "
                           "comparable with a full period.")
    if missing:
        assumptions.append(f"No data was found for: {', '.join(missing)} — those years are omitted from the chart.")
    plan["assumptions"] = assumptions
    facts = _facts_header(title, assumptions)
    for l, v in zip(labels, values):
        facts.append(f"  - {l}: {_fmt_num(v, 2)}")
    if len(values) >= 2:
        facts.append(f"Change from {labels[0]} to {labels[-1]}: {_pct(_safe_pct_change(values[-1], values[0]))}.")
    plan["facts"] = "\n".join(facts)
    return plan


# -- financial-year overview ("revenue by financial year", "FY wise profit") --
def _plan_fy_overview(df, question, sheets):
    frame = _frame_or_none(df, question)          # flagged fiscal (Apr-Mar)
    if frame is None:
        return None
    cols = _named_metric_columns(df, question)
    if not cols:
        _, col = _primary_metric(df, question, prefer=("revenue", "net_profit", "ebitda"))
        cols = [col] if col is not None else []
    if not cols:
        return None
    aggs = {c: _metric_agg(next((k for k, v in CONCEPTS.items() if find_concept_column(df, k) == c), None), c)
            for c in cols}
    gran = choose_granularity(frame["_date"], question, prefer="Y")
    table = aggregate(frame, aggs, gran)
    labels = list(table.index)
    if len(labels) < 2:
        return None
    assumptions = ["Financial year = 1 April to 31 March; FY25 = Apr 2024 - Mar 2025; "
                   "fiscal quarters: Q1 = Apr-Jun, Q2 = Jul-Sep, Q3 = Oct-Dec, Q4 = Jan-Mar."]
    partial = [l for l in labels if "mo)" in str(l)]
    if partial:
        assumptions.append("Periods marked '(n mo)' only have n months of data in the file, "
                           "so they are NOT comparable with full periods.")
    means = [abs(float(np.nanmean(table[c]))) or 1.0 for c in cols]
    if len(cols) > 1 and max(means) / max(min(means), 1e-9) > 5:
        assumptions.append(f"Only '{cols[0]}' is plotted; the other measures are on a very different scale.")
        cols = cols[:1]
    series = [_series_dict(str(c), table[c].tolist(), SERIES_COLORS[i]) for i, c in enumerate(cols)]
    unit = {"Y": "financial year", "Q": "fiscal quarter", "M": "month"}[gran]
    title = f"{' & '.join(map(str, cols))} by {unit} (Apr-Mar)"
    kind = "line" if gran == "M" else ("bars" if len(cols) == 1 else "multi_bars")
    plan = _base_plan(
        "fy_overview", kind, title, labels, series,
        xlabel={"Y": "Financial year (Apr-Mar)", "Q": "Fiscal quarter", "M": "Month"}[gran],
        ylabel=str(cols[0]) if len(cols) == 1 else "Amount",
        assumptions=assumptions, legacy_compatible=False,
        reason="Each bar totals the measure over that Apr-Mar period.")
    facts = _facts_header(title, assumptions)
    for c in cols:
        vals = table[c].tolist()
        facts += [f"  {c} - {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, vals)]
        full = [(l, v) for l, v in zip(labels, vals) if "mo)" not in str(l)]
        if len(full) >= 2:
            facts.append(f"{c}: {full[0][0]} -> {full[-1][0]} = "
                         f"{_pct(_safe_pct_change(full[-1][1], full[0][1]))} (full periods only).")
    plan["facts"] = "\n".join(facts)
    return plan


# -- monthly vs quarterly vs annual -----------------------------------------
def _sheet_frame(sheets, keywords):
    for name, sdf in (sheets or {}).items():
        if any(k in str(name).lower() for k in keywords) and sdf is not None and not sdf.empty:
            return sdf
    return None


def _plan_granularity_compare(df, question, sheets):
    concept, col = _primary_metric(df, question, prefer=("revenue", "net_profit", "ebitda"))
    if col is None:
        return None
    panels, facts_lines = [], []
    agg = _metric_agg(concept, col)
    frame = _frame_or_none(df, question)
    native = _native_granularity(frame["_date"]) if frame is not None else "Y"

    def add_panel(gran, source_df, origin):
        f = _frame_or_none(source_df)
        c = col if col in source_df.columns else find_concept_column(source_df, concept)
        if f is None or c is None:
            return
        t = aggregate(f, {c: agg}, gran)
        if len(t) < 2:
            return
        vals = t[c].tolist()
        chg = [_safe_pct_change(vals[i], vals[i - 1]) for i in range(1, len(vals))]
        volatility = float(np.nanstd(chg)) if len(chg) else float("nan")
        panels.append({"title": f"{GRAN_ADJ[gran].title()} ({origin})", "labels": list(t.index), "values": vals})
        facts_lines.append(
            f"{GRAN_ADJ[gran].title()}: {len(vals)} periods, average {_fmt_num(float(np.nanmean(vals)), 2)}, "
            f"typical period-to-period swing ±{volatility:.1f}% (std dev of % change).")

    for gran, keys in (("M", ("month",)), ("Q", ("quarter",)), ("Y", ("annual", "year"))):
        sheet = _sheet_frame(sheets, keys)
        if sheet is not None and _GRAN_RANK[gran] >= 0:
            add_panel(gran, sheet, "workbook sheet")
        elif frame is not None and _GRAN_RANK[gran] >= _GRAN_RANK[native]:
            add_panel(gran, df, "derived")
    if len(panels) < 2:
        return None
    title = f"{col}: monthly vs quarterly vs annual"
    plan = _base_plan(
        "granularity_compare", "panels", title, [], [], panels=panels, ylabel=str(col),
        assumptions=[f"'Performance' was measured with '{col}'."],
        reason="Same measure at three levels of detail: shorter periods are noisier, longer periods reveal the trend.",
    )
    plan["facts"] = "\n".join(_facts_header(title, plan["assumptions"]) + facts_lines)
    return plan


# ---------------------------------------------------------------------------
# 5b. Named metrics ("plot Marketing Expense over time", "x axis period,
#     y axis revenue", "quarterly revenue chart")
# ---------------------------------------------------------------------------
_TIME_AXIS_WORDS = {"period", "date", "time", "month", "months", "year", "years", "quarter",
                    "quarters", "timeline", "day", "days", "week", "weeks"}
_TIME_CUES = re.compile(
    r"over time|trend|monthly|quarterly|yearly|annual|annually|by (?:month|quarter|year|period)|"
    r"per (?:month|quarter|year)|each (?:month|quarter|year)|across (?:months|quarters|years|periods)|"
    r"time series|timeline|growth|history|changes?\b|\bfy\s?\d|financial year|fiscal year"
)
_SYNONYM_CONCEPTS = [
    (r"\b(?:revenue|sales|turnover|income|top[- ]line)\b", "revenue"),
    (r"\b(?:net profit|net income|profit|profits|profitability|earnings|pat)\b", "net_profit"),
    (r"\bgross profit\b", "gross_profit"),
    (r"\b(?:ebitda|operating profit)\b", "ebitda"),
    (r"\b(?:customers?|clients?|subscribers?)\b", "customers"),
    (r"\b(?:employees?|headcount|staff|workforce)\b", "employees"),
    (r"\bcash flow\b", "cash_flow"),
]


def is_time_axis_word(text) -> bool:
    """'period', 'date', 'month' ... (also tolerates parser leftovers such as
    'period y axis revenue' by looking at the first word)."""
    if not text:
        return False
    words = _norm(text).split()
    return bool(words) and (words[0] in _TIME_AXIS_WORDS or _norm(text) in _TIME_AXIS_WORDS)


def mentions_any_column(df: pd.DataFrame, question: str) -> bool:
    """True if the question names a real column, or a common measure word
    (sales/revenue/amount/...). Used to decide whether a question is
    'generic' (column-free) and therefore safe for the AI planner."""
    q = (question or "").lower()
    for c in df.columns:
        n = _norm(c)
        if len(n) >= 3 and n not in _TIME_AXIS_WORDS and re.search(rf"\b{re.escape(n)}\b", q):
            return True
    return bool(re.search(r"\b(sales|revenue|amount|total|value|income|profit|cost|expense)\b", q))


def _named_metric_columns(df: pd.DataFrame, question: str, hint_text: str = "") -> list:
    """Numeric columns the user named literally, else via synonym concepts."""
    q = f"{question or ''} {hint_text or ''}".lower()
    numeric = [c for c in _numeric_columns(df) if not _is_time_like_name(c)]
    found = []
    for c in numeric:
        n = _norm(c)
        if len(n) >= 3:
            m = re.search(rf"\b{re.escape(n)}\b", q)
            if m:
                found.append((m.start(), c))
    found.sort()
    cols = [c for _, c in found]
    if not cols:
        for pat, concept in _SYNONYM_CONCEPTS:
            m = re.search(pat, q)
            if m:
                c = find_concept_column(df, concept)
                if c is not None and c not in cols:
                    cols.append(c)
        # keep only the first two so the chart stays readable
    # drop columns fully contained in a longer named column ("expense" in "marketing expense")
    return cols[:3]


def _chart_hint_from_question(q: str):
    q = (q or "").lower()
    for hint, words in (("pie", ("pie",)), ("area", ("area chart", "area graph")),
                        ("barh", ("horizontal bar",)), ("bar", ("bar chart", "bar graph", "bars")),
                        ("line", ("line chart", "line graph", "line plot"))):
        if any(w in q for w in words):
            return hint
    return None


def plan_named_metrics(df: pd.DataFrame, question: str, axis_hints=("", "")):
    """
    A metric is named (or implied by a synonym) and the question is about time:
    draw it CHRONOLOGICALLY (Jan 2021 ... Dec 2025) instead of collapsing all
    years into 12 month buckets, and honour "monthly / quarterly / yearly".
    Returns None when there is no date dimension or no metric could be found.
    """
    hint_text = " ".join(t for t in axis_hints if t and not is_time_axis_word(t))
    axis_time = any(is_time_axis_word(t) for t in axis_hints)
    q = (question or "").lower()
    if not (axis_time or _TIME_CUES.search(q)):
        return None
    frame = _frame_or_none(df, question)
    if frame is None:
        return None
    months = frame["_date"].dt.to_period("M").nunique()
    explicit_gran = bool(re.search(r"monthly|quarterly|yearly|annual|by (?:month|quarter|year)|each (?:month|quarter|year)", q))
    scoped = bool(frame.attrs.get("fy_scoped"))
    if months <= 12 and not scoped and not axis_time and not (explicit_gran and _find_quarter_column(df) is not None):
        return None  # single year: the original monthly/fiscal-quarter branch already does this well
    if months <= 12 and not scoped and "quarter" in q and _find_quarter_column(df) is None:
        return None  # keep FISCAL_YEAR_START_MONTH behaviour of the original branch
    cols = _named_metric_columns(df, q, hint_text)
    if not cols:
        concept, col = _primary_metric(df, q, prefer=("revenue", "net_profit", "ebitda"))
        if col is None:
            return None
        cols = [col]
    gran = choose_granularity(frame["_date"], question)
    aggs = {c: _metric_agg(next((k for k, v in CONCEPTS.items() if find_concept_column(df, k) == c), None), c)
            for c in cols}
    table = aggregate(frame, aggs, gran)
    labels = list(table.index)
    if len(labels) < 2:
        return None
    hint = _chart_hint_from_question(q)

    if len(cols) == 1:
        c = cols[0]
        vals = table[c].tolist()
        label = _label_for(None, c, aggs[c])
        title = f"{label} over time ({GRAN_ADJ[gran]})" + _fy_suffix(frame)
        plan = _base_plan("named_metric", "line", title, labels, [_series_dict(label, vals, DARK)],
                          xlabel=GRAN_NAME[gran], ylabel=label, show_trend=True, annotate_extremes=True,
                          legacy_compatible=True, is_time_series=True, chart_type_hint=hint,
                          reason=f"{label} shown chronologically by {GRAN_NAME[gran].lower()}.")
        facts = _facts_header(title, [f"'{c}' aggregated by {GRAN_NAME[gran].lower()} ({aggs[c]})."])
        facts += [f"  - {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, vals)]
        facts += _first_last_facts(labels, vals, label)
        plan["facts"] = "\n".join(facts)
        return plan

    means = [abs(float(np.nanmean(table[c]))) or 1.0 for c in cols]
    scale_gap = max(means) / max(min(means), 1e-9)
    series, lines = [], []
    for i, c in enumerate(cols):
        vals = table[c].tolist()
        if scale_gap > 5:
            base = next((v for v in vals if v), None)
            vals_plot = [v / base * 100 if base else float("nan") for v in vals]
        else:
            vals_plot = vals
        series.append(_series_dict(str(c), vals_plot, SERIES_COLORS[i]))
        lines += _first_last_facts(labels, vals, str(c))
    indexed = scale_gap > 5
    title = (f"{' vs '.join(map(str, cols))} over time ({GRAN_ADJ[gran]}"
             + (", index first = 100)" if indexed else ")"))
    plan = _base_plan("named_metric", "line", title, labels, series, xlabel=GRAN_NAME[gran],
                      ylabel="Index (first period = 100)" if indexed else "Amount",
                      y_fmt="index" if indexed else "num")
    plan["facts"] = "\n".join(_facts_header(title, ["Measures were indexed to 100 because their sizes differ a lot."] if indexed else []) + lines)
    plan["reason"] = "Each measure plotted chronologically."
    return plan


# ---------------------------------------------------------------------------
# 5c. AI planner (for questions the rules do not recognise)
# ---------------------------------------------------------------------------
_AI_CHARTS = ("line", "bar", "barh", "area", "pie", "scatter")
_AI_AGGS = ("sum", "mean", "count", "max", "min")


def _json_from_text(raw):
    import json
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {}


def _ai_profile(df: pd.DataFrame, frame_date_col) -> list:
    numeric = set(_numeric_columns(df))
    out = []
    for c in list(df.columns)[:40]:
        s = df[c]
        if c in numeric:
            n = _as_number(s).dropna()
            entry = {"name": str(c), "type": "number",
                     "min": float(n.min()) if len(n) else None, "max": float(n.max()) if len(n) else None}
        elif c == frame_date_col:
            entry = {"name": str(c), "type": "date"}
        else:
            entry = {"name": str(c), "type": "category", "distinct": int(s.nunique(dropna=True))}
        entry["sample"] = [str(x)[:40] for x in s.dropna().head(3).tolist()]
        out.append(entry)
    return out


def plan_from_ai(df: pd.DataFrame, question: str, ai_service):
    """
    Ask the language model ONLY to choose columns / aggregation / chart type as
    JSON. Every column is validated against the real dataframe and all numbers
    are computed here with pandas — the model never supplies a value.
    Returns a plan dict or None (unsupported / invalid / model unavailable).
    """
    import json
    if ai_service is None or df is None or df.empty:
        return None
    built = build_period_frame(df)
    date_col = built[1] if built else None
    profile = _ai_profile(df, date_col)
    prompt = (
        "You plan a chart for a data-analysis app. Choose ONLY from the real columns below. "
        "Never invent a column. Never compute numbers.\n\n"
        f"DATA PROFILE (rows={len(df)}):\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"USER REQUEST:\n{question}\n\n"
        "Return ONLY JSON:\n"
        '{"supported": true|false, "chart": "line|bar|barh|area|pie|scatter", '
        '"x": "exact column or null", "y": ["exact numeric column", ...] , '
        '"agg": "sum|mean|count|max|min", "time_granularity": "month|quarter|year|none", '
        '"top_n": integer or null, "sort": "desc|asc|none", "title": "short title", '
        '"reason": "one sentence", "assumption": "what you assumed if the request was vague"}\n'
        "Rules: use a date column as x with time_granularity for anything about change over time; "
        "use a category column as x for comparisons/rankings; use scatter with numeric x and y for "
        "relationships; at most 3 y columns; if the data cannot answer the request set supported=false."
    )
    try:
        res = ai_service.get_chat_response(
            prompt, system_prompt="Return valid JSON only. No markdown. Never invent columns.")
        spec = _json_from_text((res or {}).get("response") if isinstance(res, dict) else res)
    except Exception:  # noqa: BLE001
        logger.exception("AI chart planner call failed")
        return None
    if not spec:
        return None
    if spec.get("supported") is False:
        # Surface exactly why, instead of silently giving up (the caller
        # turns this into the "data isn't available" message shown to the user).
        raise ChartNotSupported(str(spec.get("reason") or spec.get("assumption") or
                                    "The uploaded data does not contain what this question needs."))

    valid = {str(c): c for c in df.columns}
    numeric = set(_numeric_columns(df))
    chart = str(spec.get("chart", "")).lower()
    if chart not in _AI_CHARTS:
        chart = "bar"
    agg = str(spec.get("agg", "sum")).lower()
    agg = agg if agg in _AI_AGGS else "sum"
    ys = [valid[str(y)] for y in (spec.get("y") or []) if str(y) in valid and valid[str(y)] in numeric][:3]
    x = valid.get(str(spec.get("x"))) if spec.get("x") else None
    gran_word = str(spec.get("time_granularity", "none")).lower()
    top_n = spec.get("top_n") if isinstance(spec.get("top_n"), int) and spec.get("top_n") > 0 else None
    sort = str(spec.get("sort", "none")).lower()
    title = str(spec.get("title") or "")[:120]
    assumption = str(spec.get("assumption") or "").strip()
    assumptions = [f"AI-selected columns: x={x}, y={ys or 'row count'}, aggregation={agg}."]
    if assumption:
        assumptions.append(assumption)

    # scatter
    if chart == "scatter":
        if len(ys) >= 2 or (x in numeric and ys):
            xc, yc = (ys[0], ys[1]) if len(ys) >= 2 else (x, ys[0])
            xv, yv = _as_number(df[xc]), _as_number(df[yc])
            ok = xv.notna() & yv.notna()
            if ok.sum() < 5:
                return None
            xa, ya = xv[ok].to_numpy(), yv[ok].to_numpy()
            slope, intercept = np.polyfit(xa, ya, 1)
            r = float(np.corrcoef(xa, ya)[0, 1]) if np.std(xa) > 0 and np.std(ya) > 0 else float("nan")
            plan = _base_plan("ai_scatter", "scatter", title or f"{yc} vs {xc}", [str(i) for i in range(len(xa))], [],
                              x=xa.tolist(), y=ya.tolist(), xlabel=str(xc), ylabel=str(yc),
                              slope=float(slope), intercept=float(intercept), r=r,
                              reason=str(spec.get("reason") or "Relationship between two measures."))
            plan["assumptions"] = assumptions
            plan["facts"] = "\n".join(_facts_header(plan["title"], assumptions) +
                                       [f"{len(xa)} points, correlation r = {r:.2f}, slope {slope:+.3f}."])
            return plan
        return None

    # over time
    frame = _frame_or_none(df, question)
    is_time = frame is not None and (gran_word in ("month", "quarter", "year") or
                                     (x is not None and x == date_col))
    if is_time and ys:
        want = {"month": "M", "quarter": "Q", "year": "Y"}.get(gran_word)
        gran = choose_granularity(frame["_date"], question, prefer=want)
        aggs = {c: ("mean" if agg == "mean" else "sum") for c in ys}
        table = aggregate(frame, aggs, gran)
        labels = list(table.index)
        if len(labels) < 2:
            return None
        means = [abs(float(np.nanmean(table[c]))) or 1.0 for c in ys]
        indexed = len(ys) > 1 and max(means) / max(min(means), 1e-9) > 5
        series = []
        for i, c in enumerate(ys):
            vals = table[c].tolist()
            if indexed:
                base = next((v for v in vals if v), None)
                vals = [v / base * 100 if base else float("nan") for v in vals]
            series.append(_series_dict(str(c), vals, SERIES_COLORS[i]))
        ttl = title or f"{' vs '.join(map(str, ys))} over time ({GRAN_ADJ[gran]})"
        kind = "line" if chart in ("line", "area", "pie", "scatter") else "multi_bars"
        plan = _base_plan("ai_time", kind, ttl, labels, series, xlabel=GRAN_NAME[gran],
                          ylabel="Index (first = 100)" if indexed else (str(ys[0]) if len(ys) == 1 else "Amount"),
                          y_fmt="index" if indexed else "num", show_trend=(len(ys) == 1 and kind == "line"),
                          legacy_compatible=(len(ys) == 1), is_time_series=True,
                          chart_type_hint=chart if len(ys) == 1 else None,
                          reason=str(spec.get("reason") or ""))
        plan["assumptions"] = assumptions
        facts = _facts_header(ttl, assumptions)
        for c in ys:
            facts += [f"  {c} — {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, table[c].tolist())]
        plan["facts"] = "\n".join(facts)
        return plan

    # category comparison / ranking
    cat = x if (x is not None and x not in numeric and x != date_col) else None
    if cat is None:
        cats = [c for c in df.columns if c not in numeric and c != date_col and 2 <= df[c].nunique(dropna=True) <= 60]
        cat = cats[0] if cats else None
    if cat is None:
        return None
    work = df.copy()
    raw = work[cat].astype(str).str.strip()
    work["_key"] = raw.str.lower().str.replace(r"\s+", " ", regex=True)
    # "West" / "WEST" / "west" are one category: show the most common spelling
    display = raw.groupby(work["_key"]).agg(lambda v: v.value_counts().index[0])
    work["_cat"] = work["_key"].map(display)
    work = work[work["_cat"].ne("") & work["_cat"].str.lower().ne("nan")]
    if agg == "count" or not ys:
        g = work.groupby("_cat").size()
        series_vals = {"Count": g}
    else:
        for c in ys:
            work[c] = _as_number(work[c])
        # Measures on very different scales can't share one axis: keep the first.
        mags = [abs(float(work[c].mean())) or 1.0 for c in ys]
        if len(ys) > 1 and max(mags) / max(min(mags), 1e-9) > 5:
            assumptions.append(f"Only '{ys[0]}' is plotted; {', '.join(map(str, ys[1:]))} "
                               "is on a very different scale (ask for it separately).")
            ys = ys[:1]
        fn = {"sum": "sum", "mean": "mean", "max": "max", "min": "min", "count": "count"}[agg]
        gg = work.groupby("_cat")[ys].agg(fn)
        series_vals = {str(c): gg[c] for c in ys}
    first = next(iter(series_vals.values()))
    order = first.sort_values(ascending=(sort == "asc")).index.tolist() if sort in ("asc", "desc") else \
        first.sort_values(ascending=False).index.tolist()
    limit = top_n or (15 if len(order) > 15 else None)
    if limit:
        order = order[:limit]
    labels = [str(i) for i in order]
    if len(labels) < 1:
        return None
    series = [_series_dict(n, [float(v[i]) for i in order], SERIES_COLORS[j])
              for j, (n, v) in enumerate(series_vals.items())]
    ttl = title or f"{' & '.join(series_vals)} by {cat}"
    single = len(series) == 1
    kind = "bars" if single else "multi_bars"
    plan = _base_plan("ai_category", kind, ttl, labels, series, xlabel=str(cat),
                      ylabel=series[0]["label"] if single else "Amount",
                      legacy_compatible=single, is_time_series=False,
                      chart_type_hint=chart if single else None, reason=str(spec.get("reason") or ""))
    plan["assumptions"] = assumptions + ([f"Showing the top {limit} categories."] if limit else [])
    facts = _facts_header(ttl, plan["assumptions"])
    for s_ in series:
        facts += [f"  {s_['label']} — {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, s_["values"])]
    plan["facts"] = "\n".join(facts)
    return plan


# ---------------------------------------------------------------------------
# 6. Best-effort fallback (chart intent detected but no rule matched)
# ---------------------------------------------------------------------------
def fallback_plan(df: pd.DataFrame, question: str):
    """
    Last-resort chart so a chart request never ends with no chart at all.
    date + number -> line, category + number -> bar, number -> histogram-ish
    bar of the first values, category -> counts.
    """
    if df is None or df.empty:
        return None
    frame = _frame_or_none(df, question)
    concept, col = _primary_metric(df, question, prefer=("revenue", "net_profit", "ebitda"))
    if frame is not None and col is not None:
        gran = choose_granularity(frame["_date"], question)
        agg = _metric_agg(concept, col)
        table = aggregate(frame, {col: agg}, gran)
        labels, vals = list(table.index), table[col].tolist()
        if len(labels) >= 2:
            title = f"{col} over time ({GRAN_ADJ[gran]})"
            plan = _base_plan("fallback_time", "line", title, labels,
                              [_series_dict(str(col), vals, DARK)], xlabel=GRAN_NAME[gran],
                              ylabel=str(col), show_trend=True, annotate_extremes=True,
                              legacy_compatible=True, is_time_series=True,
                              reason=f"Best-effort trend of {col}.")
            plan["assumptions"] = [f"The request was generic, so '{col}' was charted over time."]
            plan["facts"] = "\n".join(_facts_header(title, plan["assumptions"]) +
                                       [f"  - {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, vals)])
            return plan
    # category + number
    cats = [c for c in df.columns if df[c].dtype == object and 2 <= df[c].nunique(dropna=True) <= 25]
    numeric = [c for c in _numeric_columns(df) if not _is_time_like_name(c)]
    if cats and numeric:
        cat, num = cats[0], numeric[0]
        g = df.assign(_n=_as_number(df[num])).groupby(cat)["_n"].sum().sort_values(ascending=False)
        labels, vals = [str(i) for i in g.index], g.tolist()
        title = f"{num} by {cat}"
        plan = _base_plan("fallback_category", "bars", title, labels,
                          [_series_dict(str(num), vals, DARK)], xlabel=str(cat), ylabel=str(num),
                          legacy_compatible=True, is_time_series=False,
                          reason=f"Best-effort breakdown of {num} by {cat}.")
        plan["assumptions"] = [f"The request was generic, so '{num}' was totalled by '{cat}'."]
        plan["facts"] = "\n".join(_facts_header(title, plan["assumptions"]) +
                                   [f"  - {l}: {_fmt_num(v, 2)}" for l, v in zip(labels, vals)])
        return plan
    return None


# ---------------------------------------------------------------------------
# 7. Rendering
# ---------------------------------------------------------------------------
def _axis_formatter(fmt):
    if fmt == "pct":
        return FuncFormatter(lambda v, _: f"{v:.0f}%")
    if fmt == "index":
        return FuncFormatter(lambda v, _: f"{v:.0f}")
    return FuncFormatter(lambda v, _: _fmt_num(v, 1))


def _thin_ticks(ax, labels, max_labels=14):
    n = len(labels)
    step = max(1, math.ceil(n / max_labels))
    ax.set_xticks(range(n))
    ax.set_xticklabels([l if i % step == 0 else "" for i, l in enumerate(labels)],
                       rotation=40 if n > 6 else 0, ha="right" if n > 6 else "center")


def _style_axes(fig, ax, background, text):
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.grid(True, axis="y", color=text, alpha=0.12, linewidth=0.8)
    ax.grid(False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(text)
        ax.spines[side].set_alpha(0.4)
    ax.tick_params(colors=text, labelsize=9)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)


def _contrast(background):
    try:
        h = background.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6 else "#FFFFFF"
    except Exception:  # noqa: BLE001
        return "#111111"


def _nan(v):
    return float("nan") if v is None else v


def render_plan_png(plan, background_color=None, title=None, xlabel=None, ylabel=None,
                    series_color=None, dpi=150, figsize=(10, 6)) -> bytes:
    background = background_color or "#FFFFFF"
    text = _contrast(background)
    kind = plan["kind"]
    title = title or plan["title"]

    if kind == "panels":
        panels = plan["panels"]
        fig, axes = plt.subplots(len(panels), 1, figsize=(figsize[0], 2.6 * len(panels) + 1.2), sharex=False)
        axes = np.atleast_1d(axes)
        for i, (ax, p) in enumerate(zip(axes, panels)):
            _style_axes(fig, ax, background, text)
            xs = range(len(p["labels"]))
            ax.plot(xs, p["values"], color=series_color or SERIES_COLORS[i % 3], linewidth=2, marker="o", markersize=3)
            ax.yaxis.set_major_formatter(_axis_formatter("num"))
            ax.set_title(p["title"], fontsize=10, loc="left", color=text, fontweight="bold")
            _thin_ticks(ax, p["labels"], 12)
            ax.set_ylabel(ylabel or plan.get("ylabel", ""), fontsize=9)
        fig.suptitle(title, fontsize=13, fontweight="bold", color=text)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _to_png(fig, dpi, background)

    fig, ax = plt.subplots(figsize=figsize)
    _style_axes(fig, ax, background, text)
    labels = plan["labels"]
    n = len(labels)
    xs = np.arange(n)
    extra_handles = []

    try:
        if kind == "line":
            if plan.get("band"):
                lo, hi = plan["band"]
                ax.axhspan(lo, hi, color=GREY, alpha=0.18, label="Normal range (±1.5σ)")
                ax.axhline(plan["mean_line"], color=GREY, linestyle=":", linewidth=1)
            for i, s in enumerate(plan["series"]):
                color = series_color if (series_color and len(plan["series"]) == 1) else (s.get("color") or SERIES_COLORS[i % len(SERIES_COLORS)])
                vals = [_nan(v) for v in s["values"]]
                # EY brand hierarchy: black line (legible on white) with
                # yellow marker fill — both brand colors visible on a
                # single-series chart, without a low-contrast yellow line.
                marker_face = YELLOW if (color == DARK and len(plan["series"]) == 1) else color
                ax.plot(xs, vals, color=color, linewidth=2.4, marker="o", markersize=4.5,
                        markerfacecolor=marker_face, markeredgecolor=DARK, markeredgewidth=0.7,
                        label=s["label"])
            if plan.get("show_trend") and len(plan["series"]) == 1:
                v = np.array([_nan(x) for x in plan["series"][0]["values"]], dtype=float)
                ok = ~np.isnan(v)
                if ok.sum() >= 3:
                    k, b = np.polyfit(xs[ok], v[ok], 1)
                    ax.plot(xs, k * xs + b, color=ORANGE, linestyle="--", linewidth=1.4, label="Linear trend")
            if plan.get("annotate_extremes") and len(plan["series"]) == 1:
                v = np.array([_nan(x) for x in plan["series"][0]["values"]], dtype=float)
                for idx, word in ((int(np.nanargmax(v)), "Peak"), (int(np.nanargmin(v)), "Low")):
                    ax.annotate(f"{word}: {_fmt_num(v[idx], 1)}", (xs[idx], v[idx]),
                                textcoords="offset points", xytext=(0, 12 if word == "Peak" else -18),
                                ha="center", fontsize=8, color=text,
                                arrowprops=dict(arrowstyle="-", color=text, alpha=0.4))
            for m in plan.get("markers", []) or []:
                i = m["i"]
                v = plan["series"][0]["values"][i]
                color = GREEN if m["kind"] == "high" else RED
                ax.scatter([xs[i]], [v], s=110, color=color, zorder=5, edgecolor=DARK)
                fmt = plan["series"][0].get("fmt", "num")
                txt = f"{labels[i]}\n" + (f"{v:.1f}%" if fmt == "pct" else _fmt_num(v, 1))
                ax.annotate(txt, (xs[i], v), textcoords="offset points",
                            xytext=(0, 14 if m["kind"] == "high" else -28), ha="center", fontsize=8,
                            color=text, fontweight="bold")
            if plan.get("split") is not None:
                sp = plan["split"]
                ax.axvline(sp - 0.5, color=RED, linestyle="--", linewidth=1.6)
                ax.hlines(plan["before_mean"], -0.4, sp - 0.6, colors=CYAN, linewidth=2.6, label=f"Avg before: {_fmt_num(plan['before_mean'], 1)}")
                ax.hlines(plan["after_mean"], sp - 0.4, n - 0.6, colors=GREEN, linewidth=2.6, label=f"Avg after: {_fmt_num(plan['after_mean'], 1)}")
                ax.axvspan(sp - 0.5, n - 0.5, color=YELLOW, alpha=0.10)
            ax.yaxis.set_major_formatter(_axis_formatter(plan.get("y_fmt", "num")))
            if plan.get("y_fmt") == "index":
                ax.axhline(100, color=GREY, linewidth=1, linestyle=":")

        elif kind in ("bars",):
            s = plan["series"][0]
            color = series_color or YELLOW
            edge = DARK if color == YELLOW else None
            bars = ax.bar(xs, [_nan(v) for v in s["values"]], color=color, width=0.65,
                          edgecolor=edge, linewidth=1.1 if edge else 0)
            for b, v in zip(bars, s["values"]):
                if v is not None:
                    ax.annotate(_fmt_num(v, 1), (b.get_x() + b.get_width() / 2, b.get_height()),
                                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=text)
            ax.yaxis.set_major_formatter(_axis_formatter("num"))

        elif kind == "multi_bars":
            k = len(plan["series"])
            width = 0.8 / k
            for j, s_ in enumerate(plan["series"]):
                offs = (j - (k - 1) / 2) * width
                ax.bar(xs + offs, [_nan(v) for v in s_["values"]], width=width,
                       color=s_.get("color") or SERIES_COLORS[j], label=s_["label"])
            ax.yaxis.set_major_formatter(_axis_formatter("num"))

        elif kind == "growth_bars":
            for i in plan.get("flagged", []):
                ax.axvspan(i - 0.5, i + 0.5, color="#FECACA", alpha=0.55, zorder=0)
            k = len(plan["series"])
            width = 0.8 / k
            for j, s in enumerate(plan["series"]):
                offs = (j - (k - 1) / 2) * width
                ax.bar(xs + offs, [_nan(v) for v in s["values"]], width=width,
                       color=s.get("color") or SERIES_COLORS[j], label=s["label"])
            ax.axhline(0, color=text, linewidth=0.8, alpha=0.6)
            ax.yaxis.set_major_formatter(_axis_formatter("pct"))
            if plan.get("flagged"):
                extra_handles.append(Patch(facecolor="#FECACA", label="Costs grew faster than revenue"))

        elif kind == "change_bars":
            vals = [_nan(v) for v in plan["series"][0]["values"]]
            top = set(plan.get("highlight", []))
            colors = [YELLOW if i in top else (DARK if (not pd.isna(v) and v >= 0) else GREY) for i, v in enumerate(vals)]
            bars = ax.bar(xs, vals, color=colors, edgecolor=DARK, linewidth=0.6, width=0.7)
            for i in top:
                ax.annotate(f"{vals[i]:+.1f}%", (xs[i], vals[i]), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=9, fontweight="bold", color=text)
            ax.axhline(0, color=text, linewidth=0.8, alpha=0.6)
            ax.yaxis.set_major_formatter(_axis_formatter("pct"))
            extra_handles.append(Patch(facecolor=YELLOW, edgecolor=DARK, label="Biggest improvements"))
            extra_handles.append(Patch(facecolor=DARK, edgecolor=DARK, label="Other increases"))
            extra_handles.append(Patch(facecolor=GREY, edgecolor=DARK, label="Declines"))

        elif kind == "combo":
            b, l = plan["series"][0], plan["series"][1]
            ax.bar(xs, [_nan(v) for v in b["values"]], color=series_color or YELLOW, width=0.65,
                   edgecolor=DARK, linewidth=1.0, label=b["label"])
            ax.yaxis.set_major_formatter(_axis_formatter("num"))
            ax2 = ax.twinx()
            ax2.plot(xs, [_nan(v) for v in l["values"]], color=DARK, linewidth=2.8, marker="o",
                     markerfacecolor=YELLOW, markeredgecolor=DARK, markeredgewidth=0.7, label=l["label"])
            ax2.yaxis.set_major_formatter(_axis_formatter("pct"))
            ax2.set_ylabel(plan.get("y2label", ""), color=text)
            ax2.tick_params(colors=text, labelsize=9)
            for side in ("top",):
                ax2.spines[side].set_visible(False)
            ax2.grid(False)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper left")

        elif kind == "scatter":
            x, y = np.array(plan["x"]), np.array(plan["y"])
            ax.scatter(x, y, s=46, color=YELLOW, edgecolor=DARK, linewidth=0.8, alpha=0.95, label="Data points")
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, plan["slope"] * xx + plan["intercept"], color=ORANGE, linewidth=2, linestyle="--",
                    label=f"Best fit (r = {plan['r']:.2f})")
            ax.xaxis.set_major_formatter(_axis_formatter("num"))
            ax.yaxis.set_major_formatter(_axis_formatter("num"))
            ax.grid(True, axis="x", color=text, alpha=0.12)

        if kind in ("line", "bars", "multi_bars", "growth_bars", "change_bars", "combo"):
            _thin_ticks(ax, labels, 16)
            ax.margins(y=0.14)
        if kind == "combo":
            lo, hi = ax.get_ylim()
            ax.set_ylim(0, hi * 1.08)

        ax.set_title(title, fontsize=13, fontweight="bold", color=text, loc="left", pad=12)
        ax.set_xlabel(xlabel if xlabel is not None else plan.get("xlabel", ""), fontsize=10)
        ax.set_ylabel(ylabel if ylabel is not None else plan.get("ylabel", ""), fontsize=10)
        if kind != "combo":
            handles, lab = ax.get_legend_handles_labels()
            handles = handles + extra_handles
            if handles and (len(handles) > 1 or plan.get("split") is not None):
                leg = ax.legend(handles=handles, fontsize=9, frameon=False, loc="best")
                for t in leg.get_texts():
                    t.set_color(text)
        fig.tight_layout()
        return _to_png(fig, dpi, background)
    finally:
        plt.close("all")


def _to_png(fig, dpi, facecolor):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=facecolor)
    buf.seek(0)
    return buf.getvalue()


def render_plan_base64(plan, **kwargs) -> str:
    png = render_plan_png(plan, **kwargs)
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
