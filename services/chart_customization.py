"""
Chart Customization Parser — remaining half
=============================================
PARTIAL item: "Natural-language chart customization for colors, labels,
date ranges, and other preferences."

Your UI (chatbot.html #chart-options) already lets the user pick
chart_type / chart_palette / chart_style explicitly via dropdowns, sent
as form fields to /chat/send. That covers "colors" and part of "labels"
through UI controls, not natural language yet.

What's still missing, and what this module adds, is parsing these
straight out of the typed message:
  - date ranges ("show only Jan-Jun", "just Q1", "from March to August")
  - axis/title renaming ("rename the x-axis to Region", "call it Q3 Sales")
  - data-label visibility ("show values on the bars")

This does NOT duplicate the palette/style dropdowns — it only fills the
gap of phrases the dropdowns can't express. It reuses insight_service's
own month_to_num()/MONTH_NUM_TO_SHORT so month handling stays consistent
with the rest of the app instead of introducing a second month parser.

Drop into services/chart_customization.py.

Integration point in routes/chatbot.py, before building monthly/quarterly
totals:

    from services.chart_customization import parse_chart_customization, apply_month_range_filter

    custom = parse_chart_customization(message)
    if custom.date_range and month_col:
        df = apply_month_range_filter(df, month_col, custom.date_range)
    # then compute monthly_totals(df, month_col, value_col) as usual

    # when calling chart_service.render_png_bytes(...):
    title=custom.title or default_title,
    xlabel=custom.xlabel or default_xlabel,
    ylabel=custom.ylabel or default_ylabel,
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from services.insight_service import MONTH_NAME_TO_NUM, month_to_num

_MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)


@dataclass
class ChartCustomization:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    date_range: Optional[tuple] = None  # (start_month:int, end_month:int)
    show_data_labels: Optional[bool] = None
    chart_type: Optional[str] = None
    palette: Optional[str] = None
    style: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    show_variance: Optional[bool] = None
    show_trend: Optional[bool] = None



_VALID_CHART_TYPES = {"bar", "barh", "line", "area", "pie"}


def _parse_chart_type(q: str) -> Optional[str]:
    rules = [
        ("barh", r"\b(?:horizontal\s+bar|barh)\b"),
        ("bar", r"\b(?:bar\s+chart|bar\s+graph|bar)\b"),
        ("line", r"\b(?:line\s+chart|line\s+graph|line\s+plot)\b"),
        ("area", r"\b(?:area\s+chart|area\s+graph)\b"),
        ("pie", r"\b(?:pie\s+chart|pie\s+graph|donut\s+chart|donut)\b"),
    ]
    for chart_type, pattern in rules:
        if re.search(pattern, q):
            return chart_type
    return None


def _parse_palette(q: str) -> Optional[str]:
    if re.search(r"\b(?:ey|yellow|black|grey|gray|white)\s*(?:theme|palette|colour|color)?\b", q):
        return "ey"
    for name in ("blue", "red", "green", "multi", "multicolor", "colorblind", "viridis", "mako", "rocket", "flare", "crest", "coolwarm", "pastel", "bright", "muted", "dark", "deep"):
        if re.search(rf"\b{name}\b", q):
            return "multi" if name == "multicolor" else name
    return None


def _parse_style(q: str) -> Optional[str]:
    for name in ("darkgrid", "whitegrid", "white", "dark", "ticks"):
        if re.search(rf"\b{name}\b", q):
            return name
    return None


def _parse_column(q: str, axis: str) -> Optional[str]:
    m = re.search(rf"(?:{axis}[- ]?axis|{axis})\s+(?:column|field)?\s*(?:to|as|from|use)\s+[\"']?([^,\"']+?)[\"']?(?:\s+(?:and|with|show|using)\b|$)", q)
    return m.group(1).strip() if m else None


def _parse_toggle(q: str, yes_patterns, no_patterns) -> Optional[bool]:
    if any(re.search(p, q) for p in yes_patterns):
        return True
    if any(re.search(p, q) for p in no_patterns):
        return False
    return None

def _parse_axis_rename(q: str) -> tuple:
    xlabel = ylabel = None
    m = re.search(r"(?:rename|label|set)\s+(?:the\s+)?x[- ]?axis\s+(?:to|as)\s+([A-Za-z0-9 _%$]+)", q)
    if m:
        xlabel = m.group(1).strip().rstrip(".")
    m = re.search(r"(?:rename|label|set)\s+(?:the\s+)?y[- ]?axis\s+(?:to|as)\s+([A-Za-z0-9 _%$]+)", q)
    if m:
        ylabel = m.group(1).strip().rstrip(".")
    return xlabel, ylabel


def _parse_title(q: str) -> Optional[str]:
    m = re.search(r"(?:title|call it|name it)\s+(?:to|as)?\s*[\"']?([A-Za-z0-9 _%$-]+)[\"']?", q)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _parse_month_range(q: str) -> Optional[tuple]:
    """'Jan-Jun', 'from January to June', 'Q1 only' -> (start_month, end_month)."""
    m = re.search(
        rf"\b{_MONTH_WORD}\s*(?:-|to|through|until)\s*{_MONTH_WORD}\b", q
    )
    if m:
        words = re.findall(_MONTH_WORD, m.group(0))
        if len(words) == 2:
            start = MONTH_NAME_TO_NUM.get(words[0].lower())
            end = MONTH_NAME_TO_NUM.get(words[1].lower())
            if start and end:
                return (start, end)

    q_match = re.search(r"\bq([1-4])\b", q)
    if q_match:
        quarter = int(q_match.group(1))
        start = (quarter - 1) * 3 + 1
        return (start, start + 2)

    single = re.search(rf"\bonly\s+{_MONTH_WORD}\b", q)
    if single:
        word = re.search(_MONTH_WORD, single.group(0)).group(0)
        m_num = MONTH_NAME_TO_NUM.get(word.lower())
        if m_num:
            return (m_num, m_num)

    return None


def _parse_data_labels(q: str) -> Optional[bool]:
    if "with data labels" in q or "show values" in q or "add labels" in q or "with labels" in q:
        return True
    if "without data labels" in q or "hide values" in q or "no labels" in q:
        return False
    return None


def parse_chart_customization(message: str) -> ChartCustomization:
    """Deterministic — no LLM round-trip needed for this structured a parse."""
    q = (message or "").lower()
    xlabel, ylabel = _parse_axis_rename(q)
    return ChartCustomization(
        title=_parse_title(q),
        xlabel=xlabel,
        ylabel=ylabel,
        date_range=_parse_month_range(q),
        show_data_labels=_parse_data_labels(q),
        chart_type=_parse_chart_type(q),
        palette=_parse_palette(q),
        style=_parse_style(q),
        x_column=_parse_column(q, "x"),
        y_column=_parse_column(q, "y"),
        show_variance=_parse_toggle(
            q, [r"\bshow\s+(?:variance|variation)\b", r"\badd\s+(?:variance|variation)\b"],
            [r"\bhide\s+(?:variance|variation)\b", r"\bwithout\s+(?:variance|variation)\b"],
        ),
        show_trend=_parse_toggle(
            q, [r"\bshow\s+(?:trend|trend\s+line)\b", r"\badd\s+(?:trend|trend\s+line)\b"],
            [r"\bhide\s+(?:trend|trend\s+line)\b", r"\bwithout\s+(?:trend|trend\s+line)\b"],
        ),
    )


def apply_month_range_filter(df: pd.DataFrame, month_col: str, month_range: Optional[tuple]) -> pd.DataFrame:
    """
    Filter rows to a (start_month, end_month) inclusive range using the
    SAME month parsing insight_service already relies on for monthly_totals
    (month_to_num handles names, numbers, and date-like strings uniformly).
    """
    if not month_range or not month_col or month_col not in df.columns:
        return df
    start_month, end_month = month_range
    months = df[month_col].apply(month_to_num)
    mask = months.apply(lambda m: m is not None and start_month <= m <= end_month)
    return df[mask]
