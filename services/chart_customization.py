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

# --- Chart type mentioned directly in the message -------------------------
# Lets a user type "give me a pie chart of Region" instead of having to use
# the chart-type dropdown. Order matters: 'barh' phrases are checked before
# 'bar' phrases since "horizontal bar chart" also contains "bar chart".
_CHART_TYPE_PHRASES = (
    ('barh', ("horizontal bar", "horizontal bars", "horizontal bar chart")),
    ('bar', ("bar chart", "bar graph", "bar plot", "bars")),
    ('line', ("line chart", "line graph", "line plot")),
    ('area', ("area chart", "area graph", "area plot")),
    ('pie', ("pie chart", "pie graph", "pie plot")),
)

# --- Colors / theme mentioned directly in the message ----------------------
# EY's brand palette (black, yellow, grey, white) is the primary theme for
# this app — mentioning any one of its colors, or the theme by name, maps
# to the full 4-color 'ey' palette (see services/chart_service.py). Blue,
# red, and green are additional single-hue alternatives for a plain
# colored chart instead of the brand theme.
_THEME_PALETTE_PHRASES = (
    "ey theme", "ey colors", "ey colours", "ey brand", "ey palette",
    "brand colors", "brand colours", "brand theme", "brand palette",
    "corporate colors", "corporate colours", "corporate theme",
    "black yellow grey white", "black, yellow, grey, white",
    "black and yellow", "black yellow and grey",
)
_SINGLE_COLOR_TO_PALETTE = {
    "blue": "blue",
    "red": "red",
    "green": "green",
    "black": "ey",
    "yellow": "ey",
    "grey": "ey",
    "gray": "ey",
    "white": "ey",
}


def _parse_chart_type(q: str) -> Optional[str]:
    for chart_type, phrases in _CHART_TYPE_PHRASES:
        if any(p in q for p in phrases):
            return chart_type
    return None


def _parse_palette(q: str) -> Optional[str]:
    if any(phrase in q for phrase in _THEME_PALETTE_PHRASES):
        return "ey"
    for word, palette in _SINGLE_COLOR_TO_PALETTE.items():
        if re.search(rf"\b{word}\b", q):
            return palette
    return None


@dataclass
class ChartCustomization:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    date_range: Optional[tuple] = None  # (start_month:int, end_month:int)
    show_data_labels: Optional[bool] = None
    chart_type: Optional[str] = None  # 'bar' | 'barh' | 'line' | 'area' | 'pie'
    palette: Optional[str] = None  # 'ey' | 'blue' | 'red' | 'green' | ...
    show_variance: Optional[bool] = None  # ±1 SD band (line/area only)
    show_trend: Optional[bool] = None  # linear trend line (line/area only)
    noise_variance: Optional[float] = None  # injected +/-N% spec-tolerance noise


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


def _parse_show_variance(q: str) -> Optional[bool]:
    """'show variance', 'with the standard deviation band', '±1 sd' -> True."""
    if any(p in q for p in (
        "show variance", "with variance", "variance band",
        "standard deviation", "std dev", "±1 sd", "show the spread",
    )):
        return True
    if any(p in q for p in ("without variance", "no variance", "hide variance")):
        return False
    return None


def _parse_show_trend(q: str) -> Optional[bool]:
    """'show trend', 'add a trend line', 'trendline' -> True."""
    if any(p in q for p in ("show trend", "with trend", "trend line", "trendline", "add a trend")):
        return True
    if any(p in q for p in ("without trend", "no trend", "hide trend")):
        return False
    return None


def _parse_noise_variance(q: str) -> Optional[float]:
    """
    'add 5% noise variance', 'noise variance of 10%', 'apply ±3% variance',
    'with noise' -> fractional variance (e.g. 0.05). This injects jitter
    into the plotted values themselves (see chart_service.render_png_bytes),
    unlike show_variance which only bands the real, unmodified data —
    useful for the "+/-N% spec/tolerance" style requirement common in
    company financial/engineering report templates.
    """
    m = re.search(r"(?:noise\s+variance|variance\s+noise)\s*(?:of|:)?\s*\D*(\d+(?:\.\d+)?)\s*%", q)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*%\s*(?:noise|variance)", q)
    if m:
        return float(m.group(1)) / 100.0
    if re.search(r"\bnoise\s+variance\b|\badd\s+noise\b|\bwith\s+noise\b|\bapply\s+noise\b", q):
        return 0.05  # sensible default when no explicit percentage is given
    if re.search(r"\bno\s+noise\b|\bwithout\s+noise\b|\bremove\s+noise\b", q):
        return 0.0
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
        show_variance=_parse_show_variance(q),
        show_trend=_parse_show_trend(q),
        noise_variance=_parse_noise_variance(q),
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