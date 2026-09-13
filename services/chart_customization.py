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

# --- Explicit per-element color specs ("background black", "line in
# yellow", "bars in red") ----------------------------------------------
# These are a DIFFERENT, more specific instruction than the whole-theme
# palette above: the user is naming an exact color for one specific part
# of the chart, not picking a themed multi-color set. When present, these
# take priority over _parse_palette() for that element — e.g. "keep the
# background black and the line yellow" must not collapse into just
# "use the ey theme", it must set an actual black background AND an
# actual yellow line explicitly.
COLOR_NAME_TO_HEX = {
    "black": "#000000", "white": "#FFFFFF",
    "yellow": "#FFE600", "grey": "#9DA0A6", "gray": "#9DA0A6",
    "blue": "#2563EB", "red": "#DC2626", "green": "#16A34A",
    "orange": "#F97316", "purple": "#7C3AED", "pink": "#EC4899",
    "cyan": "#06B6D4", "navy": "#1E3A8A", "maroon": "#7F1D1D",
    "gold": "#EAB308", "teal": "#0D9488", "brown": "#78350F",
    "silver": "#C0C0C0",
}
_COLOR_WORDS = "|".join(COLOR_NAME_TO_HEX.keys())

# Element keyword -> regex fragments matching the noun this color word
# describes, e.g. "background", "line", "bars"/"bar", "area", "fill".
_ELEMENT_KEYWORDS = {
    # "backgound" is a common typo (missing 'r') — accept it alongside
    # the correct spelling so it isn't silently dropped.
    "background": ("background", "backgound", "backdrop"),
    "series": ("line", "bars?", "area", "fill", "graph line", "chart line", "wedges?", "slices?", "points?"),
}


def _parse_element_color(q: str, element: str) -> Optional[str]:
    """
    Find an explicit color word tied to a specific chart element, in
    either word order: "black background" / "background in black" /
    "background of the graph black", "yellow line" / "line in/to/as
    yellow" / "line part in yellow". Returns a hex string or None.
    """
    nouns = "|".join(_ELEMENT_KEYWORDS[element])
    other_nouns = "|".join(
        n for e, ns in _ELEMENT_KEYWORDS.items() if e != element for n in ns
    )
    # "<color> <noun>" e.g. "black background", "yellow line"
    m = re.search(rf"\b({_COLOR_WORDS})\s+(?:{nouns})\b", q)
    if m:
        return COLOR_NAME_TO_HEX[m.group(1)]
    # "<noun> ... <color>", with up to a few filler words in between
    # ("of the graph", "part", "in", "colored") — but the filler must
    # NOT cross into "and"/"but" or the OTHER element's noun, otherwise
    # a single message naming two different element colors (e.g.
    # "background black and the line in yellow") would let this greedily
    # skip straight past "black" and match the far-away "yellow" instead.
    filler = rf"(?:(?!\b(?:and|but|{other_nouns})\b)\w+\s+){{0,5}}"
    m = re.search(rf"\b(?:{nouns})\b\s+{filler}({_COLOR_WORDS})\b", q)
    if m:
        return COLOR_NAME_TO_HEX[m.group(1)]
    return None


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
    background_color: Optional[str] = None  # explicit hex, e.g. "background black"
    series_color: Optional[str] = None  # explicit hex for line/bars/area/pie, e.g. "line in yellow"
    x_axis_field: Optional[str] = None  # raw text, e.g. "amount" from "x axis should be amount"
    y_axis_field: Optional[str] = None  # raw text, e.g. "period" from "y axis should be period"


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
    """
    'title Q3 Sales', 'call it "Sales Overview"', 'title of the graph is
    Amount vs Product', 'title of graph amount vs product' (no connector
    word at all) -> the title text.

    Handles an optional "of the graph/chart" filler right after the
    trigger word — without this, "title of graph amount vs product"
    would greedily capture "of graph amount vs product" as the literal
    title text instead of skipping the filler and taking just
    "amount vs product". The connector word (is/as/to/:/=) is also
    optional for the same reason: "x axis amount" style phrasing with no
    connector at all must still resolve, not just "x axis is amount".
    """
    m = re.search(
        r"(?:title|call it|name it)\s+"
        r"(?:of\s+(?:the\s+)?(?:graph|chart)\s+)?"
        r"(?:should\s+be|should\s+say|is|as|to|:|=)?\s*"
        r"[\"']?([A-Za-z0-9 _%$-]+?)[\"']?"
        r"(?:\s+and\b|\s+on\b|\s+while\b|,|\.|$)",
        q,
    )
    if m:
        text = m.group(1).strip().rstrip(".")
        if text:
            return text
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


def _parse_axis_field(q: str, axis: str) -> Optional[str]:
    """
    Parse an explicit "put THIS column on THIS axis" instruction, e.g.
    "x axis should be amount", "on the y axis it should be period",
    "y-axis: region", "make x axis = revenue". This is a DIFFERENT
    instruction than _parse_axis_rename() above — that only renames the
    text label shown on an axis, this says which real data column should
    actually drive that axis. Returns the raw phrase (resolved against
    real dataframe columns by the caller, which has access to the df).
    """
    # The connector phrase ("should be" / "is" / "as" / ":" / "=" / "to")
    # is OPTIONAL: "on x axis amount, y axis product" names the field
    # right after "axis" with no connector word at all, and must resolve
    # the same as "x axis should be amount" does.
    m = re.search(
        rf"\b{axis}[\s-]?axis\b\s*(?:it\s+)?(?:should\s+be|should\s+show|will\s+be|is|as|:|=|to)?\s+"
        r"([a-z0-9 _%$/-]+?)(?:\s+and\b|\s+on\b|\s+while\b|,|\.|$)",
        q,
    )
    if m:
        field = m.group(1).strip()
        if field:
            return field
    return None


def parse_chart_customization(message: str) -> ChartCustomization:
    """Deterministic — no LLM round-trip needed for this structured a parse."""
    q = (message or "").lower()
    xlabel, ylabel = _parse_axis_rename(q)

    # Explicit per-element colors ("black background", "line in yellow")
    # are a stronger, more specific signal than the whole-theme guess in
    # _parse_palette() — check them first, and suppress the whole-theme
    # palette guess when an explicit element color was found, so "keep
    # the background black and the line yellow" doesn't collapse into
    # "use the ey theme" just because both color words are brand colors.
    background_color = _parse_element_color(q, "background")
    series_color = _parse_element_color(q, "series")
    palette = None if (background_color or series_color) else _parse_palette(q)

    return ChartCustomization(
        title=_parse_title(q),
        xlabel=xlabel,
        ylabel=ylabel,
        date_range=_parse_month_range(q),
        show_data_labels=_parse_data_labels(q),
        chart_type=_parse_chart_type(q),
        palette=palette,
        show_variance=_parse_show_variance(q),
        show_trend=_parse_show_trend(q),
        noise_variance=_parse_noise_variance(q),
        background_color=background_color,
        series_color=series_color,
        x_axis_field=_parse_axis_field(q, 'x'),
        y_axis_field=_parse_axis_field(q, 'y'),
    )


def has_customization(custom: ChartCustomization) -> bool:
    """
    True if the message carried ANY graph customization/specification —
    a chart type, a color/theme, an axis rename or field mapping, a date
    range, data-label/variance/trend toggles, or injected noise.

    This is what lets a plain follow-up like "make the background black
    and title it Q1 Sales" or "show only Jan-Jun with data labels"
    trigger chart (re)generation on its own, even though it contains
    none of the literal CHART_KEYWORDS_DEFAULT words ("chart", "graph",
    "plot", ...). Without this, wants_chart() would miss pure
    customization messages and the user's typed spec would be silently
    dropped instead of producing a graph.
    """
    return any(
        getattr(custom, field) is not None
        for field in (
            'title', 'xlabel', 'ylabel', 'date_range', 'show_data_labels',
            'chart_type', 'palette', 'show_variance', 'show_trend',
            'noise_variance', 'background_color', 'series_color',
            'x_axis_field', 'y_axis_field',
        )
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