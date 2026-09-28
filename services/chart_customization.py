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
    ('histogram', ("histogram", "frequency distribution", "distribution chart", "distribution graph")),
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


# --- An explicit list of several distinct colors ("colors should be
# red green blue orange purple") -------------------------------------
# This is a THIRD, distinct color instruction from the two above: not a
# themed multi-color palette guess (_parse_palette), and not a single
# color for one element (_parse_element_color) — the user is naming an
# exact, ordered set of colors to cycle across the bars/slices/lines.
def _parse_color_list(q: str) -> Optional[list]:
    """
    'colors should be red green blue orange purple etc',
    'colours: red, green, blue', 'use red, green and blue for the
    slices' -> ['#DC2626', '#16A34A', '#2563EB', ...] in the order
    given. Returns None when fewer than 2 distinct colors are named,
    so a plain single-color mention ("colors should be red") still
    falls through to the existing single-hue palette logic instead.
    """
    m = re.search(
        r"\bcolou?rs\b\s*(?:should\s+be|to\s+be|:|=|are)?\s*"
        r"([a-z, ]+?)(?:\betc\b|\.|$)",
        q,
    )
    if not m:
        return None
    words = re.findall(rf"\b({_COLOR_WORDS})\b", m.group(1))
    ordered_unique = []
    for w in words:
        if w not in ordered_unique:
            ordered_unique.append(w)
    if len(ordered_unique) < 2:
        return None
    return [COLOR_NAME_TO_HEX[w] for w in ordered_unique]


_OF_TWO_COLUMNS_RE = re.compile(
    r"\b(?:chart|graph|plot|diagram|visuali[sz]e|visuali[sz]ation)\b\s*(?:of\s+)?"
    r"([a-z0-9 _%$/]+?)\s+(?:and|vs\.?|versus)\s+"
    r"([a-z0-9 _%$/]+?)"
    r"(?:\s+and\b|\s+with\b|\s+on\b|\s+background\b|\s+colou?rs?\b|\s+title\b|,|\.|$)"
)


def _parse_of_two_columns(q: str) -> Optional[tuple]:
    """
    'create a line graph of amount and product', 'bar chart of region
    and sales' -> ('amount', 'product') in the order named, so the
    caller can feed them into the SAME strong, deterministic
    "explicit x/y column" pipeline that "x axis amount, y axis
    product" already uses.

    Without this, a message that names its two columns via "chart of A
    and B" instead of "x axis A, y axis B" fell through to a much
    weaker heuristic branch that (a) defaults to counting rows instead
    of aggregating the named numeric column, and (b) still honors the
    chart-type override in theory but was masking the real underlying
    bug — the wrong DATA (row counts instead of amount) being plotted,
    which is what actually produced a bar-shaped chart of near-equal
    bar heights regardless of chart_type.
    """
    m = _OF_TWO_COLUMNS_RE.search(q)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if a and b:
        return a, b
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
    color_list: Optional[list] = None  # ordered list of hex colors, e.g. "colors should be red green blue"
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
        r"(?:\s+and\b|\s+on\b|\s+while\b|\s+colou?rs?\b|\s+background\b|,|\.|$)",
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
        try:
            from services.generic_chart_planner import mentions_fiscal_year
            fiscal = mentions_fiscal_year(q)
        except Exception:  # noqa: BLE001
            fiscal = False
        if fiscal:
            # Financial-year wording: Q1 = Apr-Jun, Q2 = Jul-Sep, Q3 = Oct-Dec,
            # Q4 = Jan-Mar (financial year is always April -> March).
            start = (3 + (quarter - 1) * 3) % 12 + 1
            return (start, start + 2)
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
    # Only look for a generic multi-color LIST when neither element
    # color matched — "background color purple and line color orange"
    # already fully explains itself via the two element-specific colors
    # above, and _parse_color_list's looser "color ... <words> ..."
    # matching would otherwise also swallow "purple" and "orange" out of
    # that same sentence and misread it as a 2-color category list too.
    color_list = None if (background_color or series_color) else _parse_color_list(q)
    # An explicit ordered color list ("colors should be red green blue")
    # is the most specific color signal of the three — it beats both the
    # whole-theme palette guess AND the single-hue guess baked into
    # _parse_palette() (which would otherwise see the word "red" in that
    # same phrase and collapse it down to a single-hue "Reds" colormap).
    palette = None if (background_color or series_color or color_list) else _parse_palette(q)

    x_axis_field = _parse_axis_field(q, 'x')
    y_axis_field = _parse_axis_field(q, 'y')
    if not x_axis_field and not y_axis_field:
        # Fall back to "chart/graph of A and B" phrasing when the user
        # didn't use the more explicit "x axis A, y axis B" wording —
        # both name the same two columns, so both should get routed
        # into the same strong, deterministic aggregation branch.
        pair = _parse_of_two_columns(q)
        if pair:
            x_axis_field, y_axis_field = pair

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
        color_list=color_list,
        x_axis_field=x_axis_field,
        y_axis_field=y_axis_field,
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
            'noise_variance', 'background_color', 'series_color', 'color_list',
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


# ---------------------------------------------------------------------------
# Multi-metric comparison detection ("compare EBITDA and Net Profit",
# "relationship between marketing spending and revenue")
# ---------------------------------------------------------------------------
# The chart pipeline elsewhere in this app only ever resolves ONE numeric
# "value" column per chart (see routes/chatbot.py's value_col detection).
# That silently drops every metric after the first when a question names
# two or more — "compare EBITDA and Net Profit" was rendering an EBITDA-
# only bar chart with Net Profit quietly discarded, and "relationship
# between marketing spending and revenue" often didn't even resolve one
# usable column, so no chart was generated at all. detect_comparison_columns
# + parse_comparison_colors below let the caller build a real multi-series
# chart (grouped bars / multiple lines) instead.

_COMPARISON_CUE_WORDS = (
    'compare', 'comparing', 'comparison', 'versus', ' vs ', ' vs. ',
    'relationship between', 'correlation between', 'against',
)


def detect_comparison_columns(df: pd.DataFrame, message: str, max_columns: int = 4) -> list:
    """
    Find two or more numeric/currency columns that are explicitly named
    in the message, ordered by where they appear in the text (so
    "compare EBITDA and Net Profit" keeps EBITDA first, Net Profit
    second — the order the user asked for and presumably wants in the
    legend/series order).

    Returns [] unless at least 2 real columns are matched by name AND
    the message reads like a genuine comparison request (contains a
    comparison cue word). This second condition avoids over-triggering
    on a message that merely mentions two unrelated numeric columns in
    passing without asking to compare them.
    """
    if not message or df is None or len(df.columns) == 0:
        return []

    from services.schema_service import classify_dataframe

    q = message.lower()
    if not any(cue in q for cue in _COMPARISON_CUE_WORDS):
        return []

    schema = classify_dataframe(df)
    numeric_types = {'number', 'currency', 'stock'}
    candidates = [c['name'] for c in schema if c['type'] in numeric_types]

    found = []
    for col in candidates:
        col_l = str(col).strip().lower()
        if not col_l:
            continue
        idx = q.find(col_l)
        if idx != -1:
            found.append((idx, col))

    found.sort(key=lambda pair: pair[0])
    names = [name for _, name in found][:max_columns]
    return names if len(names) >= 2 else []


def parse_comparison_colors(message: str, columns: list) -> dict:
    """
    Map each comparison column to an explicit color the user tied to it
    by name, e.g. "Use orange for EBITDA and green for Net Profit" ->
    {"EBITDA": "#F97316", "Net Profit": "#16A34A"}. A column with no
    explicit color mentioned is simply absent from the returned dict —
    the caller falls back to a default palette color for it.
    """
    if not message or not columns:
        return {}
    q = message.lower()
    mapping = {}
    for col in columns:
        col_l = re.escape(str(col).strip().lower())
        # "<color> for <column>" / "<color> is for the <column>"
        m = re.search(rf"\b({_COLOR_WORDS})\s+(?:is\s+)?for\s+(?:the\s+)?{col_l}\b", q)
        if not m:
            # "<column> in/as/colored <color>"
            m = re.search(rf"\b{col_l}\b\s+(?:in|as|colou?red)\s+({_COLOR_WORDS})\b", q)
        if m:
            mapping[col] = COLOR_NAME_TO_HEX[m.group(1)]
    return mapping


def wants_category_on_y_axis(message: str, category_col: str) -> bool:
    """
    True when the user explicitly asked for the category/group column
    (e.g. "Year") to sit on the Y-axis — i.e. they want a horizontal bar
    chart, category on Y and the numeric measure on X. Handles the
    common "<column> on the Y-axis" word order that _parse_axis_field()
    above does not (that one only catches "Y axis should be <column>",
    field named AFTER the word "axis"; this is field named BEFORE it).
    """
    if not message or not category_col:
        return False
    q = message.lower()
    col_l = re.escape(str(category_col).strip().lower())
    return re.search(rf"\b{col_l}\b[^.]{{0,20}}\by[\s-]?axis\b", q) is not None
