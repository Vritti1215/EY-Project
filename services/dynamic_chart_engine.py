"""
Dynamic Chart Engine
====================
Answers *any* chart-style question about an uploaded table the way ChatGPT /
Claude would — without the user naming a column or a chart type — and always
with a **comparison** the reader can act on.

Why this exists
---------------
The older branches in routes/chatbot.py were keyword driven. A question such as
"Create a graph comparing the different sales channels." was swallowed by the
"list the distinct values" shortcut (the word *different*) and never reached a
chart. Generic prompts ("Find the strongest relationships in the data",
"Show me where the biggest changes are occurring", "Build a visual overview")
had no rule at all and depended entirely on the language model being reachable.

How it works (deterministic — the AI never supplies a number)
-------------------------------------------------------------
    question ─► intent  (compare / ranking / share / trend-by-group /
                         relationship / biggest changes / unusual / overview)
             ─► REAL columns   (dimension, measures, date) resolved from the
                                question with plural / synonym matching, or
                                chosen automatically from the data profile
             ─► pandas aggregation (sum for flows, weighted ratios recomputed)
             ─► one multi-panel matplotlib figure in the EY palette
             ─► facts + a ready-made markdown summary (used when the language
                model is unavailable)

Public API
----------
    is_visual_request(question)                 -> bool
    plan_dynamic(df, question, legacy_intent)   -> dict | None
    render_panels_png / render_panels_base64    -> PNG bytes / data URI
"""

from __future__ import annotations

import base64
import io
import logging
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from services import generic_chart_planner as G

logger = logging.getLogger(__name__)

DARK, YELLOW, GREY = G.DARK, G.YELLOW, G.GREY
CYAN, ORANGE, GREEN, RED = G.CYAN, G.ORANGE, G.GREEN, G.RED
PALETTE = [DARK, YELLOW, CYAN, ORANGE, GREEN, "#7C3AED", GREY, "#0EA5E9"]

# ---------------------------------------------------------------------------
# 1. Intent vocabulary
# ---------------------------------------------------------------------------
RE_REL = re.compile(
    r"relationship|correlat|closely related|most related|\brelated\b|associat|linked|"
    r"depend|influenc|\bdrivers?\b|\bdrives?\b|affect|impact of .+ on"
)
RE_UNUSUAL = re.compile(r"unusual|anomal|outlier|abnormal|irregular|strange|unexpected|\bspikes?\b")
RE_CHANGE = re.compile(
    r"biggest changes?|largest changes?|major changes?|greatest changes?|most changes?|"
    r"changes? (?:are )?(?:occurring|happening)|where .{0,25}chang|fastest|biggest (?:jump|drop|swing)|"
    r"volatil|swings?"
)
RE_OVERVIEW = re.compile(
    r"overview|summary|summari[sz]e|explore|\bpatterns?\b|insights?|dashboard|snapshot|big picture|"
    r"business performance|explains? .{0,30}performance|understand(?:ing)? (?:this|the) (?:data|dataset)|"
    r"most useful|you think|interesting|everything|whole (?:data|dataset)|overall (?:picture|performance)|"
    r"kpis?|most relevant|relevant visual|key (?:metrics|figures)|generic"
)
RE_CONTRIB = re.compile(
    r"contribut|driving|\bshare\b|breakdown|break down|composition|proportion|\bsplit\b|\bmix\b|"
    r"make[s]? up|accounts? for|pareto|percentage of|% of|distribution of"
)
RE_RANK = re.compile(
    r"\btop\b|\bbest\b|\bworst\b|highest|lowest|\bmost\b|\bleast\b|\brank|leading|under-?perform|"
    r"\bbottom\b|perform(?:s|ing)? (?:best|worst)|biggest|largest|smallest|strongest|weakest"
)
RE_COMPARE = re.compile(
    r"compar|versus|\bvs\.?\b|\bv/s\b|differen|across|against|between|contrast|side by side|benchmark|"
    r"\beach\b|\bper\b|\bby\b|performance of|how do .{0,30} (?:stack up|differ)"
)
RE_TIME = re.compile(
    r"over time|trend|monthly|quarterly|yearly|annual|weekly|daily|timeline|month[- ]by[- ]month|"
    r"year[- ]over[- ]year|progress|growth|history|seasonal|evolv|across (?:the )?(?:months|quarters|years)"
)
RE_AUTO = re.compile(r"(?:automatically|most useful|most relevant|best|interesting|useful).{0,30}compar|compar.{0,30}(?:most useful|automatic)")
RE_VISUAL_VERB = re.compile(
    r"\b(?:show|find|identify|explore|build|create|make|highlight|visuali[sz]e|graph|plot|draw|display|"
    r"illustrate|generate|give me|see)\b"
)
RE_TOPN = re.compile(r"\b(?:top|bottom|best|worst)\s+(\d{1,2})\b")

_TIME_NAMES = {"month", "quarter", "qtr", "year", "week", "day", "period", "fy", "fiscal year",
               "month number", "date", "time"}
_ID_TOKENS = {"id", "code", "zip", "pin", "phone", "number", "no", "index", "serial", "sku", "uuid"}
_RATIO_TOKENS = {"rate", "margin", "ratio", "percent", "pct", "price", "score", "rating", "age", "index",
                 "average", "avg"}
_WEAK_DIM_TOKENS = {"type", "method", "status", "mode", "name", "kind"}
_GENERIC_TOKENS = {"sale", "total", "net", "gross", "name", "the", "of", "by"}

SYNONYMS = {
    "salespeople": "salesperson", "salesman": "salesperson", "reps": "salesperson", "rep": "salesperson",
    "executives": "salesperson", "cities": "city", "area": "region", "areas": "region",
    "territory": "region", "territories": "region", "zone": "region", "zones": "region",
    "item": "product", "items": "product", "goods": "product", "categories": "category",
    "channels": "channel", "medium": "channel", "mode": "method", "modes": "method",
    "clients": "customer", "buyers": "customer", "state": "state", "states": "state",
    "country": "country", "countries": "country", "dept": "department", "departments": "department",
}


def _sing(word: str) -> str:
    if word in SYNONYMS:
        return SYNONYMS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tok(text) -> list:
    return [_sing(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


# ---------------------------------------------------------------------------
# 2. Cheap request classifiers used by the route
# ---------------------------------------------------------------------------
def is_visual_request(question: str) -> bool:
    """True for chart-style questions that never say 'chart' ("Find the strongest
    relationships in the data", "Show me where the biggest changes are occurring")."""
    q = (question or "").lower()
    if not q.strip():
        return False
    if re.search(r"graph|chart|plot|visuali[sz]|diagram|dashboard|infographic|heat ?map|scatter|histogram", q):
        return True
    analytic = (RE_REL.search(q) or RE_CHANGE.search(q) or RE_UNUSUAL.search(q) or RE_OVERVIEW.search(q)
                or re.search(r"contributors?|comparison|most important", q))
    return bool(analytic and RE_VISUAL_VERB.search(q))


# ---------------------------------------------------------------------------
# 3. Column profiling
# ---------------------------------------------------------------------------
def _name_tokens(name) -> set:
    return set(_tok(name))


def _is_idlike(name) -> bool:
    return bool(_name_tokens(name) & _ID_TOKENS)


def _is_timelike(name) -> bool:
    n = re.sub(r"\s+", " ", str(name).strip().lower())
    return n in _TIME_NAMES or n.endswith(" year") or n.endswith("_year") or n.startswith("fy")


def _find_date_col(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        n = str(c).lower()
        if ("date" in n or n in ("period", "day", "timestamp")) and not pd.api.types.is_numeric_dtype(df[c]):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if len(df) and parsed.notna().mean() >= 0.8:
                return c
    return None


def _num(series):
    return G._as_number(series)


class _Ctx:
    def __init__(self, df: pd.DataFrame, question: str):
        self.df = df
        self.q = (question or "").lower()
        self.qtok = set(_tok(re.sub(r"(?:chart|graph|plot|visual\w*)\s+types?|types?\s+of\s+(?:chart|graph)", " ", (question or "").lower())))
        self.date_col = _find_date_col(df)
        numeric = set(G._numeric_columns(df))
        self.measures = [
            c for c in df.columns
            if c in numeric and c != self.date_col and not _is_idlike(c) and not _is_timelike(c)
            and _num(df[c]).nunique(dropna=True) > 2
        ]
        self.dims = [c for c in df.columns if self._is_dim(c, numeric)]
        self.time_dims = [
            c for c in df.columns
            if _is_timelike(c) and c != self.date_col and 2 <= df[c].nunique(dropna=True) <= 60
            and (c not in numeric or str(c).strip().lower() in ("year", "fiscal year", "fy"))
        ]
        self.primary = self._primary_measure()

    def _is_dim(self, c, numeric) -> bool:
        if c in numeric or c == self.date_col or pd.api.types.is_datetime64_any_dtype(self.df[c]):
            return False
        if _is_idlike(c) or _is_timelike(c):
            return False
        n = self.df[c].nunique(dropna=True)
        cap = 0.5 if len(self.df) >= 20 else 0.9
        return 2 <= n <= 40 and n / max(len(self.df), 1) <= cap

    # ---- measures ---------------------------------------------------------
    def kind_of(self, col) -> str:
        return "mean" if (_name_tokens(col) & _RATIO_TOKENS) else "sum"

    def _primary_measure(self):
        prefs = ("sales", "total sales", "revenue", "net revenue", "gross sales", "turnover", "income",
                 "amount", "total amount", "value", "revenue amount")
        norm = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in self.measures}
        for p in prefs:
            if p in norm:
                return norm[p]
        for c in self.measures:
            n = str(c).lower()
            if any(k in n for k in ("sales", "revenue", "amount", "income", "turnover")) and self.kind_of(c) == "sum":
                return c
        additive = [c for c in self.measures if self.kind_of(c) == "sum"]
        return additive[0] if additive else (self.measures[0] if self.measures else None)

    def find_measure(self, *keywords):
        for kw in keywords:
            for c in self.measures:
                if kw in str(c).lower():
                    return c
        return None

    def named_measures(self, exclude=frozenset()) -> list:
        """Measures the user actually named (longest phrase wins), in question order."""
        hits = []
        for c in self.measures:
            toks = [t for t in _tok(c)]
            core = [t for t in toks if t not in _GENERIC_TOKENS] or toks
            if not core or set(toks) <= set(exclude):
                continue
            if all(t in self.qtok for t in toks) or (
                all(t in self.qtok for t in core) and len(core) == len(toks)
            ):
                pos = min((self.q.find(w) for w in re.findall(r"[a-z0-9]+", str(c).lower()) if self.q.find(w) >= 0), default=999)
                hits.append((c, len(toks), pos))
        # drop measures whose words are fully contained in a longer matched measure
        keep = []
        for c, ln, pos in hits:
            if any(o != c and ln < oln and set(_tok(c)) <= set(_tok(o)) for o, oln, _ in hits):
                continue
            keep.append((c, pos))
        keep.sort(key=lambda t: t[1])
        out = [c for c, _ in keep]
        # concept words
        concept = {
            "profit": ("profit",), "profitability": ("profit margin", "profit"), "earning": ("profit",),
            "cost": ("cost",), "expense": ("cost",), "spend": ("cost",),
            "quantity": ("quantity",), "volume": ("quantity",), "unit": ("quantity",),
            "discount": ("discount amount", "discount"), "tax": ("tax",),
        }
        for word, kws in concept.items():
            if word in self.qtok or word + "s" in self.q:
                col = self.find_measure(*kws)
                if col and col not in out:
                    out.append(col)
        if self.primary and re.search(r"revenue|income|turnover|business|money|\bsales\b", self.q) and not out \
                and not (set(_tok(self.primary)) <= set(exclude)):
            out.append(self.primary)
        return out

    # ---- dimensions -------------------------------------------------------
    def resolve_dim(self):
        """Return (dim_column | None, members_filter | None)."""
        best, best_score = None, 0.0
        for c in self.dims + self.time_dims:
            toks = _tok(c)
            if not toks:
                continue
            distinct = [t for t in toks if t not in _GENERIC_TOKENS]
            matched = [t for t in toks if t in self.qtok]
            matched_distinct = [t for t in distinct if t in self.qtok]
            if not matched_distinct or all(t in _WEAK_DIM_TOKENS for t in matched_distinct):
                continue
            score = sum(1.0 if t not in _GENERIC_TOKENS else 0.4 for t in matched) / len(toks)
            score += 0.05 * len(matched_distinct)
            if c in self.time_dims:
                score *= 0.9
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            return best, self._mentioned_members(best)
        # a value named directly ("Online vs Retail Store")
        for c in self.dims:
            members = self._mentioned_members(c)
            if members and len(members) >= 2:
                return c, members
        return None, None

    def _mentioned_members(self, col):
        vals = self.df[col].dropna().astype(str).str.strip().unique().tolist()
        found = [v for v in vals if len(v) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(?![a-z0-9])", self.q)]
        return found if len(found) >= 2 else None


# ---------------------------------------------------------------------------
# 4. Aggregation helpers
# ---------------------------------------------------------------------------
def _clean_dim(df, col) -> pd.Series:
    raw = df[col].astype(str).str.strip()
    key = raw.str.lower().str.replace(r"\s+", " ", regex=True)
    display = raw.groupby(key).agg(lambda v: v.value_counts().index[0])
    out = key.map(display)
    return out.where(~df[col].isna() & out.ne("") & out.str.lower().ne("nan"))


def _member_order(ctx: _Ctx, dim, index_like, values_for_sort=None):
    """Chronological for month/quarter/year style columns, otherwise by value desc."""
    labels = list(index_like)
    if dim in ctx.time_dims:
        if ctx.date_col is not None:
            d = pd.to_datetime(ctx.df[ctx.date_col], errors="coerce")
            first = d.groupby(_clean_dim(ctx.df, dim)).min()
            return sorted(labels, key=lambda l: (pd.isna(first.get(l)), first.get(l)))
        try:
            return sorted(labels, key=lambda l: float(re.sub(r"[^0-9.]", "", str(l)) or 0))
        except ValueError:
            return sorted(labels)
    if values_for_sort is not None:
        return list(values_for_sort.sort_values(ascending=False).index)
    return labels


def _group_table(ctx: _Ctx, dim, cols, df=None) -> pd.DataFrame:
    df = ctx.df if df is None else df
    work = pd.DataFrame({"_m": _clean_dim(df, dim)})
    for c in cols:
        work[c] = _num(df[c])
    work = work.dropna(subset=["_m"])
    aggs = {c: ctx.kind_of(c) for c in cols}
    table = work.groupby("_m").agg(aggs)
    table["_rows"] = work.groupby("_m").size()
    return table


def _margin_pct(profit, base):
    return float(profit) / float(base) * 100 if base not in (0, None) and not pd.isna(base) else float("nan")


def _period_series(ctx: _Ctx, cols, by_dim=None, df=None):
    """Aggregate to a chronological period table. Returns (labels, {name: values})."""
    df = ctx.df if df is None else df
    d = pd.to_datetime(df[ctx.date_col], errors="coerce")
    ok = d.notna()
    span_months = (d[ok].max().year - d[ok].min().year) * 12 + d[ok].max().month - d[ok].min().month + 1
    quarterly = span_months > 36
    freq = "Q" if quarterly else "M"
    per = d[ok].dt.to_period(freq)
    work = pd.DataFrame({"_p": per.values}, index=df.index[ok])
    for c in cols:
        work[c] = _num(df.loc[ok, c])
    if by_dim is not None:
        work["_m"] = _clean_dim(df.loc[ok], by_dim).values
        work = work.dropna(subset=["_m"])
    full = pd.period_range(work["_p"].min(), work["_p"].max(), freq=freq)

    def label(p):
        if quarterly:  # fiscal quarter, April start (matches the rest of the app)
            fy = p.year + (1 if p.quarter >= 2 else 0)
            fq = ((p.quarter - 2) % 4) + 1
            return f"Q{fq} FY{str(fy)[-2:]}"
        return p.strftime("%b %Y")

    labels = [label(p) for p in full]
    out = {}
    if by_dim is None:
        for c in cols:
            s = work.groupby("_p")[c].agg(ctx.kind_of(c)).reindex(full)
            out[c] = (s.fillna(0) if ctx.kind_of(c) == "sum" else s).tolist()
    else:
        for c in cols[:1]:
            for m, g in work.groupby("_m"):
                s = g.groupby("_p")[c].agg(ctx.kind_of(c)).reindex(full)
                out[m] = (s.fillna(0) if ctx.kind_of(c) == "sum" else s).tolist()
    return labels, out, ("quarter" if quarterly else "month")


def _fmt(v, d=1):
    return G._fmt_num(v, d)


def _pct(v, d=1):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{d}f}%"


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _base(intent, title, panels, layout, facts, summary, assumptions, reason="", is_time=False, figsize=None):
    return {
        "intent": intent, "kind": "panels", "title": title, "labels": [], "series": [],
        "xlabel": "", "ylabel": "", "y_fmt": "num", "panels": panels, "layout": layout,
        "figsize": figsize, "facts": facts, "summary": summary, "assumptions": assumptions,
        "reason": reason, "legacy_compatible": False, "is_time_series": is_time,
    }


def _facts_block(title, assumptions, lines):
    head = ["AUTHORITATIVE DETERMINISTIC RESULT — DO NOT RECALCULATE:", f"Chart generated by the server: {title}"]
    head += [f"Assumption: {a}" for a in assumptions]
    return "\n".join(head + lines)


# ---------------------------------------------------------------------------
# 5. Planners
# ---------------------------------------------------------------------------
def _limit_members(table: pd.DataFrame, primary, limit=12):
    """Keep the top `limit` members, folding the rest into 'Other'."""
    t = table.sort_values(primary, ascending=False)
    if len(t) <= limit:
        return t, None
    top, rest = t.iloc[: limit - 1], t.iloc[limit - 1:]
    other = rest.sum(numeric_only=True)
    top = pd.concat([top, pd.DataFrame([other], index=[f"Other ({len(rest)})"])])
    return top, len(rest)


def _plan_dim_compare(ctx: _Ctx, dim, members, measures, auto_note=None):
    df = ctx.df
    if members:
        keep = _clean_dim(df, dim).str.lower().isin([m.lower() for m in members])
        df = df[keep]
    prim = ctx.primary
    named = [m for m in measures if m in ctx.measures]
    cols = named[:3] if named else [c for c in (prim, ctx.find_measure("profit")) if c]
    cols = list(dict.fromkeys(cols))
    if not cols:
        return None
    profit = ctx.find_measure("profit")
    extra = [c for c in (profit,) if c and c not in cols and ctx.kind_of(c) == "sum"]
    table = _group_table(ctx, dim, list(dict.fromkeys(cols + extra + ([prim] if prim else []))), df)
    if len(table) < 2:
        return None
    lead = cols[0]
    is_time_dim = dim in ctx.time_dims
    if is_time_dim:
        order = _member_order(ctx, dim, table.index)
        table = table.loc[order]
        shown, folded = table, None
    else:
        shown, folded = _limit_members(table, lead)
    labels = [str(i) for i in shown.index]

    # grouped bars: measures on wildly different scales get their own axis via indexing note
    mags = [abs(float(shown[c].mean())) or 1.0 for c in cols]
    assumptions = []
    if auto_note:
        assumptions.append(auto_note)
    if not named:
        assumptions.append(f"No measure was named, so '{lead}'"
                           + (f" and '{cols[1]}'" if len(cols) > 1 else "") + " were compared.")
    if folded:
        assumptions.append(f"Showing the top {len(labels) - 1} {dim} values; the other {folded} are grouped as 'Other'.")
    if is_time_dim and lead:
        assumptions.append(f"'{dim}' is a time column, so members are shown in chronological order.")
    if len(cols) > 1 and max(mags) / max(min(mags), 1e-9) > 8:
        assumptions.append(f"'{cols[1]}' is on a very different scale from '{lead}'; only '{lead}' is drawn in the main chart.")
        cols = cols[:1]

    series = [{"label": str(c), "values": [float(v) for v in shown[c].tolist()],
               "color": [DARK, YELLOW, GREY][i]} for i, c in enumerate(cols)]
    panels = [{"type": "bars", "title": f"{' vs '.join(map(str, cols))} by {dim}", "labels": labels,
               "series": series, "pos": (0, 1, 0, 2), "annotate": len(labels) <= 8 and len(series) <= 2}]

    # share of the lead measure
    lead_vals = shown[lead].clip(lower=0)
    total = float(lead_vals.sum())
    shares = [(float(v) / total * 100 if total else float("nan")) for v in lead_vals]
    if len(labels) <= 6:
        panels.append({"type": "donut", "title": f"Share of {lead}", "labels": labels,
                       "values": [float(v) for v in lead_vals.tolist()], "pos": (1, 2, 0, 1)})
    else:
        panels.append({"type": "hbars", "title": f"Share of {lead} (%)", "labels": labels,
                       "values": shares, "pos": (1, 2, 0, 1), "fmt": "pct"})

    # margin (Profit / lead) or order count
    margins = None
    if profit and profit != lead and ctx.kind_of(lead) == "sum" and lead in shown and profit in shown:
        margins = [_margin_pct(shown.loc[i, profit], shown.loc[i, lead]) for i in shown.index]
        avg = _margin_pct(shown[profit].sum(), shown[lead].sum())
        panels.append({"type": "hbars", "title": f"{profit} margin on {lead} (%)", "labels": labels,
                       "values": margins, "pos": (1, 2, 1, 2), "fmt": "pct", "ref": avg,
                       "ref_label": f"Overall {avg:.1f}%"})
    else:
        panels.append({"type": "hbars", "title": "Number of records", "labels": labels,
                       "values": [float(x) for x in shown["_rows"].tolist()], "pos": (1, 2, 1, 2)})

    # facts + summary
    rows_md, lines = [], []
    hdr = [str(dim)] + [str(c) for c in cols] + ["Share"] + ([f"{profit} margin"] if margins else []) + ["Records"]
    for k, m in enumerate(shown.index):
        row = [str(m)] + [_fmt(shown.loc[m, c], 2) for c in cols] + [_pct(shares[k])]
        if margins:
            row.append(_pct(margins[k]))
        row.append(int(shown.loc[m, "_rows"]))
        rows_md.append(row)
        lines.append("  " + f"{dim}={m}: " + ", ".join(f"{c}={_fmt(shown.loc[m, c], 2)}" for c in cols)
                     + f", share of {lead}={_pct(shares[k])}"
                     + (f", margin={_pct(margins[k])}" if margins else "")
                     + f", records={int(shown.loc[m, 'rows'.join(['_', ''])] if False else shown.loc[m, '_rows'])}")
    ranked = shown[lead].sort_values(ascending=False)
    top_m, low_m = ranked.index[0], ranked.index[-1]
    gap = (float(ranked.iloc[0]) / float(ranked.iloc[-1])) if float(ranked.iloc[-1]) > 0 else float("nan")
    insights = [f"**{top_m}** leads on {lead} with {_fmt(ranked.iloc[0], 2)} ({_pct(shares[labels.index(str(top_m))])} of the total)."]
    if not is_time_dim:
        insights.append(f"**{low_m}** is lowest at {_fmt(ranked.iloc[-1], 2)}"
                        + (f" — the leader is {gap:.1f}× larger." if not pd.isna(gap) else "."))
    if margins:
        mi = int(np.nanargmax(margins))
        ml = int(np.nanargmin(margins))
        insights.append(f"Best {profit} margin: **{labels[mi]}** ({_pct(margins[mi])}); "
                        f"weakest: **{labels[ml]}** ({_pct(margins[ml])}).")
        if labels[mi] != str(top_m):
            insights.append(f"Note the biggest {lead} member ({top_m}) is not the most profitable per unit of {lead}.")
    title = f"{' vs '.join(map(str, cols))} — comparison by {dim}"
    facts = _facts_block(title, assumptions, lines + [f"Leader: {top_m}. Lowest: {low_m}."])
    summary = (f"**{title}**\n\n" + _md_table(hdr, rows_md) + "\n\n" + "\n".join(f"- {i}" for i in insights)
               + ("\n\n_" + " ".join(assumptions) + "_" if assumptions else ""))
    return _base("dim_compare", title, panels, (2, 2), facts, summary, assumptions,
                 reason=f"Comparison of {dim} members across {', '.join(map(str, cols))}.", figsize=(12, 9))


def _plan_ranking(ctx: _Ctx, dim, members, measures, topn=None, ascending=False):
    prim = (measures[0] if measures else None) or ctx.primary
    if not prim:
        return None
    df = ctx.df
    if members:
        df = df[_clean_dim(df, dim).str.lower().isin([m.lower() for m in members])]
    table = _group_table(ctx, dim, [prim], df)
    if len(table) < 2:
        return None
    s = table[prim].sort_values(ascending=ascending)
    limit = topn or 10
    shown = s.iloc[:limit]
    total = float(table[prim].clip(lower=0).sum())
    labels = [str(i) for i in shown.index]
    vals = [float(v) for v in shown.tolist()]
    shares = [(v / total * 100 if total else float("nan")) for v in vals]
    kind_word = "Bottom" if ascending else "Top"
    title = f"{kind_word} {len(labels)} {dim} by {prim}"
    panels = [{"type": "hbars", "title": title, "labels": labels, "values": vals, "pos": (0, 1, 0, 1),
               "share": shares, "highlight_first": True}]
    cum = np.cumsum(s.sort_values(ascending=False).clip(lower=0).to_numpy()) / total * 100 if total else np.array([])
    n80 = int(np.searchsorted(cum, 80) + 1) if len(cum) else None
    panels.append({"type": "donut", "title": f"{prim}: shown vs rest",
                   "labels": labels[:6] + (["Everything else"] if len(table) > len(labels[:6]) else []),
                   "values": vals[:6] + ([max(total - sum(vals[:6]), 0.0)] if len(table) > len(labels[:6]) else []),
                   "pos": (0, 1, 1, 2)})
    lines = [f"  {l}: {prim}={_fmt(v, 2)} ({_pct(sh)} of total)" for l, v, sh in zip(labels, vals, shares)]
    if n80:
        lines.append(f"{n80} of {len(table)} {dim} values account for 80% of {prim}.")
    assumptions = [] if measures else [f"No measure was named, so '{prim}' was used."]
    assumptions.append(f"Ranked across all {len(table)} {dim} values.")
    summary = (f"**{title}**\n\n" + _md_table([str(dim), str(prim), "Share"],
               [[l, _fmt(v, 2), _pct(sh)] for l, v, sh in zip(labels, vals, shares)])
               + (f"\n\n- {n80} of {len(table)} {dim} values account for 80% of {prim}." if n80 else ""))
    return _base("ranking", title, panels, (1, 2), _facts_block(title, assumptions, lines), summary,
                 assumptions, reason=f"Ranking of {dim} by {prim}.", figsize=(13, 6.2))


def _plan_group_trend(ctx: _Ctx, dim, members, measures):
    if ctx.date_col is None:
        return None
    prim = (measures[0] if measures else None) or ctx.primary
    if not prim:
        return None
    df = ctx.df
    if members:
        df = df[_clean_dim(df, dim).str.lower().isin([m.lower() for m in members])]
    labels, by_member, gran = _period_series(ctx, [prim], by_dim=dim, df=df)
    if len(labels) < 2 or not by_member:
        return None
    totals = {m: float(np.nansum(v)) for m, v in by_member.items()}
    order = sorted(totals, key=totals.get, reverse=True)
    limit = 6
    shown = order[:limit]
    series = [{"label": str(m), "values": by_member[m], "color": PALETTE[i % len(PALETTE)]}
              for i, m in enumerate(shown)]
    assumptions = [] if measures else [f"No measure was named, so '{prim}' was used."]
    if len(order) > limit:
        assumptions.append(f"Showing the {limit} largest {dim} values of {len(order)}.")
    assumptions.append(f"Aggregated {gran}ly.")
    title = f"{prim} over time by {dim}"
    panels = [{"type": "line", "title": title, "labels": labels, "series": series, "pos": (0, 1, 0, 1)}]
    lines = []
    for m in shown:
        v = by_member[m]
        first, last = next((x for x in v if x), 0.0), v[-1]
        chg = (last - first) / abs(first) * 100 if first else float("nan")
        lines.append(f"  {dim}={m}: total {prim}={_fmt(totals[m], 2)}, first period={_fmt(first, 2)}, "
                     f"last period={_fmt(last, 2)}, change={_pct(chg)}")
    best = shown[0]
    summary = (f"**{title}**\n\n" + _md_table([str(dim), f"Total {prim}"], [[str(m), _fmt(totals[m], 2)] for m in shown])
               + f"\n\n- **{best}** contributes the most {prim} over the period.")
    return _base("group_trend", title, panels, (1, 1), _facts_block(title, assumptions, lines), summary,
                 assumptions, reason=f"{prim} trend for each {dim}.", is_time=True, figsize=(12, 6.2))


def _rank_dims(ctx: _Ctx, prim, lo=3, hi=30):
    """Order candidate dimensions by how much the measure varies across their members."""
    scored = []
    for d in ctx.dims:
        n = ctx.df[d].nunique(dropna=True)
        if not (lo <= n <= hi) and not (n == 2):
            continue
        t = _group_table(ctx, d, [prim])
        v = t[prim].astype(float)
        if len(v) < 2 or v.mean() == 0:
            continue
        cv = float(v.std(ddof=0) / abs(v.mean()))
        sweet = 1.0 if 3 <= n <= 10 else 0.7
        scored.append((cv * sweet, d))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored]


def _plan_contributors(ctx: _Ctx):
    prim = ctx.primary
    if not prim:
        return None
    dims = _rank_dims(ctx, prim)[:3]
    if not dims:
        return None
    panels, lines, md = [], [], []
    for i, d in enumerate(dims):
        t = _group_table(ctx, d, [prim])
        s = t[prim].sort_values(ascending=False)
        total = float(s.clip(lower=0).sum())
        top = s.iloc[:8]
        shares = [float(v) / total * 100 if total else float("nan") for v in top]
        panels.append({"type": "hbars", "title": f"{d}: share of {prim} (%)", "labels": [str(x) for x in top.index],
                       "values": shares, "pos": (0, 1, i, i + 1), "fmt": "pct", "highlight_first": True})
        lines.append(f"By {d}:")
        lines += [f"  {l}: {_fmt(v, 2)} ({_pct(sh)})" for l, v, sh in zip(top.index, top, shares)]
        md.append(f"**{d}** — leader **{top.index[0]}** with {_pct(shares[0])} of {prim}"
                  + (f", then {top.index[1]} ({_pct(shares[1])})" if len(top) > 1 else "") + ".")
    title = f"Biggest contributors to {prim}"
    assumptions = [f"'{prim}' was used as the measure of overall performance.",
                   f"Dimensions were chosen automatically: {', '.join(map(str, dims))} (the ones where {prim} differs most between members)."]
    summary = f"**{title}**\n\n" + "\n".join(f"- {m}" for m in md) + "\n\n_" + " ".join(assumptions) + "_"
    return _base("contributors", title, panels, (1, len(dims)), _facts_block(title, assumptions, lines), summary,
                 assumptions, reason="Top contributors across the most differentiating dimensions.",
                 figsize=(6.2 * len(dims), 5.8))


def _corr_pairs(ctx: _Ctx, cols):
    data = pd.DataFrame({c: _num(ctx.df[c]) for c in cols})
    corr = data.corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            ta, tb = set(_tok(a)) - _GENERIC_TOKENS, set(_tok(b)) - _GENERIC_TOKENS
            derived = abs(r) >= 0.98 or (ta and ta == tb)
            pairs.append((a, b, float(r), bool(derived)))
    return corr, pairs


def _strength(r):
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


def _plan_relationship(ctx: _Ctx, named):
    cols = list(dict.fromkeys(named))[:2] if len(named) >= 2 else None
    pool = ctx.measures[:]
    if len(pool) < 2:
        return None
    if cols is None:
        pool = sorted(pool, key=lambda c: float(_num(ctx.df[c]).std() / (abs(_num(ctx.df[c]).mean()) or 1)), reverse=True)[:9]
        pool = [c for c in ctx.measures if c in pool]
    corr, pairs = _corr_pairs(ctx, pool if cols is None else cols)
    if not pairs:
        return None
    if cols:
        a, b, r, _ = pairs[0]
    else:
        real = [p for p in pairs if not p[3]] or pairs
        a, b, r, _ = max(real, key=lambda p: abs(p[2]))
    x, y = _num(ctx.df[a]), _num(ctx.df[b])
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return None
    xa, ya = x[ok].to_numpy(), y[ok].to_numpy()
    slope, intercept = np.polyfit(xa, ya, 1) if np.std(xa) > 0 else (0.0, float(np.mean(ya)))
    panels = []
    if cols is None:
        panels.append({"type": "heatmap", "title": "Correlation between measures", "labels": [str(c) for c in pool],
                       "matrix": corr.loc[pool, pool].fillna(0).to_numpy().tolist(), "pos": (0, 1, 0, 1)})
        panels.append({"type": "scatter", "title": f"Strongest non-trivial pair: {b} vs {a}", "x": xa.tolist(),
                       "y": ya.tolist(), "xlabel": str(a), "ylabel": str(b), "slope": float(slope),
                       "intercept": float(intercept), "r": r, "pos": (0, 1, 1, 2)})
        layout, fig = (1, 2), (14, 6.2)
    else:
        panels.append({"type": "scatter", "title": f"{b} vs {a}", "x": xa.tolist(), "y": ya.tolist(),
                       "xlabel": str(a), "ylabel": str(b), "slope": float(slope), "intercept": float(intercept),
                       "r": r, "pos": (0, 1, 0, 1)})
        layout, fig = (1, 1), (10, 6.2)
    top = sorted(pairs, key=lambda p: abs(p[2]), reverse=True)
    lines = [f"  {p[0]} ~ {p[1]}: r={p[2]:+.2f} ({_strength(p[2])}{', derived/duplicate measure' if p[3] else ''})"
             for p in top[:10]]
    lines.append(f"Highlighted pair: {a} vs {b}, r={r:+.2f}, {len(xa)} rows, slope={slope:+.4f}.")
    title = f"Relationship between {a} and {b}" if cols else "Strongest relationships between measures"
    assumptions = []
    if cols is None:
        assumptions.append(f"Compared {len(pool)} numeric measures; ID/year-style columns were ignored.")
        if any(p[3] for p in top[:5]):
            assumptions.append("Pairs with r ≥ 0.98 are near-duplicates (one is derived from the other), "
                               "so the highlighted scatter uses the strongest independent pair.")
    hdr = ["Pair", "r", "Strength"]
    rows = [[f"{p[0]} ↔ {p[1]}", f"{p[2]:+.2f}", _strength(p[2]) + (" (derived)" if p[3] else "")] for p in top[:8]]
    verdict = ("There is no meaningful linear relationship between these measures."
               if abs(r) < 0.2 else f"This is a {_strength(r)} {'positive' if r > 0 else 'negative'} relationship (r = {r:+.2f}).")
    summary = f"**{title}**\n\n" + _md_table(hdr, rows) + f"\n\n- Highlighted: **{a}** vs **{b}**. {verdict}"
    return _base("relationship", title, panels, layout, _facts_block(title, assumptions, lines), summary,
                 assumptions, reason="Pairwise correlation of the numeric measures.", figsize=fig)


def _plan_changes(ctx: _Ctx, measures):
    if ctx.date_col is None:
        return None
    prim = (measures[0] if measures else None) or ctx.primary
    if not prim:
        return None
    labels, series, gran = _period_series(ctx, [prim])
    vals = np.array(series[prim], dtype=float)
    if len(vals) < 3:
        return None
    prev = vals[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(prev != 0, (vals[1:] - prev) / np.abs(prev) * 100, np.nan)
    pl = labels[1:]
    order = np.argsort(-np.abs(np.nan_to_num(pct)))
    hi = [int(i) for i in order[:3]]
    panels = [
        {"type": "line", "title": f"{prim} by {gran}", "labels": labels,
         "series": [{"label": str(prim), "values": vals.tolist(), "color": DARK}],
         "pos": (0, 1, 0, 1), "mark": [i + 1 for i in hi]},
        {"type": "changes", "title": f"{gran.capitalize()}-over-{gran} change in {prim} (%)", "labels": pl,
         "values": [None if np.isnan(v) else float(v) for v in pct], "highlight": hi, "pos": (1, 2, 0, 1)},
    ]
    lines = [f"  {pl[i]}: {pct[i]:+.1f}% ({labels[i]}: {_fmt(vals[i], 2)} → {labels[i + 1]}: {_fmt(vals[i + 1], 2)})" for i in hi]
    title = f"Where {prim} changed the most"
    assumptions = [f"'{prim}' was used as the measure; changes are versus the previous {gran}."]
    summary = (f"**{title}**\n\n" + _md_table(["Period", "Change", "From → To"],
               [[pl[i], f"{pct[i]:+.1f}%", f"{_fmt(vals[i], 1)} → {_fmt(vals[i + 1], 1)}"] for i in hi])
               + f"\n\n- Largest swing: **{pl[hi[0]]}** ({pct[hi[0]]:+.1f}%).")
    return _base("biggest_changes", title, panels, (2, 1), _facts_block(title, assumptions, lines), summary,
                 assumptions, reason="Period-over-period change of the main measure.", is_time=True, figsize=(12, 8.4))


def _plan_unusual(ctx: _Ctx, measures):
    prim = (measures[0] if measures else None) or ctx.primary
    if not prim:
        return None
    v = _num(ctx.df[prim]).dropna()
    if len(v) < 10:
        return None
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (_num(ctx.df[prim]) < lo) | (_num(ctx.df[prim]) > hi)
    n_out = int(out_mask.sum())
    panels = [{"type": "hist", "title": f"Distribution of {prim} per record", "values": v.tolist(), "lo": float(lo),
               "hi": float(hi), "pos": (0, 1, 0, 1)}]
    lines = [f"  {n_out} of {len(v)} records ({n_out / len(v) * 100:.1f}%) fall outside the normal range "
             f"[{_fmt(lo, 1)}, {_fmt(hi, 1)}] for {prim}."]
    dims = _rank_dims(ctx, prim)
    if dims and n_out:
        d = dims[0]
        counts = _clean_dim(ctx.df[out_mask], d).value_counts().head(8)
        if len(counts):
            panels.append({"type": "hbars", "title": f"Where the unusual {prim} values sit ({d})",
                           "labels": [str(i) for i in counts.index], "values": [float(x) for x in counts.tolist()],
                           "pos": (0, 1, 1, 2)})
            lines.append(f"Most outliers are in {d}={counts.index[0]} ({int(counts.iloc[0])}).")
    title = f"Unusual values in {prim}"
    assumptions = ["Outliers use the standard 1.5×IQR rule on the record-level values."]
    summary = f"**{title}**\n\n" + "\n".join(f"- {l.strip()}" for l in lines)
    return _base("unusual", title, panels, (1, len(panels)), _facts_block(title, assumptions, lines), summary,
                 assumptions, reason="Outlier detection on the main measure.", figsize=(6.6 * len(panels), 5.8))


def _plan_overview(ctx: _Ctx):
    prim = ctx.primary
    if not prim:
        return None
    df = ctx.df
    profit = ctx.find_measure("profit")
    total = float(_num(df[prim]).sum()) if ctx.kind_of(prim) == "sum" else float(_num(df[prim]).mean())
    kpis = [f"{'Total' if ctx.kind_of(prim) == 'sum' else 'Avg'} {prim}: {_fmt(total, 1)}"]
    if profit and profit != prim and ctx.kind_of(profit) == "sum":
        p = float(_num(df[profit]).sum())
        kpis.append(f"{profit}: {_fmt(p, 1)} ({_pct(_margin_pct(p, total))} margin)")
    kpis.append(f"{len(df):,} records")
    dims = _rank_dims(ctx, prim)[:2]
    panels, lines, assumptions = [], [f"KPIs: " + " | ".join(kpis)], [f"'{prim}' was treated as the main performance measure."]
    row = 0
    is_time = False
    if ctx.date_col is not None:
        labels, series, gran = _period_series(ctx, [prim])
        if len(labels) >= 2:
            v = series[prim]
            panels.append({"type": "line", "title": f"{prim} over time ({gran}ly)", "labels": labels,
                           "series": [{"label": str(prim), "values": v, "color": DARK}], "pos": (0, 1, 0, 2),
                           "trend": True})
            first, last = next((x for x in v if x), 0.0), v[-1]
            lines.append(f"Trend: {labels[0]} {_fmt(v[0], 1)} → {labels[-1]} {_fmt(v[-1], 1)}; "
                         f"peak {labels[int(np.argmax(v))]} ({_fmt(max(v), 1)}), low {labels[int(np.argmin(v))]} ({_fmt(min(v), 1)}).")
            row, is_time = 1, True
    if row == 0:
        vals = _num(df[prim]).dropna()
        panels.append({"type": "hist", "title": f"Distribution of {prim}", "values": vals.tolist(),
                       "lo": float("nan"), "hi": float("nan"), "pos": (0, 1, 0, 2)})
        row = 1
    md = list(kpis)
    for i, d in enumerate(dims):
        t = _group_table(ctx, d, [prim])
        s = t[prim].sort_values(ascending=False)
        tot = float(s.clip(lower=0).sum())
        top = s.iloc[:8]
        panels.append({"type": "hbars", "title": f"{prim} by {d}", "labels": [str(x) for x in top.index],
                       "values": [float(x) for x in top.tolist()], "pos": (row, row + 1, i, i + 1),
                       "highlight_first": True,
                       "share": [float(x) / tot * 100 if tot else float("nan") for x in top]})
        lines.append(f"By {d}: " + "; ".join(f"{l}={_fmt(v, 1)}" for l, v in list(top.items())[:5]))
        md.append(f"Top {d}: **{top.index[0]}** ({_pct(float(top.iloc[0]) / tot * 100 if tot else float('nan'))} of {prim})")
    title = "Visual overview of the dataset"
    summary = f"**{title}**\n\n" + "\n".join(f"- {m}" for m in md) + "\n\n_" + " ".join(assumptions) + "_"
    layout = (2, 2) if dims else (1, 2)
    return _base("overview", title, panels, layout, _facts_block(title, assumptions, lines), summary, assumptions,
                 reason="Trend plus the main breakdowns of the key measure.", is_time=is_time, figsize=(13, 9))


# ---------------------------------------------------------------------------
# 6. Router
# ---------------------------------------------------------------------------
def plan_dynamic(df: pd.DataFrame, question: str, legacy_intent=None):
    """Return a multi-panel plan, or None when the request belongs to another planner."""
    if df is None or df.empty or not (question or "").strip():
        return None
    if legacy_intent not in (None, "generic_time", "profitability", "customers_workforce"):
        return None
    try:
        ctx = _Ctx(df, question)
        if not ctx.measures:
            return None
        q = ctx.q
        dim, members = ctx.resolve_dim()
        if legacy_intent is not None and dim is None:
            return None
        named = ctx.named_measures(exclude=set(_tok(dim)) if dim else set())
        m = RE_TOPN.search(q)
        topn = int(m.group(1)) if m else None
        ascending = bool(re.search(r"worst|lowest|bottom|least|weakest|smallest|under-?perform", q))
        wants_time = bool(RE_TIME.search(q))

        if dim is None and (RE_REL.search(q) or len(named) >= 2 and re.search(r"relationship|correlat|between", q)):
            return _plan_relationship(ctx, named if len(named) >= 2 else [])
        if dim is None and RE_CHANGE.search(q):
            p = _plan_changes(ctx, named)
            if p:
                return p
        if dim is None and RE_UNUSUAL.search(q) and ctx.date_col is None:
            return _plan_unusual(ctx, named)
        if dim is not None:
            if dim in ctx.time_dims and not wants_time:
                return _plan_dim_compare(ctx, dim, members, named)
            if wants_time and ctx.date_col is not None and dim not in ctx.time_dims:
                p = _plan_group_trend(ctx, dim, members, named)
                if p:
                    return p
            if (RE_RANK.search(q) or RE_CONTRIB.search(q)) and not RE_COMPARE.search(q.replace("most", "")) \
                    or topn or (RE_RANK.search(q) and RE_CONTRIB.search(q)):
                p = _plan_ranking(ctx, dim, members, named, topn, ascending)
                if p:
                    return p
            return _plan_dim_compare(ctx, dim, members, named)
        # ---- no explicit dimension ----------------------------------------------------
        if re.search(r"contributors?|contribut|driving|share of", q):
            p = _plan_contributors(ctx)
            if p:
                return p
        auto_cmp = lambda: _rank_dims(ctx, ctx.primary, lo=3, hi=12)
        if RE_AUTO.search(q):
            best = auto_cmp()
            if best:
                return _plan_dim_compare(
                    ctx, best[0], None, named,
                    auto_note=f"You did not name what to compare, so '{best[0]}' was chosen automatically: it is the "
                              f"grouping where {ctx.primary} differs the most between members.")
        if RE_OVERVIEW.search(q):
            return _plan_overview(ctx)
        if RE_COMPARE.search(q) and not wants_time:
            best = auto_cmp()
            if best:
                return _plan_dim_compare(
                    ctx, best[0], None, named,
                    auto_note=f"No grouping was named, so '{best[0]}' was chosen automatically (largest differences in {ctx.primary}).")
        if RE_RANK.search(q) or re.search(r"important", q):
            p = _plan_contributors(ctx)
            if p:
                return p
        if wants_time:
            return None  # plain "over time" — the time-series planners handle these
    except Exception:  # noqa: BLE001
        logger.exception("Dynamic chart engine failed")
    return None


# ---------------------------------------------------------------------------
# 7. Rendering
# ---------------------------------------------------------------------------
def _fmt_axis(v, _pos=None):
    return G._fmt_num(v, 1)


def _style(ax, bg, text):
    ax.set_facecolor(bg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(text)
        ax.spines[side].set_alpha(0.35)
    ax.tick_params(colors=text, labelsize=9)
    ax.grid(True, axis="y", color=text, alpha=0.10)
    ax.set_axisbelow(True)


def _short(label, n=22):
    s = str(label)
    return s if len(s) <= n else s[: n - 1] + "…"


def _draw_panel(ax, p, bg, text, series_color=None):
    t = p["type"]
    _style(ax, bg, text)
    if t == "bars":
        labels, series = p["labels"], p["series"]
        xs, k = np.arange(len(labels)), len(series)
        width = 0.8 / k
        for j, s in enumerate(series):
            offs = (j - (k - 1) / 2) * width
            col = series_color if (series_color and k == 1) else s["color"]
            bars = ax.bar(xs + offs, s["values"], width=width * 0.94, color=col, label=s["label"],
                          edgecolor=DARK if col == YELLOW else "none", linewidth=0.9)
            if p.get("annotate"):
                for b, v in zip(bars, s["values"]):
                    ax.annotate(_fmt(v, 1), (b.get_x() + b.get_width() / 2, b.get_height()),
                                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=text)
        ax.set_xticks(xs)
        ax.set_xticklabels([_short(l, 14) for l in labels], rotation=0 if len(labels) <= 6 else 35,
                           ha="center" if len(labels) <= 6 else "right")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.margins(y=0.14)
        if k > 1:
            leg = ax.legend(fontsize=9, frameon=False, loc="best")
            for tx in leg.get_texts():
                tx.set_color(text)
    elif t == "hbars":
        labels, vals = p["labels"], [np.nan if v is None else v for v in p["values"]]
        ys = np.arange(len(labels))[::-1]
        colors = [YELLOW] * len(vals)
        if p.get("ref") is not None:
            colors = [GREEN if (not np.isnan(v) and v >= p["ref"]) else GREY for v in vals]
        elif p.get("highlight_first"):
            colors = [YELLOW] + [GREY] * (len(vals) - 1)
        bars = ax.barh(ys, vals, color=series_color or colors, edgecolor=DARK, linewidth=0.8, height=0.68)
        shares = p.get("share")
        for i, (b, v) in enumerate(zip(bars, vals)):
            if np.isnan(v):
                continue
            txt = f"{v:.1f}%" if p.get("fmt") == "pct" else _fmt(v, 1)
            if shares is not None and not np.isnan(shares[i]):
                txt += f"  ({shares[i]:.1f}%)"
            ax.annotate(txt, (b.get_width(), b.get_y() + b.get_height() / 2), textcoords="offset points",
                        xytext=(4, 0), va="center", fontsize=8, color=text)
        ax.set_yticks(ys)
        ax.set_yticklabels([_short(l) for l in labels])
        if p.get("ref") is not None:
            ax.axvline(p["ref"], color=ORANGE, linestyle="--", linewidth=1.4, label=p.get("ref_label", "Average"))
            leg = ax.legend(fontsize=8, frameon=False, loc="lower right")
            for tx in leg.get_texts():
                tx.set_color(text)
        if p.get("fmt") == "pct":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.grid(True, axis="x", color=text, alpha=0.10)
        ax.grid(False, axis="y")
        ax.margins(x=0.22)
    elif t == "donut":
        vals = [max(v, 0) for v in p["values"]]
        if sum(vals) <= 0:
            ax.axis("off")
            return
        cols = (PALETTE[:len(vals)] if len(vals) <= len(PALETTE) else (PALETTE * 3)[:len(vals)])
        wedges, _t, autot = ax.pie(vals, labels=None, colors=cols, startangle=90, counterclock=False,
                                   autopct=lambda pc: f"{pc:.0f}%" if pc >= 4 else "", pctdistance=0.78,
                                   wedgeprops=dict(width=0.42, edgecolor=bg, linewidth=1.5),
                                   textprops=dict(fontsize=8, color=DARK))
        ax.legend(wedges, [_short(l, 18) for l in p["labels"]], fontsize=8, frameon=False, loc="center left",
                  bbox_to_anchor=(0.92, 0.5), labelcolor=text)
        ax.set_aspect("equal")
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    elif t == "line":
        labels = p["labels"]
        xs = np.arange(len(labels))
        for j, s in enumerate(p["series"]):
            v = np.array([np.nan if x is None else x for x in s["values"]], dtype=float)
            col = series_color if (series_color and len(p["series"]) == 1) else s.get("color") or PALETTE[j]
            ax.plot(xs, v, color=col, linewidth=2.3, marker="o", markersize=4,
                    markerfacecolor=YELLOW if (col == DARK and len(p["series"]) == 1) else col,
                    markeredgecolor=DARK, markeredgewidth=0.6, label=s["label"])
            if p.get("trend") and len(p["series"]) == 1:
                ok = ~np.isnan(v)
                if ok.sum() >= 3:
                    k, b = np.polyfit(xs[ok], v[ok], 1)
                    ax.plot(xs, k * xs + b, color=ORANGE, linestyle="--", linewidth=1.4, label="Linear trend")
        for i in p.get("mark", []) or []:
            v = p["series"][0]["values"][i]
            ax.scatter([i], [v], s=110, color=RED, zorder=5, edgecolor=DARK)
            ax.annotate(f"{labels[i]}\n{_fmt(v, 1)}", (i, v), textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=8, color=text, fontweight="bold")
        step = max(1, int(np.ceil(len(labels) / 12)))
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=35, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.margins(y=0.16)
        if len(p["series"]) > 1 or p.get("trend"):
            leg = ax.legend(fontsize=8, frameon=False, loc="best", ncol=2 if len(p["series"]) > 3 else 1)
            for tx in leg.get_texts():
                tx.set_color(text)
    elif t == "changes":
        vals = [np.nan if v is None else v for v in p["values"]]
        xs = np.arange(len(vals))
        colors = [GREEN if (not np.isnan(v) and v >= 0) else RED for v in vals]
        ax.bar(xs, vals, color=colors, edgecolor=DARK, linewidth=0.5, width=0.72, alpha=0.9)
        for i in p.get("highlight", []):
            ax.annotate(f"{vals[i]:+.1f}%", (xs[i], vals[i]), textcoords="offset points",
                        xytext=(0, 5 if vals[i] >= 0 else -13), ha="center", fontsize=9, fontweight="bold", color=text)
        ax.axhline(0, color=text, linewidth=0.8, alpha=0.6)
        step = max(1, int(np.ceil(len(vals) / 12)))
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([p["labels"][i] for i in range(0, len(vals), step)], rotation=35, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.margins(y=0.2)
    elif t == "scatter":
        x, y = np.array(p["x"]), np.array(p["y"])
        ax.scatter(x, y, s=34, color=YELLOW, edgecolor=DARK, linewidth=0.6, alpha=0.85, label="Records")
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, p["slope"] * xx + p["intercept"], color=ORANGE, linewidth=2, linestyle="--",
                label=f"Best fit (r = {p['r']:.2f})")
        ax.set_xlabel(p.get("xlabel", ""), fontsize=10, color=text)
        ax.set_ylabel(p.get("ylabel", ""), fontsize=10, color=text)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.grid(True, axis="x", color=text, alpha=0.10)
        leg = ax.legend(fontsize=9, frameon=False, loc="best")
        for tx in leg.get_texts():
            tx.set_color(text)
    elif t == "heatmap":
        m = np.array(p["matrix"], dtype=float)
        im = ax.imshow(m, cmap="RdYlGn", vmin=-1, vmax=1)
        n = len(p["labels"])
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([_short(l, 14) for l in p["labels"]], rotation=45, ha="right")
        ax.set_yticklabels([_short(l, 14) for l in p["labels"]])
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(m[i, j]) < 0.75 else "white")
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(colors=text, labelsize=8)
    elif t == "hist":
        v = np.array(p["values"], dtype=float)
        ax.hist(v, bins=min(40, max(10, int(np.sqrt(len(v))))), color=YELLOW, edgecolor=DARK, linewidth=0.6)
        if not np.isnan(p.get("lo", np.nan)):
            ax.axvline(p["lo"], color=RED, linestyle="--", linewidth=1.4, label="Normal range (1.5×IQR)")
            ax.axvline(p["hi"], color=RED, linestyle="--", linewidth=1.4)
            leg = ax.legend(fontsize=8, frameon=False)
            for tx in leg.get_texts():
                tx.set_color(text)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_axis))
        ax.set_ylabel("Records", fontsize=9, color=text)
    ax.set_title(p.get("title", ""), fontsize=11, fontweight="bold", color=text, loc="left", pad=8)


def render_panels_png(plan, background_color=None, title=None, series_color=None, dpi=140, **_ignored) -> bytes:
    bg = background_color or "#FFFFFF"
    text = G._contrast(bg)
    rows, cols = plan["layout"]
    fig = plt.figure(figsize=plan.get("figsize") or (12, 4.6 * rows), facecolor=bg)
    try:
        gs = GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.42)
        for p in plan["panels"]:
            r0, r1, c0, c1 = p["pos"]
            ax = fig.add_subplot(gs[r0:r1, c0:c1])
            try:
                _draw_panel(ax, p, bg, text, series_color)
            except Exception:  # noqa: BLE001
                logger.exception("Panel %s failed", p.get("type"))
                ax.axis("off")
        fig.suptitle(title or plan["title"], fontsize=14, fontweight="bold", color=text, x=0.02, ha="left", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg)
        return buf.getvalue()
    finally:
        plt.close(fig)


def render_panels_base64(plan, **kwargs) -> str:
    return "data:image/png;base64," + base64.b64encode(render_panels_png(plan, **kwargs)).decode("ascii")