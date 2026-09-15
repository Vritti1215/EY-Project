"""
Structured AI chart intelligence.

Architecture:
    DataFrame -> structured profile -> AI JSON decision -> strict validation
    -> deterministic Pandas calculation -> server-side chart rendering.

The AI never computes numeric values.
"""
import json
import logging
import re
import pandas as pd
from services.schema_service import (
    classify_dataframe, resolve_numeric_column, resolve_date_column,
    resolve_group_column, detect_financial_ambiguity,
)

logger = logging.getLogger(__name__)

VALID_CHART_TYPES = ("bar", "barh", "line", "area", "pie", "histogram")
VALID_AGGREGATIONS = ("count", "sum", "mean", "min", "max", "none")

# Kept in sync with services/chart_service.py's BRAND_PALETTES /
# COLOR_ALIASES: 'ey' is the black/yellow/grey/white brand theme; blue/
# red/green are the additional single-hue alternatives. Anything mapped
# to a token here (not a raw matplotlib colormap name) is resolved by
# ChartService.resolve_palette() at render time.
COLORS = {
    "blue": "blue", "red": "red", "green": "green",
    "black": "ey", "yellow": "ey", "grey": "ey", "gray": "ey", "white": "ey",
    "purple": "Purples", "orange": "Oranges", "teal": "GnBu",
    "pink": "RdPu", "brown": "YlOrBr",
}
STYLES = {
    "dark": "dark", "darkgrid": "darkgrid", "white": "white",
    "whitegrid": "whitegrid", "minimal": "white", "clean": "white",
    "ticks": "ticks",
}


def build_data_profile(df):
    schema = classify_dataframe(df)
    cols = []
    for item in schema[:30]:
        name = item["name"]
        s = df[name]
        entry = dict(item)
        entry["sample"] = [str(x)[:80] for x in s.dropna().head(5).tolist()]
        if item["type"] in {"number", "currency", "stock"}:
            n = pd.to_numeric(s, errors="coerce").dropna()
            if len(n):
                entry["min"] = float(n.min())
                entry["max"] = float(n.max())
                entry["mean"] = round(float(n.mean()), 4)
        cols.append(entry)
    return {"rows": int(len(df)), "columns": cols}


def _json_from_response(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def parse_natural_customization(question):
    q = (question or "").lower()
    out = {}

    for typ, words in {
        "bar": ("bar chart", "bar graph", "bars"),
        "barh": ("horizontal bar", "horizontal bars"),
        "line": ("line chart", "line graph", "line plot"),
        "area": ("area chart", "area graph"),
        "pie": ("pie chart", "pie graph"),
        "histogram": ("histogram", "distribution chart", "frequency chart"),
    }.items():
        if any(w in q for w in words):
            out["chart_type"] = typ
            break

    for word, palette in COLORS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            out["palette"] = palette
            break

    for word, style in STYLES.items():
        if word in q:
            out["style"] = style
            break

    months = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
        "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
        "aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,
        "nov":11,"november":11,"dec":12,"december":12,
    }
    found = []
    for word, num in months.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            found.append(num)
    if len(found) >= 2 and any(x in q for x in (" to ", "-", "through", "from ")):
        out["month_start"], out["month_end"] = min(found[:2]), max(found[:2])

    return out


def _fallback(df, question):
    schema = classify_dataframe(df)
    numeric = resolve_numeric_column(df, schema, message=question)
    date = resolve_date_column(df, schema, message=question)
    group = resolve_group_column(df, schema, message=question, exclude={numeric, date})

    if date and numeric:
        return {
            "best_chart_type":"line", "x_column":date, "y_column":numeric,
            "group_column":None, "aggregation":"sum",
            "reason":"Ordered time data with a numeric measure.",
            "alternatives":["bar","area"]
        }
    if group and numeric:
        return {
            "best_chart_type":"bar", "x_column":group, "y_column":numeric,
            "group_column":group, "aggregation":"sum",
            "reason":"Categorical comparison of a numeric measure.",
            "alternatives":["barh","pie"]
        }
    if group:
        return {
            "best_chart_type":"bar", "x_column":group, "y_column":None,
            "group_column":group, "aggregation":"count",
            "reason":"Categorical record distribution.",
            "alternatives":["barh","pie"]
        }
    if numeric:
        return {
            "best_chart_type":"histogram", "x_column":None, "y_column":numeric,
            "group_column":None, "aggregation":"none",
            "reason":"Single numeric measure with no date/category to group by — a histogram shows its distribution.",
            "alternatives":["bar","barh"]
        }
    return {
        "best_chart_type":"bar", "x_column":None, "y_column":None,
        "group_column":None, "aggregation":"count",
        "reason":"Safe fallback.", "alternatives":[]
    }


def recommend_chart_plan(df, question, ai_service=None, overrides=None):
    schema = classify_dataframe(df)
    profile = build_data_profile(df)
    ambiguity = detect_financial_ambiguity(schema, message=question)
    fallback = _fallback(df, question)
    overrides = overrides or {}
    custom = parse_natural_customization(question)
    ai_plan = {}

    if ai_service:
        ambiguity_note = ""
        if ambiguity["is_ambiguous"]:
            ambiguity_note = (
                "\n\nIMPORTANT — this dataset has more than one plausible "
                "column of the same kind:\n"
                f"  Candidate date columns: {ambiguity['ambiguous_dates'] or 'n/a'}\n"
                f"  Candidate value columns: {ambiguity['ambiguous_values'] or 'n/a'}\n"
                "Use the DATA PROFILE (column names, sample values, min/max/mean) "
                "and the USER REQUEST wording together to infer which one is "
                "actually relevant — e.g. prefer a 'Revenue'/'Sales'/'Amount'-style "
                "column over a 'Discount'/'Tax'/'Fee' column unless the request "
                "says otherwise, and prefer a transactional date (Order/Invoice "
                "Date) over a secondary one (Due/Ship Date) unless the request "
                "specifically asks about the secondary date. Do not just pick the "
                "first column in the profile."
            )
        prompt = (
            "You are the chart-planning layer of an analytics application.\n"
            "Do NOT calculate any values. Select only real columns from the supplied profile.\n"
            "Use contextual understanding of the column names, sample values and "
            "statistics in the DATA PROFILE — not just their position — to decide "
            "which columns are most relevant to the USER REQUEST.\n"
            "Choose the best chart and up to two alternatives."
            + ambiguity_note
            + "\n\nDATA PROFILE:\n" + json.dumps(profile, ensure_ascii=False)
            + "\n\nUSER REQUEST:\n" + question
            + "\n\nReturn ONLY JSON:\n"
            '{"best_chart_type":"bar|barh|line|area|pie|histogram",'
            '"x_column":"exact column or null",'
            '"y_column":"exact numeric column or null",'
            '"group_column":"exact categorical column or null",'
            '"aggregation":"count|sum|mean|min|max|none",'
            '"reason":"short reason, and if multiple date/value columns existed, why this one was chosen",'
            '"alternatives":["bar|barh|line|area|pie|histogram"]}'
        )
        try:
            result = ai_service.get_chat_response(
                prompt,
                system_prompt=(
                    "Return valid JSON only. Never invent a column. Never calculate a number. "
                    "Never return markdown."
                )
            )
            ai_plan = _json_from_response(result.get("response"))
        except Exception as exc:
            logger.warning("Chart planner AI failed: %s", exc)

    valid = {str(c): c for c in df.columns}
    plan = dict(fallback)

    if ai_plan:
        typ = str(ai_plan.get("best_chart_type","")).lower()
        if typ in VALID_CHART_TYPES:
            plan["best_chart_type"] = typ

        for key in ("x_column","y_column","group_column"):
            val = ai_plan.get(key)
            if val is None:
                plan[key] = None
            elif str(val) in valid:
                plan[key] = valid[str(val)]

        agg = str(ai_plan.get("aggregation","")).lower()
        if agg in VALID_AGGREGATIONS:
            plan["aggregation"] = agg

        if ai_plan.get("reason"):
            plan["reason"] = str(ai_plan["reason"])

        alts = ai_plan.get("alternatives")
        if isinstance(alts, list):
            plan["alternatives"] = [
                str(x).lower() for x in alts
                if str(x).lower() in VALID_CHART_TYPES
                and str(x).lower() != plan["best_chart_type"]
            ][:2]

    # User explicitly requested customization > AI decision.
    plan["best_chart_type"] = overrides.get("chart_type") or custom.get("chart_type") or plan["best_chart_type"]
    plan["palette"] = overrides.get("palette") or custom.get("palette")
    plan["style"] = overrides.get("style") or custom.get("style")
    plan["month_start"] = custom.get("month_start")
    plan["month_end"] = custom.get("month_end")
    plan["selection_source"] = "ai" if ai_plan else "deterministic_fallback"
    plan["profile"] = profile
    plan["ambiguity"] = ambiguity
    plan["warnings"] = [ambiguity["message"]] if ambiguity["is_ambiguous"] else []

    # Final validation.
    if plan["best_chart_type"] not in VALID_CHART_TYPES:
        plan["best_chart_type"] = fallback["best_chart_type"]
    for key in ("x_column","y_column","group_column"):
        if plan.get(key) not in df.columns:
            plan[key] = fallback.get(key)
    if plan.get("aggregation") not in VALID_AGGREGATIONS:
        plan["aggregation"] = fallback["aggregation"]

    return plan


def _period_values(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= .60:
        return parsed, "datetime"
    mapping = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
        "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
        "aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,
        "nov":11,"november":11,"dec":12,"december":12,
    }
    return series.astype(str).str.strip().str.lower().map(mapping), "month"


def prepare_chart_data(df, plan):
    work = df.copy()
    x = plan.get("x_column")
    y = plan.get("y_column")
    group = plan.get("group_column") or x
    agg = plan.get("aggregation","count")

    if x and x in work.columns:
        period, kind = _period_values(work[x])
        work["_chart_key"] = period
        if plan.get("month_start") and plan.get("month_end"):
            if kind == "datetime":
                work = work[work["_chart_key"].dt.month.between(
                    int(plan["month_start"]), int(plan["month_end"])
                )]
            else:
                work = work[work["_chart_key"].between(
                    int(plan["month_start"]), int(plan["month_end"])
                )]
        key = "_chart_key"
    elif group and group in work.columns:
        work["_chart_key"] = work[group].astype(str).str.strip()
        work = work[work["_chart_key"].ne("")]
        key = "_chart_key"
    else:
        if not y or y not in work.columns:
            raise ValueError("No usable column was selected.")
        vals = pd.to_numeric(work[y], errors="coerce").dropna()
        return [str(i+1) for i in range(min(60,len(vals)))], [float(v) for v in vals.iloc[:60]], y

    if agg == "count" or not y:
        grouped = work.groupby(key, dropna=True).size()
        ylabel = "Count"
    else:
        work["_numeric"] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=["_numeric"])
        grouped = getattr(work.groupby(key)["_numeric"], agg)()
        ylabel = str(y)

    if isinstance(grouped.index, pd.DatetimeIndex):
        grouped = grouped.sort_index()

    pairs = []
    for k, v in grouped.items():
        if pd.isna(k):
            continue
        if isinstance(k, pd.Timestamp):
            label = k.strftime("%b %Y")
        elif isinstance(k, (int,float)) and 1 <= float(k) <= 12:
            label = pd.Timestamp(2000, int(k), 1).strftime("%b")
        else:
            label = str(k)
        pairs.append((label, float(v)))

    if len(pairs) > 60:
        pairs = sorted(pairs, key=lambda p:p[1], reverse=True)[:60]
    return [p[0] for p in pairs], [p[1] for p in pairs], ylabel


def build_chart_variants(df, plan, chart_service):
    labels, values, ylabel = prepare_chart_data(df, plan)
    best = plan["best_chart_type"]
    alternatives = [x for x in plan.get("alternatives",[]) if x != best][:2]

    # Guarantee useful alternatives where possible.
    for candidate in ("line","bar","barh","area","pie","histogram"):
        if len(alternatives) >= 2 or candidate == best:
            continue
        if candidate == "pie" and (len(labels) > 8 or any(v < 0 for v in values)):
            continue
        if candidate == "histogram" and plan.get("group_column"):
            # Histogram plots a raw numeric distribution — not a
            # meaningful alternative once data has been grouped/aggregated
            # by a category or date.
            continue
        alternatives.append(candidate)

    xlabel = str(plan.get("x_column") or plan.get("group_column") or "Period")
    title = f"{ylabel} by {xlabel}"
    variants = []

    for typ in [best] + alternatives:
        try:
            image = chart_service.render_base64(
                labels, values, chart_type=typ,
                title=title if typ == best else f"{title} — {typ.title()} alternative",
                xlabel=plan.get("xlabel") or xlabel,
                ylabel=plan.get("ylabel") or ylabel,
                palette=plan.get("palette"),
                style=plan.get("style"),
            )
            variants.append({"chart_type":typ, "chart_image":image, "title":title})
        except Exception as exc:
            logger.warning("Chart variant %s failed: %s", typ, exc)

    return {
        "labels": labels, "values": values, "ylabel": ylabel,
        "best_chart_type": best, "variants": variants, "plan": plan
    }