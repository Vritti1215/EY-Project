"""
Multi-Chart Recommender
=========================
PENDING item: "Explore generating multiple recommended chart types from a
single user prompt."

insight_service.suggest_chart_type() already picks ONE best type
(AI-assisted, heuristic fallback guaranteed). This module reuses that
exact function unchanged as the "Best" pick, then adds 1-2 deterministic
*alternatives* from the same shape signal — no second AI call needed.

Drop into services/chart_recommender.py.

Usage (e.g. in routes/chatbot.py, alongside the existing suggest_chart_type
call it already makes):

    from services.chart_recommender import recommend_chart_set

    chart_set = recommend_chart_set(
        labels, values,
        is_time_series=is_time_series,
        ai_service=ai_service,
        group_label=group_col,
    )
    # chart_set.best.chart_type -> exactly what suggest_chart_type() would
    #                               have returned; render this as today
    # chart_set.alternatives     -> render as extra thumbnails if the
    #                               client requests a multi-chart response
    # chart_set.summary_text()   -> "Best: Line, alternatives: Bar + Area"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from services.insight_service import suggest_chart_type, VALID_CHART_TYPES


@dataclass
class ChartPick:
    chart_type: str
    reason: str
    source: str  # 'ai' | 'heuristic' | 'heuristic_fallback' | 'alternative'


@dataclass
class ChartRecommendationSet:
    best: ChartPick
    alternatives: list = field(default_factory=list)

    def all_types(self) -> list:
        return [self.best.chart_type] + [a.chart_type for a in self.alternatives]

    def summary_text(self) -> str:
        text = f"Best: {self.best.chart_type.capitalize()}"
        if self.alternatives:
            alts = " + ".join(a.chart_type.capitalize() for a in self.alternatives)
            text += f", alternatives: {alts}"
        return text


_REASONS = {
    "bar": "Compares a moderate number of categories side by side.",
    "barh": "Keeps long category labels readable when there are many groups.",
    "line": "Shows the trend/direction across an ordered time axis.",
    "area": "Emphasizes magnitude/volume over time, not just direction.",
    "pie": "Shows each category's share of one whole.",
}


def _alternatives_for(best_type: str, n_categories: int, is_time_series: bool) -> list:
    """
    Deterministic alternative picks using the same shape signals
    suggest_chart_type()'s own heuristic already considers (category
    count, time-series flag). No AI call.
    """
    if is_time_series:
        candidates = ["area", "bar"] if best_type == "line" else ["line", "bar"]
    elif n_categories <= 6:
        candidates = ["bar", "pie"] if best_type != "pie" else ["bar", "barh"]
    elif n_categories > 12:
        candidates = ["barh", "bar"] if best_type != "barh" else ["bar", "line"]
    else:
        candidates = ["bar", "barh"] if best_type != "bar" else ["barh", "line"]

    out = []
    for c in candidates:
        if c != best_type and c in VALID_CHART_TYPES:
            out.append(ChartPick(c, _REASONS.get(c, ""), "alternative"))
        if len(out) == 2:
            break
    return out


def recommend_chart_set(
    labels: list,
    values: list,
    is_time_series: bool = False,
    ai_service=None,
    group_label: Optional[str] = None,
    max_alternatives: int = 2,
) -> ChartRecommendationSet:
    """
    Wraps insight_service.suggest_chart_type() unchanged (still the single
    source of truth for the "Best" pick — still AI-assisted with a
    guaranteed heuristic fallback) and adds ranked alternatives so a
    single prompt can return multiple recommended chart types.
    """
    best_type, source = suggest_chart_type(
        labels, values, is_time_series=is_time_series,
        ai_service=ai_service, group_label=group_label,
    )
    best = ChartPick(best_type, _REASONS.get(best_type, ""), source)
    alternatives = _alternatives_for(best_type, len(labels or []), is_time_series)[:max_alternatives]
    return ChartRecommendationSet(best=best, alternatives=alternatives)