"""
Prompt Guidance Service
==========================
PARTIAL item: "Improve handling of generic prompts and edge cases through
system-prompt enhancements."

insight_service.py's tabular Q&A path already has a decent system prompt
("quote pre-computed figures, say so clearly if you can't answer"). What's
still missing per the status table: strengthening this for AMBIGUOUS or
UNSUPPORTED requests specifically — e.g. "make it better", "analyze this"
with no question, or asking for a chart type / column that doesn't exist
in the data.

This module classifies the incoming message into a few deterministic
buckets and returns the right system prompt for each, so ai_service.py's
generic default doesn't have to carry every case alone.

Drop into services/prompt_guidance.py.

Integration point wherever get_chat_response(..., system_prompt=...) is
currently called with a hand-built string (routes/chatbot.py, uploads.py):

    from services.prompt_guidance import classify_prompt_intent, system_prompt_for

    intent = classify_prompt_intent(message, has_file=bool(attached_file),
                                     available_columns=df.columns.tolist() if df is not None else None)
    system_prompt = system_prompt_for(intent)
    result = ai_service.get_chat_response(prompt, system_prompt=system_prompt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

VAGUE_PHRASES = (
    "make it better", "analyze this", "what do you think", "tell me about this",
    "help me", "do something", "improve this", "fix this", "look at this",
)

CHART_REQUEST_KEYWORDS = (
    "graph", "chart", "plot", "visualize", "visualise", "trend", "diagram",
)


@dataclass
class PromptIntent:
    category: str  # 'vague' | 'vague_with_data' | 'unsupported_column' | 'chart_no_data' | 'data_question' | 'general_chat'
    detail: Optional[str] = None


def classify_prompt_intent(
    message: str,
    has_file: bool = False,
    available_columns: Optional[list] = None,
) -> PromptIntent:
    """
    Deterministic bucket assignment — no AI call. Keeps the "which system
    prompt to use" decision fast and predictable.
    """
    text = (message or "").strip().lower()

    if not text:
        return PromptIntent("vague", "empty message")

    if any(phrase in text for phrase in VAGUE_PHRASES) and len(text.split()) <= 6:
        # A vague request against an already-uploaded file with usable
        # columns is NOT the same as a vague request with nothing to work
        # from: when there's data, the model should infer a sensible
        # default analysis instead of stalling on a clarifying question
        # (see 'vague_with_data' below) — flexibility over rigid prompts.
        if has_file and available_columns:
            return PromptIntent("vague_with_data", text)
        return PromptIntent("vague", text)

    wants_chart = any(k in text for k in CHART_REQUEST_KEYWORDS)
    if wants_chart and not has_file:
        return PromptIntent("chart_no_data", text)

    if has_file and available_columns:
        # Look for a column-like mention (quoted or capitalized word) that
        # doesn't match any real column — a common source of "unsupported"
        # requests ("chart the Profit column" when no such column exists).
        mentioned = re.findall(r'"([^"]+)"|\b([A-Z][a-zA-Z]{2,})\b', message or "")
        mentioned_flat = [m[0] or m[1] for m in mentioned]
        known_lower = {str(c).lower() for c in available_columns}
        unresolved = [m for m in mentioned_flat if m.lower() not in known_lower and len(m) > 3]
        if wants_chart and unresolved and not any(u.lower() in known_lower for u in unresolved):
            # Only flag when NONE of the mentioned terms match a real column
            # — avoids false positives on ordinary capitalized words.
            plausible_column_ref = any(
                any(word in c.lower() for word in unresolved[0].lower().split())
                for c in known_lower
            )
            if not plausible_column_ref:
                return PromptIntent("unsupported_column", unresolved[0])

    if has_file:
        return PromptIntent("data_question", text)

    return PromptIntent("general_chat", text)


_PROMPTS = {
    "vague": (
        "The user's request is vague or underspecified and no data has "
        "been provided this turn. Do not guess at what they want or "
        "produce a generic response. Instead, ask ONE short, specific "
        "clarifying question that would let you actually help — e.g. what "
        "data, metric, or outcome they have in mind. Keep it to one or "
        "two sentences."
    ),
    "vague_with_data": (
        "The user's request is vague (e.g. 'analyze this', 'show me "
        "something useful') but a dataset IS available this turn — a "
        "'STRUCTURED DATA PROFILE' / schema block below lists the real "
        "columns and their inferred types (numeric, date, categorical, "
        "currency, etc). Do NOT stall on a clarifying question. Instead, "
        "infer the most useful default analysis yourself: pick the most "
        "business-relevant numeric column as the financial measure "
        "(revenue/sales/amount-like columns over generic counts when "
        "both exist), pick a sensible grouping or date column to break "
        "it down by, and choose the chart type that best fits that shape "
        "(bar for category comparisons, line/area for a date/time axis, "
        "pie only for a small number of share-of-whole categories). State "
        "briefly what you chose to analyze and why, so the user can "
        "redirect you if it's not what they meant — but always deliver a "
        "real, data-grounded answer first rather than asking permission."
    ),
    "unsupported_column": (
        "The user referenced a column or field that does not appear in "
        "the uploaded dataset. Do NOT fabricate data for it or pretend it "
        "exists. Clearly state that this column isn't in the file, list "
        "the columns that ARE available, and — instead of stopping there "
        "— proceed to answer using the closest reasonable match from the "
        "real columns (e.g. 'Revenue' when they said 'Sales'), noting "
        "that substitution explicitly."
    ),
    "chart_no_data": (
        "The user is asking for a chart or visualization but has not "
        "uploaded a file in this conversation turn. Do not invent sample "
        "data or describe a hypothetical chart as if it were real. Ask "
        "them to upload a CSV/Excel file first, and briefly mention what "
        "kind of chart you could build once they do."
    ),
    "data_question": (
        "You are a data analyst assistant. A 'STRUCTURED DATA PROFILE' "
        "schema block below lists the real columns and their inferred "
        "types — use it to work out, for THIS question: (1) the user's "
        "analytical intent (comparison, trend, breakdown, ranking, "
        "total, etc.), (2) which column(s) are actually relevant even if "
        "the user's wording doesn't match a column name exactly (loose, "
        "flexible matching — 'income' can mean a 'Revenue' column, 'by "
        "region' can mean a 'State'/'Territory' column), and (3) the "
        "financial measure that answers it. If a 'Pre-computed figures' "
        "or 'Pre-computed data profile' block is provided, treat those "
        "numbers as authoritative — quote them directly and never "
        "recalculate, estimate, or contradict them. Be precise and cite "
        "specific rows/columns/categories from the data. Only say you "
        "can't answer confidently if the data genuinely has nothing "
        "relevant to work with — prefer a reasonable, clearly-labeled "
        "best-effort answer over refusing."
    ),
    "general_chat": (
        "You are an intelligent AI assistant built for an enterprise "
        "environment. You provide clear, professional, and helpful "
        "responses. Format your responses with markdown when appropriate. "
        "Be concise but thorough. If a request is ambiguous, ask a brief "
        "clarifying question rather than assuming."
    ),
}


def system_prompt_for(intent: PromptIntent) -> str:
    """Return the system prompt string for a classified intent bucket."""
    return _PROMPTS.get(intent.category, _PROMPTS["general_chat"])


# --- NEW: generic-input / edge-case reinforcement, layered on top of any
# of the buckets above whenever a file/schema is in play. This is the
# piece that specifically targets "the model should infer required
# columns, financial measures, and visualization type, and should handle
# generic/edge-case inputs flexibly rather than requiring a rigid,
# highly-structured prompt from the user."
EDGE_CASE_INSTRUCTIONS = (
    "\n\nHANDLING GENERIC & EDGE-CASE INPUTS:\n"
    "- The user will often phrase things casually or imprecisely — infer "
    "intent from context and the available schema rather than requiring "
    "exact column names or rigid phrasing.\n"
    "- If the dataset has only one plausible numeric column, use it "
    "without asking which measure they mean.\n"
    "- If a request could reasonably map to more than one column (e.g. "
    "two date-like columns), pick the more likely one based on naming "
    "and mention the alternative briefly rather than blocking on it.\n"
    "- If the file has no numeric column at all, say so plainly and "
    "offer a count/category breakdown instead of a financial one.\n"
    "- If the file is empty, has a single row, or a single column, state "
    "that limitation directly instead of fabricating trends or "
    "comparisons that the data can't actually support.\n"
    "- Never fabricate columns, rows, or figures that are not present in "
    "the provided schema or pre-computed results."
)
