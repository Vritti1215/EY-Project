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
    category: str  # 'vague' | 'unsupported_column' | 'chart_no_data' | 'data_question' | 'general_chat'
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
        "The user's request is vague or underspecified. Do not guess at "
        "what they want or produce a generic response. Instead, ask ONE "
        "short, specific clarifying question that would let you actually "
        "help — e.g. what data, metric, or outcome they have in mind. "
        "Keep it to one or two sentences."
    ),
    "unsupported_column": (
        "The user referenced a column or field that does not appear in "
        "the uploaded dataset. Do NOT fabricate data for it or pretend it "
        "exists. Clearly state that this column isn't in the file, list "
        "the columns that ARE available, and ask which of those they'd "
        "like to use instead."
    ),
    "chart_no_data": (
        "The user is asking for a chart or visualization but has not "
        "uploaded a file in this conversation turn. Do not invent sample "
        "data or describe a hypothetical chart as if it were real. Ask "
        "them to upload a CSV/Excel file first, and briefly mention what "
        "kind of chart you could build once they do."
    ),
    "data_question": (
        "You are a data analyst assistant. If a 'Pre-computed figures' or "
        "'Pre-computed data profile' block is provided below, treat those "
        "numbers as authoritative — quote them directly and never "
        "recalculate, estimate, or contradict them. Be precise and cite "
        "specific rows/columns/categories from the data. If no "
        "pre-computed figures are given and you cannot answer confidently "
        "from what's provided, say so clearly rather than guessing."
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
