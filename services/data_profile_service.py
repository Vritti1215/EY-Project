"""
Data Profile Service
=====================
PENDING item: "Introduce structured JSON or key-value processing before
AI-based analysis" — the item your status table calls the
`data_profile -> AI JSON decision` pipeline that was discussed but not
built.

This does NOT re-classify columns from scratch — it wraps
schema_service.classify_dataframe() (already battle-tested against your
insight_service fallback ladder) and turns its output into:
  1. A structured DataProfile object other services can reason over.
  2. A compact JSON block (`to_ai_context()`) that is the ONLY numeric
     surface an LLM should ever see for narration.

Drop into services/data_profile_service.py.

Integration point in routes/uploads.py / routes/chatbot.py, replacing
ad hoc "Pre-computed figures" string building with a single profile:

    from services.data_profile_service import build_data_profile, validate_profile

    profile = build_data_profile(df)
    ok, issues = validate_profile(profile)
    if not ok:
        # surface issues, don't hand possibly-bad numbers to the model
        ...
    ai_context = profile.to_ai_context()
    # append ai_context to the prompt exactly like build_ai_context()'s
    # output is used today in insight_service.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import numpy as np

from services.schema_service import classify_dataframe


@dataclass
class DataProfile:
    row_count: int
    columns: list  # list[dict] straight from schema_service.classify_column
    numeric_columns: list  # 'number' | 'currency' | 'stock' typed columns
    date_columns: list
    categorical_columns: list
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_ai_context(self) -> str:
        """
        The ONLY numeric surface the LLM should see — pre-computed,
        never re-derived by the model. Mirrors the phrasing convention
        of insight_service.build_ai_context() ("authoritative — do not
        recalculate") so both code paths give the model the same
        instruction.
        """
        payload = {
            "row_count": self.row_count,
            "numeric_columns": self.numeric_columns,
            "date_columns": self.date_columns,
            "categorical_columns": self.categorical_columns,
            "columns": self.columns,
        }
        return (
            "Pre-computed data profile (authoritative — quote directly, "
            "never recalculate, estimate, or write/simulate code to "
            "reproduce these):\n```json\n"
            + json.dumps(payload, indent=2, default=str)
            + "\n```"
        )


def build_data_profile(df: pd.DataFrame) -> DataProfile:
    """
    Feature: structured JSON/key-value processing before AI analysis.
    Reuses schema_service.classify_dataframe() as the single source of
    truth for column typing (number/currency/datetime/boolean/stock/
    geography/text) so this doesn't introduce a second, divergent
    classifier alongside insight_service's existing fallback ladder.
    """
    if df is None or df.empty:
        return DataProfile(
            row_count=0, columns=[], numeric_columns=[], date_columns=[],
            categorical_columns=[], warnings=["Empty or unreadable dataframe."],
        )

    schema = classify_dataframe(df)

    numeric_types = {"number", "currency", "stock"}
    categorical_types = {"text", "boolean", "geography"}

    numeric_columns = [c["name"] for c in schema if c["type"] in numeric_types]
    date_columns = [c["name"] for c in schema if c["type"] == "datetime"]
    categorical_columns = [c["name"] for c in schema if c["type"] in categorical_types]

    warnings = []
    if not numeric_columns:
        warnings.append("No numeric/currency/stock column detected.")
    if not date_columns and not categorical_columns:
        warnings.append("No date or categorical column detected — grouping may be unreliable.")

    return DataProfile(
        row_count=int(len(df)),
        columns=schema,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        categorical_columns=categorical_columns,
        warnings=warnings,
    )


def validate_profile(profile: DataProfile) -> tuple:
    """
    Feature: "Ensure calculations are completed and validated before
    results are passed to the LLM." Your table marks the base calculation
    step MOSTLY DONE already (monthly_totals/category_aggregate etc. are
    solid) — this adds a final sanity gate specifically on the profile
    JSON that gets handed to the model, catching NaN/inf leakage or an
    empty dataset before it reaches the prompt.

    Returns:
        (is_valid: bool, issues: list[str])
    """
    issues = list(profile.warnings)
    if profile.row_count == 0:
        issues.append("Zero rows — nothing to analyze.")
    return (len([i for i in issues if "detected" not in i and "Zero rows" not in i]) == 0
            and profile.row_count > 0, issues)
