"""Tests for services/generic_chart_planner.py (run: python -m pytest tests/test_generic_chart_planner.py)."""
import json
import numpy as np
import pandas as pd

from services import generic_chart_planner as gp


def _financial_df():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=48, freq="MS")
    rev = 100 + np.arange(48) * 2 + rng.normal(0, 3, 48)
    return pd.DataFrame({
        "Date": dates, "Year": dates.year, "Quarter": ["Q%d" % ((d.month - 1) // 3 + 1) for d in dates],
        "Revenue": rev, "COGS": rev * 0.5, "Operating Expenses": rev * 0.2,
        "Net Profit": rev * 0.1, "EBITDA": rev * 0.2,
        "Customers": 1000 + np.arange(48) * 10, "Employees": 50 + np.arange(48) // 6,
    })


class _FakeAI:
    def __init__(self, spec):
        self.spec = spec

    def get_chat_response(self, prompt, system_prompt=None, **kw):
        return {"response": json.dumps(self.spec)}


def test_generic_time_is_chronological_not_row_counts():
    plan = gp.plan_chart(_financial_df(), "Create a graph showing how the data changes over time.")
    assert plan and plan["intent"] == "generic_time"
    assert len(plan["labels"]) > 12          # not collapsed into 12 month buckets
    assert plan["series"][0]["label"].startswith("Revenue")


def test_business_words_map_to_real_columns():
    plan = gp.plan_chart(_financial_df(), "Show how the company's workforce has changed over time.")
    assert plan and "Employees" in plan["title"]
    plan = gp.plan_chart(_financial_df(), "Show how the customer base has changed over time.")
    assert plan and "Customers" in plan["title"]


def test_missing_columns_return_none_instead_of_bogus_chart():
    df = _financial_df().drop(columns=["Employees"])
    assert gp.plan_chart(df, "Visualise how the workforce has changed over time.") is None


def test_every_intent_renders_a_png():
    df = _financial_df()
    for q in (
        "Highlight periods where costs appear to be increasing faster than income.",
        "Visualise the relationship between the company's income and profitability.",
        "Compare the company's financial performance before and after major changes.",
        "Identify and visualise periods of unusually high or low profitability.",
        "Show which periods had the largest improvements in performance.",
        "Create a graph showing how efficiently the company is performing over time.",
    ):
        plan = gp.plan_chart(df, q)
        assert plan, q
        assert gp.render_plan_png(plan)[:4] == b"\x89PNG"


def test_ai_plan_is_validated_against_real_columns():
    df = pd.DataFrame({"Region": ["N", "S", "n", "E"] * 5, "Amount": np.arange(20.0)})
    good = {"supported": True, "chart": "bar", "x": "Region", "y": ["Amount"], "agg": "sum"}
    plan = gp.plan_from_ai(df, "which region is best", _FakeAI(good))
    assert plan and sorted(plan["labels"]) == ["E", "N", "S"]      # 'N' and 'n' merged
    bad = {"supported": True, "chart": "bar", "x": "Nope", "y": ["Ghost"], "agg": "sum"}
    plan = gp.plan_from_ai(df, "which region is best", _FakeAI(bad))
    assert plan is None or "Ghost" not in json.dumps(plan)          # never invents a column

    # An explicit "supported: false" from the model raises ChartNotSupported
    # (caught by the route and turned into a plain "data not available" message)
    # rather than silently returning None.
    import pytest
    with pytest.raises(gp.ChartNotSupported):
        gp.plan_from_ai(df, "x", _FakeAI({"supported": False, "reason": "no such column"}))


def test_fy_parsing_is_april_to_march():
    fys = gp.detect_fiscal_years("compare FY24 and FY2025 revenue")
    assert [l for l, _, _ in fys] == ["FY24", "FY25"]
    _, start, end = fys[1]
    assert (start.month, start.day) == (4, 1) and (end.month, end.day) == (3, 31)
    assert start.year == 2024 and end.year == 2025


def test_fy_not_in_data_is_reported_not_silently_charted():
    df = _financial_df()  # Jan 2021 .. Dec 2024
    avail = gp.fy_availability(df, "show revenue for FY30")
    assert avail is not None and avail["available"]["FY30"] is False
    assert gp.plan_chart(df, "show revenue for FY30") is None


def test_fy_compare_only_uses_years_present_in_the_data():
    df = _financial_df()
    plan = gp.plan_chart(df, "compare FY22 and FY23 revenue")
    assert plan and plan["intent"] == "fy_compare"
    assert plan["labels"] == ["FY22", "FY23"]


def test_every_financial_year_wording_means_april_to_march():
    for q in ("FY25", "fy 2025", "FY'25", "FY-25", "FY24-25", "FY2024-2025", "F.Y. 2024-25",
              "financial year 2025", "fiscal yr 25", "FYE 2025"):
        (label, start, end), = gp.detect_fiscal_years(q)
        assert label == "FY25", q
        assert (start.year, start.month) == (2024, 4) and (end.year, end.month) == (2025, 3), q


def test_fy_quarters_halves_and_relative_terms():
    (l, s, e), = gp.detect_fiscal_years("revenue Q2 FY25")
    assert (l, s.month, e.month) == ("Q2 FY25", 7, 9)
    (l, s, e), = gp.detect_fiscal_years("H1 FY25")
    assert (s.month, e.month) == (4, 9)
    ref = pd.Timestamp(2025, 12, 1)
    assert [x[0] for x in gp.detect_fiscal_years("this fy vs last financial year", ref_date=ref)] == ["FY25", "FY26"]
    assert gp.mentions_fiscal_year("revenue by fiscal quarter") and not gp.mentions_fiscal_year("financial performance")


def test_fiscal_bucketing_and_partial_year_labels():
    df = _financial_df()                       # Jan 2021 .. Dec 2024
    plan = gp.plan_chart(df, "revenue by financial year")
    assert plan and plan["intent"] == "fy_overview"
    assert plan["labels"][0].startswith("FY21") and "mo)" in plan["labels"][0]   # Jan-Mar 2021 only
    assert plan["labels"][1] == "FY22"                                            # Apr 2021 - Mar 2022
    assert "mo)" in plan["labels"][-1]                                            # FY25 only Apr-Dec 2024


def test_single_fy_chart_starts_in_april_and_is_scoped():
    plan = gp.plan_chart(_financial_df(), "show revenue trend for FY23")
    assert plan and plan["labels"][0] == "Apr 2022" and plan["labels"][-1] == "Mar 2023"
    assert len(plan["labels"]) == 12
