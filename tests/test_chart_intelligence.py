
import pandas as pd
from services.chart_intelligence import build_data_profile, recommend_chart_plan, parse_natural_customization

class FakeAI:
    def get_chat_response(self, prompt, system_prompt=None):
        return {"response": '{"best_chart_type":"line","x_column":"Date","y_column":"Revenue","group_column":null,"aggregation":"sum","reason":"Time trend","alternatives":["bar","area"]}'}

def test_profile():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-04-01", periods=3, freq="MS"),
        "Revenue": [100, 120, 140],
        "Region": ["N","S","N"]
    })
    assert build_data_profile(df)["rows"] == 3

def test_ai_plan_is_validated():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-04-01", periods=3, freq="MS"),
        "Revenue": [100, 120, 140],
        "Region": ["N","S","N"]
    })
    p = recommend_chart_plan(df, "show revenue trend", FakeAI())
    assert p["best_chart_type"] == "line"
    assert p["x_column"] == "Date"
    assert p["y_column"] == "Revenue"

def test_customization():
    c = parse_natural_customization("make a blue line chart from January to June with a clean style")
    assert c["chart_type"] == "line"
    assert c["palette"] == "Blues"
    assert c["style"] == "white"
    assert c["month_start"] == 1
    assert c["month_end"] == 6
