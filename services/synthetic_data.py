"""
Synthetic Test Data Generator
================================
DONE item: "Add variance, noise, and realistic imperfections to
synthetic test datasets."

Previously this only added random per-row noise (a flat Gaussian around
one mean) plus formatting messiness (mixed date formats/casing, missing
values, duplicates). That's enough to stress-test column detection, but
it doesn't look like real business data, which has *structure* in its
variance: seasonality, a growth/decline trend, weekday effects, regional
and product-level differences, occasional refunds, and outliers that are
plausible one-off events rather than pure noise. This version adds that
layer while keeping all the original messiness.

Usage:
    from services.synthetic_data import generate_messy_sales_data
    df = generate_messy_sales_data(rows=500, seed=42)
    df.to_excel('messy_sales_test.xlsx', index=False)

Extend test_chatbot_file_upload.py with a case using this, e.g.:

    def test_chat_handles_messy_synthetic_data(self):
        df = generate_messy_sales_data(rows=200, seed=1)
        ...write df to a temp .xlsx, upload via /chat/send as today...
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%b %Y", "%B %d, %Y", "%d-%b-%Y",
]

CATEGORY_CASE_VARIANTS = lambda s: random.choice([s.upper(), s.lower(), s.title(), s])

# --- Realistic business-pattern parameters -----------------------------
# Monthly seasonality multiplier (Jan..Dec): slow start, summer dip,
# a Nov/Dec holiday surge — a common retail/B2B pattern.
MONTH_SEASONALITY = [0.85, 0.88, 0.95, 1.00, 1.02, 0.90,
                      0.85, 0.92, 1.00, 1.08, 1.20, 1.35]

# Weekday multiplier (Mon..Sun) — weekends noticeably quieter for a
# B2B-style sales dataset.
WEEKDAY_SEASONALITY = [1.05, 1.08, 1.10, 1.05, 1.00, 0.55, 0.45]

# Region-level baseline differences (some regions just run bigger orders).
REGION_BASELINE = {
    "North": 1.15, "South": 0.90, "East": 1.00, "West": 1.25, "Central": 0.80,
}

# Product-level baseline differences.
PRODUCT_BASELINE = {
    "Widget A": 0.70, "Widget B": 0.85, "Gadget X": 1.20,
    "Gadget Y": 1.05, "Component Z": 1.60,
}


def _random_date_string(base_date: datetime) -> str:
    fmt = random.choice(DATE_FORMATS)
    return base_date.strftime(fmt)


def _inject_outlier(value: float, magnitude: float = 8.0) -> float:
    return round(value * magnitude, 2)


def generate_messy_sales_data(
    rows: int = 500,
    seed: int = None,
    outlier_rate: float = 0.02,
    missing_rate: float = 0.03,
    duplicate_rate: float = 0.02,
    inconsistent_date_format_rate: float = 0.4,
    mixed_case_rate: float = 0.5,
    refund_rate: float = 0.03,
    annual_growth: float = 0.18,
) -> pd.DataFrame:
    """
    Generate a synthetic sales dataset with realistic imperfections AND
    realistic business patterns:

    Business patterns (the "signal" a real dataset would have):
    - annual growth trend: later orders trend ~`annual_growth` higher
      than earlier ones (default +18% end-of-year vs start).
    - monthly seasonality: a Nov/Dec holiday surge and a summer dip
      (MONTH_SEASONALITY), not a flat average all year.
    - weekday effect: weekend orders are smaller/rarer (WEEKDAY_SEASONALITY).
    - region- and product-level baselines: e.g. "West" and "Component Z"
      run bigger tickets than "Central"/"Widget A" — not every category
      centered on the same mean.
    - occasional refunds/returns: a small fraction of rows are negative
      Amount values, as real transaction exports usually have.

    Messiness (imperfections that stress column-detection/validation):
    - date column with mixed formats (not all ISO)
    - category column with inconsistent casing ('north', 'North', 'NORTH')
    - numeric column with occasional outliers (plausible one-off bulk
      orders layered on top of the business pattern above, not pure
      random noise floating free of it)
    - random missing values (NaN) sprinkled across columns
    - a small percentage of exact duplicate rows
    - an identifier column (Order ID) to test is_id_like() exclusion logic

    Args:
        rows: number of base rows to generate before duplicates are added.
        seed: random seed for reproducibility (None = nondeterministic).
        outlier_rate: fraction of numeric values replaced with spikes.
        missing_rate: fraction of cells (across value columns) set to NaN.
        duplicate_rate: fraction of extra duplicate rows appended.
        inconsistent_date_format_rate: fraction of dates using a
            non-ISO format.
        mixed_case_rate: fraction of category values with randomized case.
        refund_rate: fraction of rows turned into negative-amount refunds.
        annual_growth: fractional growth from Jan 1 to Dec 31 (a linear
            trend layered under the seasonality/noise).

    Returns:
        pandas.DataFrame
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    regions = list(REGION_BASELINE.keys())
    products = list(PRODUCT_BASELINE.keys())

    base_date = datetime(2025, 1, 1)
    records = []

    for i in range(rows):
        day_offset = random.randint(0, 364)
        order_date = base_date + timedelta(days=day_offset)
        date_str = (
            _random_date_string(order_date)
            if random.random() < inconsistent_date_format_rate
            else order_date.strftime("%Y-%m-%d")
        )

        region = random.choice(regions)
        region_factor = REGION_BASELINE[region]
        if random.random() < mixed_case_rate:
            region = CATEGORY_CASE_VARIANTS(region)

        product = random.choice(products)
        product_factor = PRODUCT_BASELINE[product]

        # Business pattern: trend + seasonality + weekday effect, all
        # multiplied together before the per-order random noise is added
        # — this is what makes the series look like a real business
        # rather than samples drawn from one flat distribution.
        trend_factor = 1.0 + annual_growth * (day_offset / 364.0)
        month_factor = MONTH_SEASONALITY[order_date.month - 1]
        weekday_factor = WEEKDAY_SEASONALITY[order_date.weekday()]

        combined_factor = region_factor * product_factor * trend_factor * month_factor * weekday_factor
        base_amount = round(np.random.normal(loc=500, scale=90) * combined_factor, 2)
        base_amount = max(base_amount, 5.0)

        if random.random() < outlier_rate:
            # A plausible one-off bulk order/spike, layered on the
            # already-patterned amount rather than replacing it outright.
            base_amount = _inject_outlier(base_amount, magnitude=random.uniform(6, 12))
        elif random.random() < refund_rate:
            # A refund/return: a real transaction export usually has some.
            base_amount = -round(base_amount * random.uniform(0.2, 1.0), 2)

        quantity = max(1, int(np.random.poisson(lam=4 * region_factor)))

        records.append({
            "Order ID": f"ORD-{100000 + i}",
            "Order Date": date_str,
            "Region": region,
            "Product": product,
            "Quantity": quantity,
            "Amount": base_amount,
        })

    df = pd.DataFrame(records)

    # Missing values sprinkled across non-identifier columns.
    for col in ["Order Date", "Region", "Product", "Quantity", "Amount"]:
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    # Exact duplicate rows.
    n_dupes = int(len(df) * duplicate_rate)
    if n_dupes > 0:
        dupes = df.sample(n=n_dupes, replace=True, random_state=seed)
        df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle so duplicates/missing rows aren't clustered at the end.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_messy_student_data(rows: int = 300, seed: int = None) -> pd.DataFrame:
    """
    A second domain (non-financial) synthetic set — mirrors the
    marksheet/venue example schema_service.py's docstring calls out, to
    confirm column detection generalizes beyond sales data.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    campuses = ["Delhi", "Mumbai", "Pune", "Bengaluru", "Chennai"]
    time_slots = ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"]

    records = []
    for i in range(rows):
        campus = random.choice(campuses)
        if random.random() < 0.4:
            campus = CATEGORY_CASE_VARIANTS(campus)

        marks = round(np.random.normal(loc=70, scale=15), 1)
        marks = min(max(marks, 0), 100)
        if random.random() < 0.02:
            marks = round(random.uniform(150, 200), 1)  # impossible-score outlier

        records.append({
            "Roll No": f"R{2000 + i}",
            "Campus": campus,
            "Test Time (hh:mm AM/PM)": random.choice(time_slots),
            "Marks": marks,
        })

    df = pd.DataFrame(records)
    mask = np.random.random(len(df)) < 0.03
    df.loc[mask, "Marks"] = np.nan
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)