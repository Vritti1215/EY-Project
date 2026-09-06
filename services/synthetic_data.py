"""
Synthetic Test Data Generator
================================
PENDING item: "Add variance, noise, and realistic imperfections to
synthetic test datasets."

Your table notes the Test_Venue workbook was tested against the full
3,400 rows — but it's clean data. This generates messier datasets to
stress-test the column-detection (schema_service) and validation
(data_profile_service) work: inconsistent date formats, mixed casing,
outliers, missing values, and duplicate rows.

Drop into tests/synthetic_data.py (or services/ if you want to reuse it
outside tests, e.g. for demoing the upload flow).

Usage:
    from tests.synthetic_data import generate_messy_sales_data
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
) -> pd.DataFrame:
    """
    Generate a synthetic sales dataset with realistic imperfections:

    - date column with mixed formats (not all ISO) — stresses
      schema_service._looks_like_datetime()'s pattern+parse thresholds
    - category column with inconsistent casing ('north', 'North', 'NORTH')
    - numeric column with occasional outliers (order-of-magnitude spikes)
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

    Returns:
        pandas.DataFrame
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    regions = ["North", "South", "East", "West", "Central"]
    products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Component Z"]

    base_date = datetime(2025, 1, 1)
    records = []

    for i in range(rows):
        order_date = base_date + timedelta(days=random.randint(0, 364))
        date_str = (
            _random_date_string(order_date)
            if random.random() < inconsistent_date_format_rate
            else order_date.strftime("%Y-%m-%d")
        )

        region = random.choice(regions)
        if random.random() < mixed_case_rate:
            region = CATEGORY_CASE_VARIANTS(region)

        product = random.choice(products)
        base_amount = round(np.random.normal(loc=500, scale=150), 2)
        base_amount = max(base_amount, 5.0)

        if random.random() < outlier_rate:
            base_amount = _inject_outlier(base_amount, magnitude=random.uniform(6, 12))

        quantity = max(1, int(np.random.poisson(lam=4)))

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
