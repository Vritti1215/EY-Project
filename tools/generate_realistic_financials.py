"""
Generate a realistic company financial workbook for testing the EY portal.

Includes trend, seasonality, noise, correlated costs, occasional shocks,
multiple years, regions/products, and a small amount of missing data.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def generate_financial_data(start="2023-04-01", periods=36, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    regions = {"North": .29, "South": .24, "East": .22, "West": .25}
    products = {"Enterprise": .48, "Consumer": .31, "SMB": .21}
    rows = []

    for i, date in enumerate(dates):
        trend = 1 + .006 * i
        seasonal = 1 + .08 * np.sin(2 * np.pi * (date.month - 1) / 12)
        shock = rng.normal(0, .035)
        if rng.random() < .055:
            shock += rng.choice([-1, 1]) * rng.uniform(.08, .18)

        company_revenue = 42_000_000 * trend * seasonal * (1 + shock)
        cogs_ratio = np.clip(.56 + rng.normal(0, .018), .50, .64)
        opex_ratio = np.clip(.22 + rng.normal(0, .012), .18, .27)
        company_cogs = company_revenue * cogs_ratio
        company_opex = company_revenue * opex_ratio

        for region, region_share in regions.items():
            for product, product_share in products.items():
                interaction = rng.normal(1.0, .045)
                revenue = company_revenue * region_share * product_share * interaction
                cogs = company_cogs * region_share * product_share * interaction
                opex = company_opex * region_share * product_share * interaction
                gross = revenue - cogs
                operating = gross - opex

                rows.append({
                    "Date": date,
                    "Year": date.year,
                    "Month": date.strftime("%b"),
                    "Fiscal Quarter": f"Q{((date.month - 4) % 12) // 3 + 1}",
                    "Region": region,
                    "Product": product,
                    "Revenue": round(revenue, 2),
                    "COGS": round(cogs, 2),
                    "Operating Expense": round(opex, 2),
                    "Gross Profit": round(gross, 2),
                    "Operating Profit": round(operating, 2),
                    "Units Sold": max(1, int(revenue / rng.uniform(1300, 1900))),
                    "Customer Satisfaction": round(float(np.clip(rng.normal(4.05, .32), 2.5, 5.0)), 2),
                })

    df = pd.DataFrame(rows)
    mask = rng.random(len(df)) < .012
    df.loc[mask, "Customer Satisfaction"] = np.nan
    return df


def save_excel(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_financial_data()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        summary = (
            df.groupby("Date", as_index=False)[
                ["Revenue", "COGS", "Operating Expense", "Gross Profit", "Operating Profit"]
            ].sum()
        )
        summary.to_excel(writer, index=False, sheet_name="Monthly Summary")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/Realistic_Company_Financials.xlsx")
    args = parser.parse_args()
    print(f"Created {save_excel(args.output)}")
