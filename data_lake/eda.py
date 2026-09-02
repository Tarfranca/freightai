from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "gold" / "ml_dataset.csv"
OUTPUT_PATH = BASE_DIR / "eda_summary.txt"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> None:
    df = load_dataset()

    print_section("1. Columns and data types")
    print(df.dtypes.to_string())

    print_section("2. Null values per column")
    null_counts = df.isnull().sum()
    print(null_counts.to_string())

    target = "target_shipping_cost_usd"
    print_section("3. Basic statistics for target_shipping_cost_usd")
    stats = df[target].describe()
    print(stats)
    print(f"min: {df[target].min()}")
    print(f"max: {df[target].max()}")
    print(f"mean: {df[target].mean()}")
    print(f"median: {df[target].median()}")
    print(f"std: {df[target].std()}")

    print_section("4. Distribution of transport_mode and route_type")
    print("transport_mode distribution:")
    print(df["transport_mode"].value_counts(dropna=False).to_string())
    print("\nroute_type distribution:")
    print(df["route_type"].value_counts(dropna=False).to_string())

    print_section("5. Correlation between numeric columns and the target")
    numeric_df = df.select_dtypes(include=["number"])
    correlations = numeric_df.corr(numeric_only=True)[target].sort_values(ascending=False)
    print(correlations.to_string())

    # Save a text summary to disk too.
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("1. Columns and data types\n")
        handle.write(df.dtypes.to_string() + "\n\n")
        handle.write("2. Null values per column\n")
        handle.write(null_counts.to_string() + "\n\n")
        handle.write("3. Basic statistics for target_shipping_cost_usd\n")
        handle.write(stats.to_string() + "\n")
        handle.write(f"min: {df[target].min()}\n")
        handle.write(f"max: {df[target].max()}\n")
        handle.write(f"mean: {df[target].mean()}\n")
        handle.write(f"median: {df[target].median()}\n")
        handle.write(f"std: {df[target].std()}\n\n")
        handle.write("4. Distribution of transport_mode and route_type\n")
        handle.write("transport_mode distribution:\n")
        handle.write(df["transport_mode"].value_counts(dropna=False).to_string() + "\n\n")
        handle.write("route_type distribution:\n")
        handle.write(df["route_type"].value_counts(dropna=False).to_string() + "\n\n")
        handle.write("5. Correlation between numeric columns and the target\n")
        handle.write(correlations.to_string())

    print(f"\nSaved summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
