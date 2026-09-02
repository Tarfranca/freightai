from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "gold" / "ml_dataset.csv"
OUTPUT_PATH = BASE_DIR / "gold" / "ml_ready.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["route_type"]).copy()

    if "target_shipping_cost_usd" not in df.columns and "shipping_cost_usd" in df.columns:
        df = df.rename(columns={"shipping_cost_usd": "target_shipping_cost_usd"})

    drop_columns = ["order_id", "date_text", "mitigation_action", "disruption_event", "shipping_cost_usd"]
    for column in drop_columns:
        if column in df.columns:
            df = df.drop(columns=column)

    categorical_columns = ["transport_mode", "route_type", "product_category", "delivery_status"]
    for column in categorical_columns:
        if column in df.columns:
            df = pd.get_dummies(df, columns=[column], prefix=column, dtype=float)

    for column in df.columns:
        if df[column].isna().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
            else:
                mode_value = df[column].mode()
                fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
                df[column] = df[column].fillna(fill_value)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
