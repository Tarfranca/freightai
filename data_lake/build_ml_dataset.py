from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "warehouse" / "warehouse.db"
GOLD_DIR = BASE_DIR / "gold"
OUTPUT_PATH = GOLD_DIR / "ml_dataset.csv"

RAW_FILES = {
    "usd_brl_daily.csv": "usd_brl_close",
    "brent_daily.csv": "brent_close",
    "gold_daily.csv": "gold_close",
    "dxy_daily.csv": "dxy_close",
    "us_10y_yield.csv": "us_10y_yield_close",
    "bdry_etf.csv": "bdry_close",
    "zim_daily.csv": "zim_close",
    "vix_daily.csv": "vix_close",
}


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def load_raw_feature_files() -> pd.DataFrame:
    frames = []
    for filename, close_col_name in RAW_FILES.items():
        file_path = BASE_DIR / "raw" / next(
            folder for folder in ["macro_finance", "energy", "commodity", "market"] if (BASE_DIR / "raw" / folder / filename).exists()
        ) / filename
        if not file_path.exists():
            continue

        raw = pd.read_csv(file_path)
        raw.columns = [str(c).strip() for c in raw.columns]

        if "Date" not in raw.columns and "date" in raw.columns:
            raw = raw.rename(columns={"date": "Date"})
        if "Close" not in raw.columns and "close" in raw.columns:
            raw = raw.rename(columns={"close": "Close"})
        if {"Date", "Close"}.issubset(raw.columns):
            df = raw[["Date", "Close"]].copy()
            df["date_text"] = normalize_date(df["Date"])
            df = df.dropna(subset=["date_text"])
            df = df[["date_text", "Close"]].rename(columns={"Close": close_col_name})
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid raw feature files were found under data_lake/raw.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date_text", how="outer")

    merged = merged.sort_values("date_text").reset_index(drop=True)
    return merged


COMMODITY_RENAME = {
    "BRL=X": "usd_brl_close",
    "BZ=F": "brent_close",
    "GC=F": "gold_close",
    "^VIX": "vix_close",
    "ZIM": "zim_close",
    "BDRY": "bdry_close",
    "DX-Y.NYB": "dxy_close",
    "HG=F": "copper_close",
    "ZS=F": "soybeans_close",
}

# fact_supply_chain's route_type values don't match corridor_risk_scores.csv's
# corridor names 1:1 — this is the mapping. "Commodity" has no clear corridor
# equivalent, so it's left unmapped (-> NaN -> median-filled downstream by
# preprocess.py) rather than guessed.
ROUTE_TYPE_TO_CORRIDOR = {
    "Atlantic": "North Atlantic",
    "Pacific": "Transpacific",
    "Suez": "Suez/Red Sea",
    "Intra-Asia": "Intra-Asia",
}


def load_warehouse_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = 'fact_supply_chain'",
                conn,
            )
            if tables.empty:
                return pd.DataFrame()

            base = pd.read_sql_query("SELECT * FROM fact_supply_chain", conn)
            if base.empty or "date_text" not in base.columns:
                return pd.DataFrame()

            return base
    except Exception:
        return pd.DataFrame()


def join_commodities(base: pd.DataFrame) -> pd.DataFrame:
    path = GOLD_DIR / "commodities_master.csv"
    if not path.exists():
        return base
    commodities = pd.read_csv(path, parse_dates=["date"])
    available = [col for col in COMMODITY_RENAME if col in commodities.columns]
    selected = commodities[["date"] + available].rename(columns=COMMODITY_RENAME)
    selected["date_text"] = selected["date"].dt.strftime("%Y-%m-%d")
    selected = selected.drop(columns="date")
    return base.merge(selected, on="date_text", how="left")


def join_gpr(base: pd.DataFrame) -> pd.DataFrame:
    path = BASE_DIR / "raw" / "risk" / "gpr_index.csv"
    if not path.exists():
        return base
    gpr = pd.read_csv(path, parse_dates=["date"])
    gpr_monthly = gpr.assign(year=gpr["date"].dt.year, month=gpr["date"].dt.month)[["year", "month", "gpr_global"]]

    base_dates = pd.to_datetime(base["date_text"], errors="coerce")
    base = base.assign(_year=base_dates.dt.year, _month=base_dates.dt.month)
    merged = base.merge(gpr_monthly, left_on=["_year", "_month"], right_on=["year", "month"], how="left")
    return merged.drop(columns=["_year", "_month", "year", "month"], errors="ignore")


def join_oni(base: pd.DataFrame) -> pd.DataFrame:
    path = BASE_DIR / "raw" / "risk" / "oni_enso.csv"
    if not path.exists():
        return base
    oni = pd.read_csv(path)[["year", "month", "oni_value"]]

    base_dates = pd.to_datetime(base["date_text"], errors="coerce")
    base = base.assign(_year=base_dates.dt.year, _month=base_dates.dt.month)
    merged = base.merge(oni, left_on=["_year", "_month"], right_on=["year", "month"], how="left")
    return merged.drop(columns=["_year", "_month", "year", "month"], errors="ignore")


def join_corridor_risk(base: pd.DataFrame) -> pd.DataFrame:
    path = GOLD_DIR / "corridor_risk_scores.csv"
    if not path.exists() or "route_type" not in base.columns:
        return base
    corridor = pd.read_csv(path)[["corridor", "composite_score"]].rename(
        columns={"composite_score": "corridor_risk"}
    )
    base = base.assign(_corridor_name=base["route_type"].map(ROUTE_TYPE_TO_CORRIDOR))
    merged = base.merge(corridor, left_on="_corridor_name", right_on="corridor", how="left")
    return merged.drop(columns=["_corridor_name", "corridor"], errors="ignore")


def build_fallback_dataset() -> pd.DataFrame:
    base = load_raw_feature_files()

    base["id"] = range(1, len(base) + 1)
    base["order_id"] = base["id"].map(lambda x: f"SC-{x:05d}")
    base["origin_city"] = "N/A"
    base["destination_city"] = "N/A"
    base["route_type"] = ["Atlantic" if i % 2 == 0 else "Pacific" for i in range(len(base))]
    base["transport_mode"] = ["Sea" if i % 2 == 0 else "Air" for i in range(len(base))]
    base["product_category"] = ["Consumer Electronics" if i % 2 == 0 else "Pharmaceuticals" for i in range(len(base))]
    base["base_lead_time_days"] = 5 + (base["id"] % 10)
    base["scheduled_lead_time_days"] = base["base_lead_time_days"] + 2
    base["actual_lead_time_days"] = base["scheduled_lead_time_days"] + (base["id"] % 3)
    base["delay_days"] = (base["actual_lead_time_days"] - base["scheduled_lead_time_days"]).clip(lower=0)
    base["delivery_status"] = ["On Time" if i % 3 != 0 else "Late" for i in range(len(base))]
    base["disruption_event"] = "Geopolitical risk"
    base["geopolitical_risk_index"] = 0.5 + (base["id"] % 10) * 0.1
    base["weather_severity_index"] = 1.0 + (base["id"] % 8) * 0.2
    base["inflation_rate_pct"] = 3.0 + (base["id"] % 8) * 0.1
    
    numeric_cols = [
        "usd_brl_close",
        "brent_close",
        "gold_close",
        "dxy_close",
        "us_10y_yield_close",
        "bdry_close",
        "zim_close",
        "vix_close",
    ]
    for col in numeric_cols:
        if col not in base.columns:
            base[col] = 0.0

    base["shipping_cost_usd"] = (
        2500
        + 120 * base["usd_brl_close"]
        + 150 * base["brent_close"] / 100
        + 3 * base["gold_close"] / 100
        + 100 * base["dxy_close"] / 100
        + 1500 * base["us_10y_yield_close"] / 10
        + 25 * base["bdry_close"]
        + 10 * base["zim_close"]
        + 40 * base["vix_close"] / 100
    )
    base["target_shipping_cost_usd"] = base["shipping_cost_usd"]

    base["date_text"] = base["date_text"].astype(str)
    return base


def build_ml_dataset() -> pd.DataFrame:
    warehouse_df = load_warehouse_data()
    if warehouse_df.empty or "date_text" not in warehouse_df.columns:
        return build_fallback_dataset()

    df = join_commodities(warehouse_df)
    df = join_gpr(df)
    df = join_oni(df)
    df = join_corridor_risk(df)
    return df


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df = build_ml_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved ML dataset to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
