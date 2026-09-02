from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAKE = Path(__file__).resolve().parent
RAW = LAKE / "raw"
BRONZE = LAKE / "bronze"
SILVER = LAKE / "silver"
GOLD = LAKE / "gold"
META = LAKE / "metadata"

DOMAIN_RULES = [
    ("energy", ["brent", "oil", "fuel", "petrol", "diesel", "lpg", "crude", "gasoline"]),
    ("trade", ["export", "importer", "exporter", "trade", "africa", "cpc", "hs_code"]),
    ("ports", ["port", "harbor", "shipping", "pub150", "locode", "sea_area"]),
    ("supply_chain", ["supply_chain", "shipment", "route", "logistics", "lead_time", "distance_km", "transport_mode"]),
    ("macro_finance", ["macro", "financial", "yield", "equity", "vix", "exchange_rate", "bond"]),
    ("market", ["market", "trend", "return", "volatility", "daily_return", "close_price", "open_price"]),
    ("commodity", ["commodity", "price"]),
]

CLASSIFICATION_OVERRIDES = {
    "BrentOilPrices (1).csv": "energy",
    "commodity_prices_supply_chain.csv": "commodity",
    "Daily_Port_Activity_Data_and_Trade_Estimates (1).csv": "ports",
    "Dolfut.csv": "market",
    "em_macro_financial_daily_2020_2025.csv": "macro_finance",
    "global_fuel_prices_2020_2026.csv": "energy",
    "global_supply_chain_disruption_v1.csv": "supply_chain",
    "Import Export Trade Data Africa.csv": "trade",
    "Importante global_supply_chain_risk_2026.csv": "supply_chain",
    "Market_Trend_External - Indice com geopolitical risk.csv": "market",
    "UpdatedPub150 (1).csv": "ports",
}


def normalize_filename(name: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*\)", "", name)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned + ".csv"


def infer_domain(name: str, columns: list[str] | None = None) -> str:
    if name in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[name]

    lowered_name = name.lower()
    lowered_columns = " ".join((columns or [])).lower()

    if any(token in lowered_columns for token in ["portid", "portname", "un/locode", "world port index", "main port name"]):
        return "ports"
    if any(token in lowered_columns for token in ["shipment_id", "distance_km", "lead_time", "transport_mode", "route_type"]):
        return "supply_chain"
    if any(token in lowered_columns for token in ["exporter", "importer", "hs code", "country of origin", "destination country"]):
        return "trade"
    if any(token in lowered_columns for token in ["usd_exchangerate", "policy_rate", "bond_yield", "equity_index_level", "vix_close"]):
        return "macro_finance"
    if any(token in lowered_columns for token in ["daily_return", "volatility_range", "open_price", "close_price", "vix_close"]):
        return "market"

    for domain, keywords in DOMAIN_RULES:
        if any(keyword in lowered_name for keyword in keywords):
            return domain

    if any(token in lowered_name for token in ["importante", "risk"]):
        return "supply_chain"

    return "general"


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def read_columns(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def build_catalog() -> list[dict]:
    catalog = []
    for csv_file in sorted(ROOT.glob("*.csv")):
        columns = read_columns(csv_file)
        domain = infer_domain(csv_file.name, columns)
        catalog.append({
            "source_file": csv_file.name,
            "domain": domain,
            "normalized_file": normalize_filename(csv_file.name),
            "column_count": len(columns),
            "columns": "; ".join(columns[:8]) + ("..." if len(columns) > 8 else ""),
        })
    return catalog


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    BRONZE.mkdir(parents=True, exist_ok=True)
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(ROOT.glob("*.csv")):
        columns = read_columns(csv_file)
        domain = infer_domain(csv_file.name, columns)
        raw_dest = RAW / domain / csv_file.name
        bronze_dest = BRONZE / domain / normalize_filename(csv_file.name)
        copy_file(csv_file, raw_dest)
        copy_file(csv_file, bronze_dest)

    catalog = build_catalog()
    catalog_path = META / "dataset_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_file", "domain", "normalized_file", "column_count", "columns"])
        writer.writeheader()
        writer.writerows(catalog)

    for domain in sorted({row["domain"] for row in catalog}):
        (GOLD / domain).mkdir(parents=True, exist_ok=True)
        (SILVER / domain).mkdir(parents=True, exist_ok=True)

    print(f"Created data lake layout under: {LAKE}")
    print(f"Catalog entries: {len(catalog)}")


if __name__ == "__main__":
    main()
