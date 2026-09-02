from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LAKE_DIR = Path(__file__).resolve().parent
DB_PATH = LAKE_DIR / "warehouse" / "warehouse.db"
OUTPUT_PATH = LAKE_DIR / "gold" / "ml_dataset.csv"

SQL_QUERY = """
WITH macro_daily AS (
    SELECT
        date_text,
        AVG(usd_exchange_rate) AS usd_exchange_rate,
        AVG(policy_rate_pct) AS policy_rate_pct,
        AVG(bond_yield_pct) AS bond_yield_pct,
        AVG(yield_spread_pct) AS yield_spread_pct,
        AVG(equity_index_level) AS equity_index_level,
        AVG(oil_price_usd_bbl) AS oil_price_usd_bbl
    FROM fact_macro_finance
    GROUP BY date_text
),
energy_daily AS (
    SELECT
        date_text,
        AVG(price_usd) AS avg_fuel_price_usd,
        AVG(brent_crude_usd) AS avg_brent_crude_usd,
        AVG(tax_percentage) AS avg_tax_percentage
    FROM fact_energy_prices
    GROUP BY date_text
),
market_daily AS (
    SELECT
        date_text,
        AVG(open_price) AS open_price,
        AVG(close_price) AS close_price,
        AVG(high_price) AS high_price,
        AVG(low_price) AS low_price,
        AVG(volume) AS volume,
        AVG(daily_return_pct) AS daily_return_pct,
        AVG(volatility_range) AS volatility_range,
        AVG(vix_close) AS vix_close,
        AVG(economic_news_flag) AS economic_news_flag,
        AVG(sentiment_score) AS sentiment_score,
        AVG(federal_rate_change_flag) AS federal_rate_change_flag,
        AVG(geopolitical_risk_score) AS geopolitical_risk_score,
        AVG(currency_index) AS currency_index
    FROM fact_market
    GROUP BY date_text
)
SELECT
    s.id,
    s.date_text,
    s.order_id,
    s.origin_city,
    s.destination_city,
    s.route_type,
    s.transport_mode,
    s.product_category,
    s.base_lead_time_days,
    s.scheduled_lead_time_days,
    s.actual_lead_time_days,
    s.delay_days,
    s.delivery_status,
    s.disruption_event,
    s.geopolitical_risk_index,
    s.weather_severity_index,
    s.inflation_rate_pct,
    s.shipping_cost_usd AS target_shipping_cost_usd,
    m.usd_exchange_rate,
    m.policy_rate_pct,
    m.bond_yield_pct,
    m.yield_spread_pct,
    m.equity_index_level,
    m.oil_price_usd_bbl,
    e.avg_fuel_price_usd,
    e.avg_brent_crude_usd,
    e.avg_tax_percentage,
    mk.open_price,
    mk.close_price,
    mk.high_price,
    mk.low_price,
    mk.volume,
    mk.daily_return_pct,
    mk.volatility_range,
    mk.vix_close,
    mk.economic_news_flag,
    mk.sentiment_score,
    mk.federal_rate_change_flag,
    mk.geopolitical_risk_score,
    mk.currency_index
FROM fact_supply_chain s
LEFT JOIN macro_daily m
    ON m.date_text = s.date_text
LEFT JOIN energy_daily e
    ON e.date_text = s.date_text
LEFT JOIN market_daily mk
    ON mk.date_text = s.date_text
ORDER BY s.date_text, s.order_id;
"""


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(SQL_QUERY).fetchall()

    if not rows:
        raise RuntimeError(f"No rows found in warehouse database at {DB_PATH}")

    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    print(f"Saved ML dataset with {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
