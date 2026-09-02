from __future__ import annotations

import csv
import re
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAKE = Path(__file__).resolve().parent
RAW = LAKE / "raw"
SILVER = LAKE / "silver"
WAREHOUSE = LAKE / "warehouse"
DB_PATH = WAREHOUSE / "warehouse.db"
SCHEMA_PATH = LAKE / "warehouse_schema.sql"

COLUMN_MAPS = {
    "commodity_prices_supply_chain.csv": {
        "date": "date_text",
        "year": "year",
        "month": "month",
        "commodity": "commodity_name",
        "category": "category",
        "unit": "unit",
        "currency": "currency",
        "price": "price_usd",
    },
    "global_fuel_prices_2020_2026.csv": {
        "date": "date_text",
        "country": "country_name",
        "region": "region_name",
        "income_level": "income_level",
        "subsidy_level": "subsidy_level",
        "petrol_usd_liter": "petrol_price_usd_liter",
        "diesel_usd_liter": "diesel_price_usd_liter",
        "lpg_usd_liter": "lpg_price_usd_liter",
        "brent_crude_usd": "brent_crude_usd",
        "tax_percentage": "tax_percentage",
    },
    "BrentOilPrices (1).csv": {
        "Date": "date_text",
        "Price": "price_usd",
    },
    "em_macro_financial_daily_2020_2025.csv": {
        "Date": "date_text",
        "Country": "country_name",
        "USD_ExchangeRate": "usd_exchange_rate",
        "Policy_Rate(%)": "policy_rate_pct",
        "10Y_Bond_Yield(%)": "bond_yield_pct",
        "Yield_Spread(10Y_vs_US)(%)": "yield_spread_pct",
        "Equity_Index_Level": "equity_index_level",
        "Oil_Price(USD_per_bbl)": "oil_price_usd_bbl",
    },
    "Daily_Port_Activity_Data_and_Trade_Estimates (1).csv": {
        "date": "date_text",
        "year": "year",
        "month": "month",
        "day": "day",
        "portid": "port_id",
        "portname": "port_name",
        "country": "country_name",
        "ISO3": "iso3",
        "portcalls_container": "portcalls_container",
        "portcalls_dry_bulk": "portcalls_dry_bulk",
        "portcalls_general_cargo": "portcalls_general_cargo",
        "portcalls_roro": "portcalls_roro",
        "portcalls_tanker": "portcalls_tanker",
        "portcalls": "portcalls_total",
        "import_container": "import_container",
        "import_dry_bulk": "import_dry_bulk",
        "import_general_cargo": "import_general_cargo",
        "import_roro": "import_roro",
        "import_tanker": "import_tanker",
    },
    "Import Export Trade Data Africa.csv": {
        "DATE": "date_text",
        "TYPE": "trade_type",
        "CPC DESCRIPTION": "cpc_description",
        "EXPORTER NAME": "exporter_name",
        "IMPORTER NAME": "importer_name",
        "DECLARANT NAME": "declarant_name",
        "Origin Country": "origin_country",
        "Destination Country": "destination_country",
        "HS CODE": "hs_code",
        "HS CODE DESCRIPTION": "hs_description",
        "QUANTITY UOM": "quantity_uom",
        "QUANTITY": "quantity",
        "NO OF PACKAGE TYPE": "no_of_package_type",
        "GROSS WEIGHT": "gross_weight",
        "GROSS WEIGHT UOM": "gross_weight_uom",
        "NET WEIGHT": "net_weight",
        "NET WEIGHT UOM": "net_weight_uom",
        "PACKAGE TYPE": "package_type",
        "CUSTOMS VALUE BWP": "customs_value_bwp",
        "DECLARATION OFFICE": "declaration_office",
    },
    "global_supply_chain_disruption_v1.csv": {
        "Order_ID": "order_id",
        "Order_Date": "date_text",
        "Origin_City": "origin_city",
        "Destination_City": "destination_city",
        "Route_Type": "route_type",
        "Transportation_Mode": "transport_mode",
        "Product_Category": "product_category",
        "Base_Lead_Time_Days": "base_lead_time_days",
        "Scheduled_Lead_Time_Days": "scheduled_lead_time_days",
        "Actual_Lead_Time_Days": "actual_lead_time_days",
        "Delay_Days": "delay_days",
        "Delivery_Status": "delivery_status",
        "Disruption_Event": "disruption_event",
        "Geopolitical_Risk_Index": "geopolitical_risk_index",
        "Weather_Severity_Index": "weather_severity_index",
        "Inflation_Rate_Pct": "inflation_rate_pct",
        "Shipping_Cost_USD": "shipping_cost_usd",
        "Order_Weight_Kg": "order_weight_kg",
        "Mitigation_Action_Taken": "mitigation_action",
    },
    "Importante global_supply_chain_risk_2026.csv": {
        "Shipment_ID": "shipment_id",
        "Date": "date_text",
        "Origin_Port": "origin_port",
        "Destination_Port": "destination_port",
        "Transport_Mode": "transport_mode",
        "Product_Category": "product_category",
        "Distance_km": "distance_km",
        "Weight_MT": "weight_mt",
        "Fuel_Price_Index": "fuel_price_index",
        "Geopolitical_Risk_Score": "risk_score",
    },
    "Market_Trend_External - Indice com geopolitical risk.csv": {
        "Date": "date_text",
        "Open_Price": "open_price",
        "Close_Price": "close_price",
        "High_Price": "high_price",
        "Low_Price": "low_price",
        "Volume": "volume",
        "Daily_Return_Pct": "daily_return_pct",
        "Volatility_Range": "volatility_range",
        "VIX_Close": "vix_close",
        "Economic_News_Flag": "economic_news_flag",
        "Sentiment_Score": "sentiment_score",
        "Federal_Rate_Change_Flag": "federal_rate_change_flag",
        "GeoPolitical_Risk_Score": "geopolitical_risk_score",
        "Currency_Index": "currency_index",
    },
    "Dolfut.csv": {
        "": "row_number",
        "Date": "date_text",
        "Close": "close_price",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Volume": "volume",
        "Returns": "daily_return_pct",
    },
    "UpdatedPub150 (1).csv": {
        "OID_": "port_id",
        "World Port Index Number": "world_port_index_number",
        "Region Name": "region_name",
        "Main Port Name": "port_name",
        "Alternate Port Name": "alternate_port_name",
        "UN/LOCODE": "un_locode",
        "Country Code": "country_code",
        "World Water Body": "world_water_body",
        "IHO S-130 Sea Area": "sea_area",
        "Sailing Direction or Publication": "sailing_direction",
        "Publication Link": "publication_link",
        "Standard Nautical Chart": "standard_nautical_chart",
        "IHO S-57 Electronic Navigational Chart": "iho_s57",
        "IHO S-101 Electronic Navigational Chart": "iho_s101",
        "Digital Nautical Chart": "digital_nautical_chart",
        "Tidal Range (m)": "tidal_range_m",
        "Entrance Width (m)": "entrance_width_m",
        "Channel Depth (m)": "channel_depth_m",
        "Anchorage Depth (m)": "anchorage_depth_m",
        "Cargo Pier Depth (m)": "cargo_pier_depth_m",
    },
}


def snake_case(value: str) -> str:
    value = re.sub(r'(?<!^)(?=[A-Z])', '_', value)
    value = re.sub(r'[^0-9A-Za-z]+', '_', value)
    value = re.sub(r'_+', '_', value).strip('_').lower()
    return value


def clean_value(value):
    if value is None:
        return None
    value = str(value).strip()
    if value in {'', 'nan', 'NaN', 'None', 'null'}:
        return None
    return value


def to_float(value):
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned.replace(',', ''))
    except ValueError:
        return None


def to_int(value):
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_date(value):
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d %H:%M:%S%z', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(cleaned, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(cleaned.replace('Z', '+00:00')).strftime('%Y-%m-%d')
    except ValueError:
        return None


def normalize_filename(name: str) -> str:
    cleaned = re.sub(r'\s*\([^)]*\)', '', name)
    cleaned = cleaned.strip().lower()
    cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned + '.csv'


def standardize_source_file(src_path: Path, dest_path: Path, mapping: dict) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open('r', encoding='utf-8-sig', newline='') as fin, dest_path.open('w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            return
        output_fields = []
        for col in reader.fieldnames:
            if col is None:
                continue
            base = mapping.get(col, snake_case(col))
            output_fields.append(base)
        writer = csv.DictWriter(fout, fieldnames=output_fields)
        writer.writeheader()
        for row in reader:
            out_row = {}
            for key, value in row.items():
                if key is None:
                    continue
                target = mapping.get(key, snake_case(key))
                if key == '':
                    target = 'row_number'
                if target == 'date_text':
                    out_row[target] = parse_date(value)
                elif target in {'year', 'month', 'day'}:
                    out_row[target] = to_int(value)
                elif target.endswith(('price', 'rate', 'yield', 'index', 'score', 'weight', 'cost', 'value')) or target in {'volume', 'quantity', 'gross_weight', 'net_weight', 'customs_value_bwp', 'shipping_cost_usd', 'tax_percentage'}:
                    out_row[target] = to_float(value)
                elif target in {'portcalls_container', 'portcalls_dry_bulk', 'portcalls_general_cargo', 'portcalls_roro', 'portcalls_tanker', 'portcalls_total'}:
                    out_row[target] = to_int(value)
                else:
                    out_row[target] = clean_value(value)
            writer.writerow(out_row)


def build_silver_layer() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)
    for src in sorted(RAW.rglob('*.csv')):
        domain = src.relative_to(RAW).parts[0]
        dest = SILVER / domain / normalize_filename(src.name)
        mapping = COLUMN_MAPS.get(src.name, {})
        standardize_source_file(src, dest, mapping)


def ensure_db() -> sqlite3.Connection:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with SCHEMA_PATH.open('r', encoding='utf-8') as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def insert_dim_country(conn):
    rows = []
    for csv_path in [SILVER / 'energy' / 'global_fuel_prices_2020_2026_csv.csv', SILVER / 'macro_finance' / 'em_macro_financial_daily_2020_2025_csv.csv']:
        if csv_path.exists():
            with csv_path.open('r', encoding='utf-8', newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    country = clean_value(row.get('country_name'))
                    if country:
                        rows.append((country, clean_value(row.get('region_name')), clean_value(row.get('income_level'))))
    for country, region, income in sorted(set(rows), key=lambda x: (x[0] or '', x[1] or '', x[2] or '')):
        conn.execute(
            'INSERT OR IGNORE INTO dim_country(country_name, region_name, income_level) VALUES (?, ?, ?)',
            (country, region, income),
        )
    conn.commit()


def insert_dim_commodity(conn):
    commodity_path = SILVER / 'commodity' / 'commodity_prices_supply_chain_csv.csv'
    if commodity_path.exists():
        with commodity_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            seen = set()
            for row in reader:
                name = clean_value(row.get('commodity_name'))
                if name and name not in seen:
                    seen.add(name)
                    conn.execute(
                        'INSERT OR IGNORE INTO dim_commodity(commodity_name, category, unit, currency) VALUES (?, ?, ?, ?)',
                        (name, clean_value(row.get('category')), clean_value(row.get('unit')), clean_value(row.get('currency'))),
                    )
    conn.commit()


def insert_dim_port(conn):
    for csv_path in [SILVER / 'ports' / 'daily_port_activity_data_and_trade_estimates_csv.csv', SILVER / 'ports' / 'updatedpub150_csv.csv']:
        if not csv_path.exists():
            continue
        with csv_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                port_id = clean_value(row.get('port_id')) or clean_value(row.get('world_port_index_number')) or clean_value(row.get('port_name'))
                port_name = clean_value(row.get('port_name')) or clean_value(row.get('main_port_name'))
                country = clean_value(row.get('country_name')) or clean_value(row.get('country_code'))
                iso3 = clean_value(row.get('iso3'))
                region = clean_value(row.get('region_name'))
                water_body = clean_value(row.get('world_water_body'))
                if port_id and port_name:
                    conn.execute(
                        'INSERT OR IGNORE INTO dim_port(port_id, port_name, country_name, iso3, region_name, world_water_body) VALUES (?, ?, ?, ?, ?, ?)',
                        (str(port_id), port_name, country, iso3, region, water_body),
                    )
    conn.commit()


def insert_fact_commodity_prices(conn):
    path = SILVER / 'commodity' / 'commodity_prices_supply_chain_csv.csv'
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            conn.execute(
                'INSERT INTO fact_commodity_prices(date_text, commodity_name, category, unit, currency, price_usd) VALUES (?, ?, ?, ?, ?, ?)',
                (
                    clean_value(row.get('date_text')),
                    clean_value(row.get('commodity_name')),
                    clean_value(row.get('category')),
                    clean_value(row.get('unit')),
                    clean_value(row.get('currency')),
                    to_float(row.get('price_usd')),
                ),
            )
    conn.commit()


def insert_fact_energy_prices(conn):
    fuel_path = SILVER / 'energy' / 'global_fuel_prices_2020_2026_csv.csv'
    if fuel_path.exists():
        with fuel_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_energy_prices(date_text, country_name, region_name, fuel_type, price_usd, brent_crude_usd, tax_percentage) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        clean_value(row.get('country_name')),
                        clean_value(row.get('region_name')),
                        'petrol',
                        to_float(row.get('petrol_price_usd_liter')),
                        to_float(row.get('brent_crude_usd')),
                        to_float(row.get('tax_percentage')),
                    )
                )
                conn.execute(
                    'INSERT INTO fact_energy_prices(date_text, country_name, region_name, fuel_type, price_usd, brent_crude_usd, tax_percentage) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        clean_value(row.get('country_name')),
                        clean_value(row.get('region_name')),
                        'diesel',
                        to_float(row.get('diesel_price_usd_liter')),
                        to_float(row.get('brent_crude_usd')),
                        to_float(row.get('tax_percentage')),
                    )
                )
                conn.execute(
                    'INSERT INTO fact_energy_prices(date_text, country_name, region_name, fuel_type, price_usd, brent_crude_usd, tax_percentage) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        clean_value(row.get('country_name')),
                        clean_value(row.get('region_name')),
                        'lpg',
                        to_float(row.get('lpg_price_usd_liter')),
                        to_float(row.get('brent_crude_usd')),
                        to_float(row.get('tax_percentage')),
                    )
                )
    brent_path = SILVER / 'energy' / 'brentoilprices_csv.csv'
    if brent_path.exists():
        with brent_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_energy_prices(date_text, country_name, region_name, fuel_type, price_usd, brent_crude_usd, tax_percentage) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        'Global',
                        'Global',
                        'brent_oil',
                        to_float(row.get('price_usd')),
                        to_float(row.get('price_usd')),
                        None,
                    )
                )
    conn.commit()


def insert_fact_macro_finance(conn):
    path = SILVER / 'macro_finance' / 'em_macro_financial_daily_2020_2025_csv.csv'
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            conn.execute(
                'INSERT INTO fact_macro_finance(date_text, country_name, usd_exchange_rate, policy_rate_pct, bond_yield_pct, yield_spread_pct, equity_index_level, oil_price_usd_bbl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    clean_value(row.get('date_text')),
                    clean_value(row.get('country_name')),
                    to_float(row.get('usd_exchange_rate')),
                    to_float(row.get('policy_rate_pct')),
                    to_float(row.get('bond_yield_pct')),
                    to_float(row.get('yield_spread_pct')),
                    to_float(row.get('equity_index_level')),
                    to_float(row.get('oil_price_usd_bbl')),
                )
            )
    conn.commit()


def insert_fact_port_activity(conn):
    path = SILVER / 'ports' / 'daily_port_activity_data_and_trade_estimates_csv.csv'
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            conn.execute(
                'INSERT INTO fact_port_activity(date_text, port_id, port_name, country_name, iso3, portcalls_container, portcalls_dry_bulk, portcalls_general_cargo, portcalls_roro, portcalls_tanker, portcalls_total, import_container, import_dry_bulk, import_general_cargo, import_roro, import_tanker) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    clean_value(row.get('date_text')),
                    clean_value(row.get('port_id')),
                    clean_value(row.get('port_name')),
                    clean_value(row.get('country_name')),
                    clean_value(row.get('iso3')),
                    to_int(row.get('portcalls_container')),
                    to_int(row.get('portcalls_dry_bulk')),
                    to_int(row.get('portcalls_general_cargo')),
                    to_int(row.get('portcalls_roro')),
                    to_int(row.get('portcalls_tanker')),
                    to_int(row.get('portcalls_total')),
                    to_float(row.get('import_container')),
                    to_float(row.get('import_dry_bulk')),
                    to_float(row.get('import_general_cargo')),
                    to_float(row.get('import_roro')),
                    to_float(row.get('import_tanker')),
                )
            )
    conn.commit()


def insert_fact_trade(conn):
    path = SILVER / 'trade' / 'import_export_trade_data_africa_csv.csv'
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            conn.execute(
                'INSERT INTO fact_trade(date_text, trade_type, exporter_name, importer_name, origin_country, destination_country, hs_code, hs_description, quantity, quantity_uom, gross_weight, gross_weight_uom, customs_value_bwp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    clean_value(row.get('date_text')),
                    clean_value(row.get('trade_type')),
                    clean_value(row.get('exporter_name')),
                    clean_value(row.get('importer_name')),
                    clean_value(row.get('origin_country')),
                    clean_value(row.get('destination_country')),
                    clean_value(row.get('hs_code')),
                    clean_value(row.get('hs_description')),
                    to_float(row.get('quantity')),
                    clean_value(row.get('quantity_uom')),
                    to_float(row.get('gross_weight')),
                    clean_value(row.get('gross_weight_uom')),
                    to_float(row.get('customs_value_bwp')),
                )
            )
    conn.commit()


def insert_fact_supply_chain(conn):
    path = SILVER / 'supply_chain' / 'global_supply_chain_disruption_v1_csv.csv'
    if path.exists():
        with path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_supply_chain(date_text, order_id, origin_city, destination_city, route_type, transport_mode, product_category, base_lead_time_days, scheduled_lead_time_days, actual_lead_time_days, delay_days, delivery_status, disruption_event, geopolitical_risk_index, weather_severity_index, inflation_rate_pct, shipping_cost_usd, order_weight_kg, mitigation_action, risk_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        clean_value(row.get('order_id')),
                        clean_value(row.get('origin_city')),
                        clean_value(row.get('destination_city')),
                        clean_value(row.get('route_type')),
                        clean_value(row.get('transport_mode')),
                        clean_value(row.get('product_category')),
                        to_int(row.get('base_lead_time_days')),
                        to_int(row.get('scheduled_lead_time_days')),
                        to_int(row.get('actual_lead_time_days')),
                        to_int(row.get('delay_days')),
                        clean_value(row.get('delivery_status')),
                        clean_value(row.get('disruption_event')),
                        to_float(row.get('geopolitical_risk_index')),
                        to_float(row.get('weather_severity_index')),
                        to_float(row.get('inflation_rate_pct')),
                        to_float(row.get('shipping_cost_usd')),
                        to_float(row.get('order_weight_kg')),
                        clean_value(row.get('mitigation_action')),
                        to_float(row.get('geopolitical_risk_index')),
                    )
                )
    risk_path = SILVER / 'supply_chain' / 'importante_global_supply_chain_risk_2026_csv.csv'
    if risk_path.exists():
        with risk_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_supply_chain(date_text, order_id, origin_city, destination_city, route_type, transport_mode, product_category, base_lead_time_days, scheduled_lead_time_days, actual_lead_time_days, delay_days, delivery_status, disruption_event, geopolitical_risk_index, weather_severity_index, inflation_rate_pct, shipping_cost_usd, order_weight_kg, mitigation_action, risk_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        clean_value(row.get('shipment_id')),
                        clean_value(row.get('origin_port')),
                        clean_value(row.get('destination_port')),
                        None,
                        clean_value(row.get('transport_mode')),
                        clean_value(row.get('product_category')),
                        None,
                        None,
                        None,
                        None,
                        None,
                        'Geopolitical risk',
                        to_float(row.get('risk_score')),
                        None,
                        None,
                        None,
                        to_float(row.get('weight_mt')) * 1000,
                        'Risk monitoring',
                        to_float(row.get('risk_score')),
                    )
                )
    conn.commit()


def insert_fact_market(conn):
    market_path = SILVER / 'market' / 'market_trend_external_indice_com_geopolitical_risk_csv.csv'
    if market_path.exists():
        with market_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_market(date_text, open_price, close_price, high_price, low_price, volume, daily_return_pct, volatility_range, vix_close, economic_news_flag, sentiment_score, federal_rate_change_flag, geopolitical_risk_score, currency_index, source_dataset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        to_float(row.get('open_price')),
                        to_float(row.get('close_price')),
                        to_float(row.get('high_price')),
                        to_float(row.get('low_price')),
                        to_int(row.get('volume')),
                        to_float(row.get('daily_return_pct')),
                        to_float(row.get('volatility_range')),
                        to_float(row.get('vix_close')),
                        to_int(row.get('economic_news_flag')),
                        to_float(row.get('sentiment_score')),
                        to_int(row.get('federal_rate_change_flag')),
                        to_float(row.get('geopolitical_risk_score')),
                        to_float(row.get('currency_index')),
                        'market_trend_external',
                    )
                )
    dolfut_path = SILVER / 'market' / 'dolfut_csv.csv'
    if dolfut_path.exists():
        with dolfut_path.open('r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                conn.execute(
                    'INSERT INTO fact_market(date_text, open_price, close_price, high_price, low_price, volume, daily_return_pct, volatility_range, vix_close, economic_news_flag, sentiment_score, federal_rate_change_flag, geopolitical_risk_score, currency_index, source_dataset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        clean_value(row.get('date_text')),
                        to_float(row.get('open_price')),
                        to_float(row.get('close_price')),
                        to_float(row.get('high_price')),
                        to_float(row.get('low_price')),
                        to_int(row.get('volume')),
                        to_float(row.get('daily_return_pct')),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        'dolfut',
                    )
                )
    conn.commit()


def populate_dim_dates(conn):
    date_values = set()
    for table_name, column in [
        ('fact_commodity_prices', 'date_text'),
        ('fact_energy_prices', 'date_text'),
        ('fact_macro_finance', 'date_text'),
        ('fact_port_activity', 'date_text'),
        ('fact_trade', 'date_text'),
        ('fact_supply_chain', 'date_text'),
        ('fact_market', 'date_text'),
    ]:
        rows = conn.execute(f'SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL').fetchall()
        for (value,) in rows:
            if value:
                date_values.add(value)
    for value in sorted(date_values):
        parsed = parse_date(value)
        if not parsed:
            continue
        dt = datetime.strptime(parsed, '%Y-%m-%d')
        conn.execute(
            'INSERT OR IGNORE INTO dim_date(date_text, year, month, day) VALUES (?, ?, ?, ?)',
            (parsed, dt.year, dt.month, dt.day),
        )
    conn.commit()


def main() -> None:
    build_silver_layer()
    conn = ensure_db()
    insert_dim_country(conn)
    insert_dim_commodity(conn)
    insert_dim_port(conn)
    insert_fact_commodity_prices(conn)
    insert_fact_energy_prices(conn)
    insert_fact_macro_finance(conn)
    insert_fact_port_activity(conn)
    insert_fact_trade(conn)
    insert_fact_supply_chain(conn)
    insert_fact_market(conn)
    populate_dim_dates(conn)
    conn.close()

    print(f'Created silver layer at: {SILVER}')
    print(f'Created warehouse database at: {DB_PATH}')


if __name__ == '__main__':
    main()
