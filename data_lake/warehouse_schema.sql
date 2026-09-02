PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_date (
    date_text TEXT PRIMARY KEY,
    year INTEGER,
    month INTEGER,
    day INTEGER
);

CREATE TABLE IF NOT EXISTS dim_country (
    country_name TEXT PRIMARY KEY,
    region_name TEXT,
    income_level TEXT
);

CREATE TABLE IF NOT EXISTS dim_commodity (
    commodity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity_name TEXT NOT NULL UNIQUE,
    category TEXT,
    unit TEXT,
    currency TEXT
);

CREATE TABLE IF NOT EXISTS dim_port (
    port_id TEXT PRIMARY KEY,
    port_name TEXT,
    country_name TEXT,
    iso3 TEXT,
    region_name TEXT,
    world_water_body TEXT
);

CREATE TABLE IF NOT EXISTS fact_commodity_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    commodity_name TEXT,
    category TEXT,
    unit TEXT,
    currency TEXT,
    price_usd REAL
);

CREATE TABLE IF NOT EXISTS fact_energy_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    country_name TEXT,
    region_name TEXT,
    fuel_type TEXT,
    price_usd REAL,
    brent_crude_usd REAL,
    tax_percentage REAL
);

CREATE TABLE IF NOT EXISTS fact_macro_finance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    country_name TEXT,
    usd_exchange_rate REAL,
    policy_rate_pct REAL,
    bond_yield_pct REAL,
    yield_spread_pct REAL,
    equity_index_level REAL,
    oil_price_usd_bbl REAL
);

CREATE TABLE IF NOT EXISTS fact_port_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    port_id TEXT,
    port_name TEXT,
    country_name TEXT,
    iso3 TEXT,
    portcalls_container INTEGER,
    portcalls_dry_bulk INTEGER,
    portcalls_general_cargo INTEGER,
    portcalls_roro INTEGER,
    portcalls_tanker INTEGER,
    portcalls_total INTEGER,
    import_container REAL,
    import_dry_bulk REAL,
    import_general_cargo REAL,
    import_roro REAL,
    import_tanker REAL
);

CREATE TABLE IF NOT EXISTS fact_trade (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    trade_type TEXT,
    exporter_name TEXT,
    importer_name TEXT,
    origin_country TEXT,
    destination_country TEXT,
    hs_code TEXT,
    hs_description TEXT,
    quantity REAL,
    quantity_uom TEXT,
    gross_weight REAL,
    gross_weight_uom TEXT,
    customs_value_bwp REAL
);

CREATE TABLE IF NOT EXISTS fact_supply_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    order_id TEXT,
    origin_city TEXT,
    destination_city TEXT,
    route_type TEXT,
    transport_mode TEXT,
    product_category TEXT,
    base_lead_time_days INTEGER,
    scheduled_lead_time_days INTEGER,
    actual_lead_time_days INTEGER,
    delay_days INTEGER,
    delivery_status TEXT,
    disruption_event TEXT,
    geopolitical_risk_index REAL,
    weather_severity_index REAL,
    inflation_rate_pct REAL,
    shipping_cost_usd REAL,
    order_weight_kg REAL,
    mitigation_action TEXT,
    risk_score REAL
);

CREATE TABLE IF NOT EXISTS fact_market (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_text TEXT,
    open_price REAL,
    close_price REAL,
    high_price REAL,
    low_price REAL,
    volume INTEGER,
    daily_return_pct REAL,
    volatility_range REAL,
    vix_close REAL,
    economic_news_flag INTEGER,
    sentiment_score REAL,
    federal_rate_change_flag INTEGER,
    geopolitical_risk_score REAL,
    currency_index REAL,
    source_dataset TEXT
);
