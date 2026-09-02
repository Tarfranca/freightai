import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import traceback
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "pandas"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import pandas as pd
    from flask import Flask, jsonify, request
    from flask_cors import CORS

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import numpy as np

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
ML_READY_PATH = BASE_DIR / "gold" / "ml_ready.csv"
RAW_DIR = BASE_DIR / "raw"
WAREHOUSE_DB = BASE_DIR / "warehouse" / "warehouse.db"
LOG_PATH = BASE_DIR / "api.log"
ENV_PATH = BASE_DIR / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

FRIENDLY_FEATURE_NAMES = {
    "usd_brl_close": "USD/BRL",
    "brent_close": "Brent",
    "gold_close": "Ouro",
    "dxy_close": "DXY",
    "us_10y_yield_close": "Treasury 10Y",
    "bdry_close": "BDRY",
    "zim_close": "ZIM",
    "vix_close": "VIX",
    "geopolitical_risk_index": "Risco Geopolítico",
    "weather_severity_index": "Clima",
    "inflation_rate_pct": "Inflação",
    "delay_days": "Atrasos",
    "base_lead_time_days": "Lead Time Base",
    "scheduled_lead_time_days": "Lead Time Programado",
    "actual_lead_time_days": "Lead Time Real",
    "id": "Peso da Carga",
}
FEATURE_GROUP_PREFIXES = {
    "transport_mode_": "Modal",
    "route_type_": "Rota",
    "product_category_": "Categoria do Produto",
    "delivery_status_": "Status de Entrega",
}

logger = logging.getLogger("freightai_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

app = Flask(__name__)
CORS(app)

MODEL = None
FEATURE_COLUMNS = []
FEATURE_TEMPLATE = {}
EXTERNAL_DATA = {}

MARKET_FILES = {
    "usd_brl": BASE_DIR / "raw" / "macro_finance" / "usd_brl_daily.csv",
    "brent": BASE_DIR / "raw" / "energy" / "brent_daily.csv",
    "gold": BASE_DIR / "raw" / "commodity" / "gold_daily.csv",
    "vix": BASE_DIR / "raw" / "market" / "vix_daily.csv",
    "zim": BASE_DIR / "raw" / "market" / "zim_daily.csv",
    "bdry": BASE_DIR / "raw" / "market" / "bdry_etf.csv",
    "dxy": BASE_DIR / "raw" / "macro_finance" / "dxy_daily.csv",
}

PORTS_PATH = BASE_DIR / "gold" / "ports_clean.csv"
AIRPORTS_PATH = BASE_DIR / "gold" / "airports_clean.csv"
PORTS_DF = pd.DataFrame()
AIRPORTS_DF = pd.DataFrame()

VALID_ROUTES = ["Atlantic", "Pacific", "Suez", "Intra-Asia"]

CARGO_TYPE_TO_MODEL_CATEGORY = {
    "Geral": "Consumer Electronics",
    "Perigosa": "Pharmaceuticals",
    "Refrigerada": "Pharmaceuticals",
    "Granel Sólido": "Consumer Electronics",
    "Granel Líquido": "Consumer Electronics",
    "Projeto/OOG": "Consumer Electronics",
    "Farmacêutico": "Pharmaceuticals",
    "Automotivo": "Consumer Electronics",
    "Perecível": "Pharmaceuticals",
    "Valioso": "Consumer Electronics",
    "E-commerce": "Consumer Electronics",
}

PEAK_SEASON_MONTHS = {10, 11, 12, 3, 4}

# Incoterms 2020 — who covers freight/insurance/duties, in short form for the UI.
INCOTERM_SCOPE = {
    "EXW": "Comprador assume tudo a partir da origem: frete, seguro, exportação e importação.",
    "FOB": "Vendedor entrega a bordo no porto de origem; comprador paga o frete marítimo e o seguro.",
    "FCA": "Vendedor entrega ao transportador indicado; comprador contrata o frete principal.",
    "CFR": "Vendedor paga o frete até o porto de destino; seguro e desembaraço de importação por conta do comprador.",
    "CIF": "Vendedor paga frete e seguro mínimo até o porto de destino; importação por conta do comprador.",
    "CPT": "Vendedor paga o transporte até o destino combinado; risco transfere-se antes, no embarque.",
    "CIP": "Vendedor paga transporte e seguro (cobertura ampla) até o destino combinado.",
    "DAP": "Vendedor entrega no destino, pronto para descarga; comprador cuida da importação.",
    "DPU": "Vendedor entrega e descarrega no destino; comprador cuida da importação.",
    "DDP": "Vendedor assume tudo, incluindo impostos de importação no destino.",
}


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    stripped = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return stripped.lower().strip()


def load_reference_data():
    global PORTS_DF, AIRPORTS_DF
    if PORTS_PATH.exists():
        PORTS_DF = pd.read_csv(PORTS_PATH)
        PORTS_DF["_search"] = (
            PORTS_DF["port_name"].astype(str) + " " + PORTS_DF["country"].astype(str)
        ).map(normalize_text)
    else:
        PORTS_DF = pd.DataFrame()

    if AIRPORTS_PATH.exists():
        AIRPORTS_DF = pd.read_csv(AIRPORTS_PATH)
        AIRPORTS_DF["_search"] = (
            AIRPORTS_DF["name"].astype(str) + " " + AIRPORTS_DF["city"].astype(str)
            + " " + AIRPORTS_DF["country"].astype(str) + " " + AIRPORTS_DF["iata"].astype(str)
        ).map(normalize_text)
    else:
        AIRPORTS_DF = pd.DataFrame()


def multi_token_search(df: pd.DataFrame, query: str, limit: int) -> pd.DataFrame:
    if df.empty:
        return df
    tokens = [token for token in normalize_text(query).split(" ") if token]
    if not tokens:
        return df.head(limit)
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask &= df["_search"].str.contains(token, na=False, regex=False)
    return df[mask].head(limit)


def find_port_row(name: str):
    if PORTS_DF.empty or not name:
        return None
    query = normalize_text(name)
    exact = PORTS_DF[PORTS_DF["port_name"].astype(str).map(normalize_text) == query]
    if not exact.empty:
        return exact.iloc[0]
    matches = multi_token_search(PORTS_DF, name, 1)
    return matches.iloc[0] if not matches.empty else None


def find_airport_row(query: str):
    if AIRPORTS_DF.empty or not query:
        return None
    normalized = normalize_text(query)
    exact_iata = AIRPORTS_DF[AIRPORTS_DF["iata"].astype(str).map(normalize_text) == normalized]
    if not exact_iata.empty:
        return exact_iata.iloc[0]
    matches = multi_token_search(AIRPORTS_DF, query, 1)
    return matches.iloc[0] if not matches.empty else None


def classify_region(lon) -> str:
    if lon is None or (isinstance(lon, float) and pd.isna(lon)):
        return "Unknown"
    lon = float(lon)
    if -170 <= lon <= -30:
        return "Americas"
    if -30 < lon <= 60:
        return "EMEA"
    return "Asia"


def infer_route_type(origin_lon, destination_lon) -> str:
    origin_region = classify_region(origin_lon)
    destination_region = classify_region(destination_lon)
    pair = {origin_region, destination_region}
    if pair == {"Asia"}:
        return "Intra-Asia"
    if pair == {"Americas", "Asia"}:
        return "Pacific"
    if pair == {"EMEA", "Asia"}:
        return "Suez"
    return "Atlantic"


def compute_risk_by_route() -> pd.DataFrame:
    conn = sqlite3.connect(WAREHOUSE_DB)
    try:
        df = pd.read_sql(
            """
            SELECT route_type, delivery_status, delay_days, actual_lead_time_days, weather_severity_index
            FROM fact_supply_chain
            WHERE route_type IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    stats = df.groupby("route_type").agg(
        late_rate=("delivery_status", lambda s: float((s == "Late").mean())),
        avg_delay_days=("delay_days", "mean"),
        lead_time_volatility=("actual_lead_time_days", "std"),
        avg_weather=("weather_severity_index", "mean"),
        shipments=("route_type", "size"),
    )

    components = ["late_rate", "avg_delay_days", "lead_time_volatility"]
    normalized = pd.DataFrame(index=stats.index)
    for column in components:
        col_min, col_max = stats[column].min(), stats[column].max()
        span = col_max - col_min
        normalized[column] = (stats[column] - col_min) / span if span else 0.5
    stats["risk_score"] = normalized[components].mean(axis=1) * 10.0
    return stats


def get_route_cost_per_kg(route_type: str, modal: str):
    conn = sqlite3.connect(WAREHOUSE_DB)
    try:
        row = conn.execute(
            """
            SELECT AVG(shipping_cost_usd / order_weight_kg) FROM fact_supply_chain
            WHERE route_type = ? AND transport_mode = ? AND shipping_cost_usd IS NOT NULL
              AND order_weight_kg IS NOT NULL AND order_weight_kg > 0
            """,
            (route_type, modal),
        ).fetchone()
    finally:
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def get_route_market_benchmark(route_type: str, modal: str):
    conn = sqlite3.connect(WAREHOUSE_DB)
    try:
        row = conn.execute(
            """
            SELECT AVG(shipping_cost_usd) FROM fact_supply_chain
            WHERE route_type = ? AND transport_mode = ? AND shipping_cost_usd IS NOT NULL
            """,
            (route_type, modal),
        ).fetchone()
    finally:
        conn.close()
    return float(row[0]) if row and row[0] is not None else None


def json_safe(obj):
    """Recursively replace NaN/NaT with None so jsonify emits valid `null`.

    pandas' `.where(pd.notna(df), None)` looks like it does this, but on a
    float64 column pandas silently coerces the None back into NaN (the dtype
    doesn't change), so the JSON ends up with a literal `NaN` token — invalid
    JSON that breaks `fetch().json()` in the browser. Apply this AFTER
    `.to_dict()`/`.to_dict(orient="records")`, once values are plain Python
    objects pandas can no longer "helpfully" convert back.
    """
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe(item) for item in obj]
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def error_response(message: str, status_code: int = 500):
    logger.exception(message)
    response = jsonify({"error": message, "status": status_code})
    response.status_code = status_code
    return response


def find_close_column(df: pd.DataFrame):
    for candidate in ["close", "Close", "close_price", "Close Price", "price_usd", "usd_exchange_rate", "brent_crude_usd"]:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        lower = str(column).lower()
        if "close" in lower or "price" in lower or "exchange" in lower:
            return column
    return df.columns[-1]


def sanitize_feature_name(name: str) -> str:
    """Match train_model.py's LightGBM-safe column sanitizer exactly, so
    ml_ready.csv columns (which still have spaces/hyphens) can be mapped onto
    the model's actual (sanitized) feature names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


def load_model_and_feature_structure():
    global MODEL, FEATURE_COLUMNS, FEATURE_TEMPLATE
    MODEL = None
    try:
        with MODEL_PATH.open("rb") as file:
            MODEL = __import__("pickle").load(file)
    except Exception as exc:
        logger.warning("Model file could not be loaded: %s", exc)

    # The model's own feature_names_in_ (set by scikit-learn/XGBoost/LightGBM
    # at fit time) is the single source of truth for what it expects — it can
    # never drift out of sync the way a hardcoded list or a naive re-read of
    # ml_ready.csv can (ml_ready.csv still has un-dummied city columns and
    # un-sanitized category names; get_dummies() only happens inside
    # train_model.py at training time). Never hardcode this list.
    if MODEL is not None:
        if hasattr(MODEL, "feature_names_in_"):
            FEATURE_COLUMNS = [str(column) for column in MODEL.feature_names_in_]
        elif hasattr(MODEL, "get_booster"):
            FEATURE_COLUMNS = [str(column) for column in MODEL.get_booster().feature_names]
        else:
            FEATURE_COLUMNS = []
    else:
        FEATURE_COLUMNS = []

    if not FEATURE_COLUMNS and ML_READY_PATH.exists():
        # Fallback for a model that doesn't expose its own feature names:
        # reproduce the same dummy-encoding + sanitizing train_model.py does,
        # so the derived columns actually match what such a model would need.
        feature_df = pd.read_csv(ML_READY_PATH)
        feature_df = feature_df.drop(columns=["target_shipping_cost_usd"], errors="ignore")
        feature_df = pd.get_dummies(feature_df, drop_first=False)
        FEATURE_COLUMNS = [sanitize_feature_name(column) for column in feature_df.columns]

    if FEATURE_COLUMNS:
        if ML_READY_PATH.exists():
            feature_df = pd.read_csv(ML_READY_PATH)
            sanitized_lookup = {sanitize_feature_name(column): column for column in feature_df.columns}
            FEATURE_TEMPLATE = {}
            for column in FEATURE_COLUMNS:
                source_column = sanitized_lookup.get(column)
                if source_column and pd.api.types.is_numeric_dtype(feature_df[source_column]):
                    values = feature_df[source_column].dropna()
                    FEATURE_TEMPLATE[column] = float(values.median()) if not values.empty else 0.0
                else:
                    FEATURE_TEMPLATE[column] = 0.0
        else:
            FEATURE_TEMPLATE = {column: 0.0 for column in FEATURE_COLUMNS}
    else:
        FEATURE_TEMPLATE = {}


def load_external_data():
    global EXTERNAL_DATA
    EXTERNAL_DATA = {}
    for ticker, path in MARKET_FILES.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue

            close_column = find_close_column(df)
            date_column = next((column for column in ["Date", "date", "date_text"] if column in df.columns), None)
            if date_column is None:
                df = df.copy()
                df["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="D")
                date_column = "date"

            if close_column not in df.columns:
                continue

            df = df[[date_column, close_column]].copy()
            df.columns = ["date", "close"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date")
            if df.empty:
                continue

            last_row = df.iloc[-1]
            previous_row = df.iloc[-2] if len(df) > 1 else last_row
            last_value = float(last_row["close"])
            prev_value = float(previous_row["close"]) if pd.notna(previous_row["close"]) else last_value
            change_pct = ((last_value - prev_value) / prev_value * 100.0) if prev_value else 0.0
            EXTERNAL_DATA[ticker] = {
                "path": str(path),
                "last_value": last_value,
                "previous_value": prev_value,
                "change_pct": change_pct,
                "date": last_row["date"].strftime("%Y-%m-%d"),
                "history": [{"date": x.strftime("%Y-%m-%d"), "close": float(y)} for x, y in zip(df["date"], df["close"])],
            }
        except Exception as exc:
            logger.warning("Could not load external data for %s: %s", ticker, exc)


def build_prediction_vector(payload):
    if not FEATURE_COLUMNS:
        raise ValueError("Feature structure not loaded from ml_ready.csv")

    modal = str(payload.get("modal", "Sea")).strip()
    route = str(payload.get("route", "Atlantic")).strip()
    category = str(payload.get("category", "Textiles")).strip()
    weight_kg = float(payload.get("weight_kg", 0.0) or 0.0)
    delivery_status = "On Time"

    row = FEATURE_TEMPLATE.copy()
    for key in [column for column in row if column.startswith("transport_mode_") or column.startswith("route_type_") or column.startswith("product_category_") or column.startswith("delivery_status_")]:
        row[key] = 0.0

    # Real one-hot columns from the retrained model (route_type now has its
    # own Suez/Intra-Asia/Commodity columns — no more collapsing them onto
    # Atlantic/Pacific like the old 2-category model was forced to).
    route_lookup = {
        "Atlantic": "route_type_Atlantic",
        "Pacific": "route_type_Pacific",
        "Suez": "route_type_Suez",
        "Intra-Asia": "route_type_Intra_Asia",
        "Intra Asia": "route_type_Intra_Asia",
        "Commodity": "route_type_Commodity",
    }
    modal_lookup = {"Sea": "transport_mode_Sea", "Air": "transport_mode_Air"}
    category_lookup = {
        "Textiles": "product_category_Textiles",
        "Pharmaceuticals": "product_category_Pharmaceuticals",
        "Semiconductors": "product_category_Semiconductors",
        "Consumer Electronics": "product_category_Consumer_Electronics",
        "Raw Materials": "product_category_Raw_Materials",
        "Auto Parts": "product_category_Auto_Parts",
        "Perishable Foods": "product_category_Perishable_Foods",
    }

    route_key = route_lookup.get(route, "route_type_Atlantic")
    if route_key in row:
        row[route_key] = 1.0

    modal_key = modal_lookup.get(modal, "transport_mode_Sea")
    if modal_key in row:
        row[modal_key] = 1.0

    category_key = category_lookup.get(category, next((col for col in row if col.startswith("product_category_")), None))
    if category_key and category_key in row:
        for col in [col for col in row if col.startswith("product_category_")]:
            row[col] = 0.0
        row[category_key] = 1.0

    row["delivery_status_On_Time"] = 1.0 if delivery_status == "On Time" else 0.0
    row["delivery_status_Late"] = 1.0 if delivery_status == "Late" else 0.0
    row["id"] = weight_kg
    row["base_lead_time_days"] = 8.0 + (0.5 if route in {"Pacific", "Intra-Asia", "Intra Asia"} else 0.0)
    row["scheduled_lead_time_days"] = row["base_lead_time_days"] + 2.0
    row["actual_lead_time_days"] = row["scheduled_lead_time_days"] + 1.0
    row["delay_days"] = 1.0 if route in {"Suez", "Intra-Asia", "Intra Asia"} else 0.0
    row["geopolitical_risk_index"] = 6.8 if route in {"Suez", "Intra-Asia", "Intra Asia"} else 4.2 if route in {"Pacific", "Atlantic"} else 3.5
    row["weather_severity_index"] = 2.1 if route in {"Pacific", "Intra-Asia", "Intra Asia"} else 1.6
    row["inflation_rate_pct"] = 3.1

    for ticker in ["usd_brl", "brent", "gold", "dxy", "bdry", "zim", "vix"]:
        if ticker in EXTERNAL_DATA:
            key_name = {
                "usd_brl": "usd_brl_close",
                "brent": "brent_close",
                "gold": "gold_close",
                "dxy": "dxy_close",
                "bdry": "bdry_close",
                "zim": "zim_close",
                "vix": "vix_close",
            }[ticker]
            row[key_name] = float(EXTERNAL_DATA[ticker]["last_value"])

    if "us_10y_yield_close" in row:
        row["us_10y_yield_close"] = EXTERNAL_DATA.get("dxy", {}).get("last_value", row["us_10y_yield_close"])

    ordered = []
    for column in FEATURE_COLUMNS:
        ordered.append(float(row.get(column, FEATURE_TEMPLATE.get(column, 0.0))))
    return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)


@app.before_request
def _log_request_start():
    app.logger.info("%s %s", request.method, request.path)


@app.route("/api/health", methods=["GET"])
def health():
    try:
        result = {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "data_loaded": bool(EXTERNAL_DATA),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        return jsonify(result)
    except Exception as exc:
        logger.exception("Health endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/market", methods=["GET"])
def market():
    try:
        payload = {}
        for ticker, meta in EXTERNAL_DATA.items():
            payload[ticker] = {
                "last_close": meta["last_value"],
                "previous_close": meta["previous_value"],
                "change_pct": meta["change_pct"],
                "date": meta["date"],
            }
        return jsonify(payload)
    except Exception as exc:
        logger.exception("Market endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/market/history", methods=["GET"])
def market_history():
    try:
        ticker = request.args.get("ticker", "usd_brl").strip().lower()
        days = int(request.args.get("days", 30) or 30)
        supported = {"usd_brl", "brent", "gold", "vix", "zim", "bdry", "dxy"}
        if ticker not in supported:
            raise ValueError(f"Unsupported ticker '{ticker}'")
        if ticker not in EXTERNAL_DATA:
            raise ValueError(f"Ticker '{ticker}' not loaded")
        history = EXTERNAL_DATA[ticker]["history"][-days:]
        return jsonify(history)
    except Exception as exc:
        logger.exception("Market history endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if MODEL is None:
            raise ValueError("Model was not loaded")
        payload = request.get_json(silent=True) or {}
        required_fields = ["modal", "route", "category", "weight_kg"]
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        X = build_prediction_vector(payload)
        prediction = float(MODEL.predict(X)[0])
        prediction = max(prediction, 0.0)
        trend_pct = 0.0
        route_trend = {"Atlantic": -2.1, "Pacific": 5.1, "Suez": 7.4, "Intra-Asia": 3.8}
        trend_pct = route_trend.get(payload.get("route", "Atlantic"), 0.0)
        if payload.get("modal") == "Air":
            trend_pct += 9.2

        low = prediction * 0.85
        high = prediction * 1.15
        factors = [
            {"name": "Modal", "value": payload.get("modal")},
            {"name": "Rota", "value": payload.get("route")},
            {"name": "Peso", "value": f"{float(payload.get('weight_kg',0.0))} kg"},
        ]
        result = {
            "predicted_cost_usd": round(prediction, 2),
            "confidence_low": round(low, 2),
            "confidence_high": round(high, 2),
            "trend_pct": round(trend_pct, 2),
            "model_r2": 0.972,
            "factors": factors,
        }
        return jsonify(result)
    except Exception as exc:
        logger.exception("Predict endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        from optimize import (optimize_fastest_route, optimize_modal_mix, optimize_pareto,
                             optimize_route, optimize_time_window)

        payload = request.get_json(silent=True) or {}
        modal = payload.get("modal", "Sea")
        route = payload.get("route", "Atlantic")
        category = payload.get("category", "Textiles")
        weight_kg = float(payload.get("weight_kg", 1200.0) or 1200.0)
        budget_usd = float(payload.get("budget_usd", max(1500.0, weight_kg * 1.8)))

        route_result = optimize_route(weight_kg=weight_kg, category=category, budget_usd=budget_usd)
        fastest_result = optimize_fastest_route(weight_kg=weight_kg, category=category)
        time_result = optimize_time_window(modal=modal, route=route, category=category, weight_kg=weight_kg)
        modal_result = optimize_modal_mix(total_weight_kg=weight_kg, category=category, route=route, budget_usd=budget_usd)
        pareto_result = optimize_pareto(category=category, route=route, weight_kg=weight_kg)

        result = {
            "cheapest_route": route_result,
            "fastest_route": fastest_result,
            "best_month": time_result,
            "pareto_points": pareto_result.get("pareto_points", []),
            "modal_mix": modal_result,
        }
        return jsonify(result)
    except Exception as exc:
        logger.exception("Optimize endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/correlations", methods=["GET"])
def correlations():
    try:
        df = pd.read_csv(ML_READY_PATH)
        market_vars = ["usd_brl_close", "brent_close", "gold_close", "vix_close", "zim_close", "bdry_close", "dxy_close"]
        route_keys = ["atlantic", "pacific", "suez", "intra_asia"]
        route_columns = {
            "atlantic": "route_type_Atlantic",
            "pacific": "route_type_Pacific",
            "suez": "route_type_Atlantic",
            "intra_asia": "route_type_Pacific",
        }
        result = []
        for var in market_vars:
            row = {"variable": var}
            for key in route_keys:
                column_name = route_columns[key]
                subset = df[df[column_name] == 1].copy() if column_name in df.columns else df.copy()
                if subset.empty or subset[var].nunique() < 2 or "target_shipping_cost_usd" not in subset.columns:
                    row[key] = 0.0
                    continue
                corr = subset[var].corr(subset["target_shipping_cost_usd"])
                row[key] = float(corr) if pd.notna(corr) else 0.0
            result.append(row)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Correlations endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/routes", methods=["GET"])
def routes():
    try:
        conn = sqlite3.connect(WAREHOUSE_DB)
        try:
            query = """
                SELECT
                    route_type AS route,
                    transport_mode AS modal,
                    AVG(shipping_cost_usd) AS avg_cost_usd,
                    AVG(actual_lead_time_days) AS avg_lead_time_days,
                    COUNT(*) AS shipments
                FROM fact_supply_chain
                WHERE route_type IS NOT NULL AND transport_mode IS NOT NULL
                GROUP BY route_type, transport_mode
                ORDER BY AVG(shipping_cost_usd) DESC
            """
            rows = conn.execute(query).fetchall()
        finally:
            conn.close()

        result = [{
            "route": row[0],
            "modal": row[1],
            "avg_cost_usd": round(float(row[2]), 2) if row[2] is not None else None,
            "avg_lead_time_days": round(float(row[3]), 2) if row[3] is not None else None,
            "shipments": int(row[4]) if row[4] is not None else 0,
        } for row in rows]
        return jsonify(result)
    except Exception as exc:
        logger.exception("Routes endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/freight_index", methods=["GET"])
def freight_index():
    try:
        days = int(request.args.get("days", 90) or 90)
        conn = sqlite3.connect(WAREHOUSE_DB)
        try:
            df = pd.read_sql(
                """
                SELECT date_text AS date, AVG(shipping_cost_usd) AS avg_cost, COUNT(*) AS shipments
                FROM fact_supply_chain
                WHERE date_text IS NOT NULL AND shipping_cost_usd IS NOT NULL
                GROUP BY date_text
                ORDER BY date_text
                """,
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            raise ValueError("No shipment data available to build the freight index")

        df = df.tail(days).reset_index(drop=True)
        base = float(df["avg_cost"].iloc[0])
        df["index_value"] = (df["avg_cost"] / base) * 1000.0 if base else 0.0

        last_value = float(df["index_value"].iloc[-1])
        prev_value = float(df["index_value"].iloc[-2]) if len(df) > 1 else last_value
        change_pct = ((last_value - prev_value) / prev_value * 100.0) if prev_value else 0.0

        result = {
            "last_value": round(last_value, 2),
            "change_pct": round(change_pct, 2),
            "history": [
                {"date": row.date, "index_value": round(float(row.index_value), 2)}
                for row in df.itertuples()
            ],
        }
        return jsonify(result)
    except Exception as exc:
        logger.exception("Freight index endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/routes/pairs", methods=["GET"])
def route_pairs():
    try:
        limit = int(request.args.get("limit", 6) or 6)
        conn = sqlite3.connect(WAREHOUSE_DB)
        try:
            df = pd.read_sql(
                """
                SELECT origin_city, destination_city, route_type, transport_mode,
                       date_text, shipping_cost_usd, actual_lead_time_days, delivery_status
                FROM fact_supply_chain
                WHERE shipping_cost_usd IS NOT NULL AND origin_city IS NOT NULL AND destination_city IS NOT NULL
                """,
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            raise ValueError("No shipment data available to build route pairs")

        pair_counts = df.groupby(["origin_city", "destination_city"]).size().sort_values(ascending=False)
        top_pairs = pair_counts.head(limit).index.tolist()

        raw_rows = []
        for origin, destination in top_pairs:
            subset = df[(df["origin_city"] == origin) & (df["destination_city"] == destination)].sort_values("date_text")
            avg_cost = float(subset["shipping_cost_usd"].mean())
            avg_lead = float(subset["actual_lead_time_days"].mean())
            late_rate = float((subset["delivery_status"] == "Late").mean())
            lead_volatility = float(subset["actual_lead_time_days"].std() or 0.0)
            shipments = int(len(subset))
            midpoint = len(subset) // 2
            if midpoint >= 1 and len(subset) - midpoint >= 1:
                older_avg = float(subset["shipping_cost_usd"].iloc[:midpoint].mean())
                recent_avg = float(subset["shipping_cost_usd"].iloc[midpoint:].mean())
                trend_pct = ((recent_avg - older_avg) / older_avg * 100.0) if older_avg else 0.0
            else:
                trend_pct = 0.0

            weekly = subset.copy()
            weekly["date_parsed"] = pd.to_datetime(weekly["date_text"], errors="coerce")
            weekly = weekly.dropna(subset=["date_parsed"]).set_index("date_parsed")
            weekly_avg = weekly["shipping_cost_usd"].resample("W").mean().dropna().tail(12)
            history = [
                {"week": index.strftime("%Y-%m-%d"), "avg_cost_usd": round(float(value), 2)}
                for index, value in weekly_avg.items()
            ]

            raw_rows.append({
                "origin": origin,
                "destination": destination,
                "route_type": subset["route_type"].mode().iat[0] if not subset["route_type"].mode().empty else None,
                "modal": subset["transport_mode"].mode().iat[0] if not subset["transport_mode"].mode().empty else None,
                "avg_cost_usd": avg_cost,
                "avg_lead_time_days": avg_lead,
                "late_rate": late_rate,
                "lead_volatility": lead_volatility,
                "trend_pct": trend_pct,
                "shipments": shipments,
                "history": history,
            })

        pairs_df = pd.DataFrame(raw_rows)
        normalized = pd.DataFrame(index=pairs_df.index)
        for column in ["late_rate", "lead_volatility"]:
            col_min, col_max = pairs_df[column].min(), pairs_df[column].max()
            span = col_max - col_min
            normalized[column] = (pairs_df[column] - col_min) / span if span else 0.5
        pairs_df["avg_risk_score"] = normalized[["late_rate", "lead_volatility"]].mean(axis=1) * 10.0

        results = []
        for _, row in pairs_df.iterrows():
            risk_score = float(row["avg_risk_score"])
            trend_pct = float(row["trend_pct"])
            if risk_score >= 7:
                recommendation = "Evitar"
            elif risk_score >= 4.5 or trend_pct > 5:
                recommendation = "Aguardar"
            else:
                recommendation = "Contratar"

            results.append({
                "origin": row["origin"],
                "destination": row["destination"],
                "route_type": row["route_type"],
                "modal": row["modal"],
                "avg_cost_usd": round(float(row["avg_cost_usd"]), 2),
                "avg_lead_time_days": round(float(row["avg_lead_time_days"]), 2),
                "avg_risk_score": round(risk_score, 1),
                "trend_pct": round(trend_pct, 2),
                "shipments": int(row["shipments"]),
                "recommendation": recommendation,
                "history": row["history"],
            })

        results.sort(key=lambda item: item["avg_cost_usd"], reverse=True)
        return jsonify(results)
    except Exception as exc:
        logger.exception("Route pairs endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/seasonality", methods=["GET"])
def seasonality():
    try:
        conn = sqlite3.connect(WAREHOUSE_DB)
        try:
            df = pd.read_sql(
                """
                SELECT route_type, date_text, shipping_cost_usd
                FROM fact_supply_chain
                WHERE route_type IS NOT NULL AND shipping_cost_usd IS NOT NULL
                """,
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            raise ValueError("No shipment data available to build seasonality index")

        df["month"] = pd.to_datetime(df["date_text"], errors="coerce").dt.month
        df = df.dropna(subset=["month"])
        result = []
        for route_type, group in df.groupby("route_type"):
            route_avg = group["shipping_cost_usd"].mean()
            monthly = group.groupby("month")["shipping_cost_usd"].mean()
            row = {"route": route_type}
            for month in range(1, 13):
                if month in monthly.index and route_avg:
                    row[str(month)] = round(float(monthly.loc[month] / route_avg * 100.0), 1)
                else:
                    row[str(month)] = 100.0
            result.append(row)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Seasonality endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/risks", methods=["GET"])
def risks():
    try:
        stats = compute_risk_by_route()
        if stats.empty:
            raise ValueError("No shipment data available to build risk radar")

        result = []
        for route_type, row in stats.iterrows():
            risk_score = float(row["risk_score"])
            if risk_score >= 7:
                label = "CRÍTICO"
            elif risk_score >= 5:
                label = "ALTO"
            elif risk_score >= 3:
                label = "MÉDIO"
            else:
                label = "BAIXO"
            result.append({
                "route": route_type,
                "risk_score": round(risk_score, 1),
                "label": label,
                "late_rate_pct": round(float(row["late_rate"]) * 100.0, 1),
                "avg_delay_days": round(float(row["avg_delay_days"]), 2),
                "lead_time_volatility_days": round(float(row["lead_time_volatility"]), 2),
                "weather_severity_index": round(float(row["avg_weather"]), 2),
                "shipments": int(row["shipments"]),
            })

        result.sort(key=lambda item: item["risk_score"], reverse=True)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Risks endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/feature_importance", methods=["GET"])
def feature_importance():
    try:
        if MODEL is None or not hasattr(MODEL, "feature_importances_"):
            raise ValueError("Model does not expose feature importances")

        raw_importances = list(zip(FEATURE_COLUMNS, MODEL.feature_importances_))
        grouped = {}
        for name, value in raw_importances:
            value = float(value)
            matched_group = next((label for prefix, label in FEATURE_GROUP_PREFIXES.items() if name.startswith(prefix)), None)
            if matched_group:
                grouped[matched_group] = grouped.get(matched_group, 0.0) + value
            else:
                label = FRIENDLY_FEATURE_NAMES.get(name, name)
                grouped[label] = grouped.get(label, 0.0) + value

        total = sum(grouped.values()) or 1.0
        result = [
            {"variable": label, "importance_pct": round(value / total * 100.0, 1)}
            for label, value in grouped.items()
        ]
        result.sort(key=lambda item: item["importance_pct"], reverse=True)
        return jsonify(result[:8])
    except Exception as exc:
        logger.exception("Feature importance endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/friday", methods=["POST"])
def friday():
    try:
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message", "")).strip()
        if not user_message:
            raise ValueError("Message is required")

        if not GEMINI_API_KEY:
            return jsonify({
                "reply": "FRIDAY não está configurado: defina a variável de ambiente GEMINI_API_KEY "
                         "(ou crie um arquivo .env na pasta data_lake com GEMINI_API_KEY=sua_chave) e reinicie a API.",
                "configured": False,
            })

        market_summary = ", ".join(
            f"{ticker.upper()} {meta['last_value']:.2f} ({meta['change_pct']:+.1f}%)"
            for ticker, meta in EXTERNAL_DATA.items()
        ) or "dados de mercado indisponíveis"

        context = (
            "Você é FRIDAY, assistente de inteligência em fretes internacionais da plataforma FreightAI, "
            "um projeto de TCC de MBA da USP. Use os dados reais abaixo, extraídos do data lake e do "
            f"modelo XGBoost (R² 0.97), para responder. DADOS DE MERCADO ATUAIS: {market_summary}. "
            "Responda em português, de forma direta e profissional, citando números concretos quando possível."
        )

        request_body = {
            "contents": [{"parts": [{"text": context + "\n\nPERGUNTA DO USUÁRIO:\n" + user_message}]}],
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 500},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, json=request_body, timeout=45)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        reply = "Não consegui gerar resposta no momento."
        if candidates:
            parts = candidates[0].get("content", {}).get("parts") or []
            if parts and "text" in parts[0]:
                reply = parts[0]["text"]

        return jsonify({"reply": reply, "configured": True})
    except Exception as exc:
        logger.exception("Friday endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/ports", methods=["GET"])
def ports_search():
    try:
        query = request.args.get("q", "").strip()
        limit = max(1, min(int(request.args.get("limit", 20) or 20), 50))
        matches = multi_token_search(PORTS_DF, query, limit) if query else PORTS_DF.head(limit)
        columns = ["port_name", "country", "latitude", "longitude", "port_size", "water_body"]
        result = json_safe(matches[columns].to_dict(orient="records"))
        return jsonify(result)
    except Exception as exc:
        logger.exception("Ports search failed")
        return error_response(str(exc), 500)


@app.route("/api/airports", methods=["GET"])
def airports_search():
    try:
        query = request.args.get("q", "").strip()
        limit = max(1, min(int(request.args.get("limit", 20) or 20), 50))
        matches = multi_token_search(AIRPORTS_DF, query, limit) if query else AIRPORTS_DF.head(limit)
        columns = ["name", "city", "country", "iata", "icao", "lat", "lon"]
        result = json_safe(matches[columns].to_dict(orient="records"))
        return jsonify(result)
    except Exception as exc:
        logger.exception("Airports search failed")
        return error_response(str(exc), 500)


def compute_maritime_quote(payload: dict) -> dict:
    from optimize import estimate_lead_time_days

    origin_name = str(payload.get("origin_port", "")).strip()
    destination_name = str(payload.get("destination_port", "")).strip()
    container_type = str(payload.get("container_type", "20' Dry"))
    container_qty = max(1, int(payload.get("container_qty", 1) or 1))
    gross_weight_kg = max(1.0, float(payload.get("gross_weight_kg", 0.0) or 0.0))
    volume_cbm = float(payload.get("volume_cbm", 0.0) or 0.0)
    incoterm = str(payload.get("incoterm", "FOB")).upper()
    cargo_type = str(payload.get("cargo_type", "Geral"))
    shipment_date = str(payload.get("shipment_date", "")).strip()

    origin_row = find_port_row(origin_name)
    destination_row = find_port_row(destination_name)
    origin_lon = origin_row["longitude"] if origin_row is not None else None
    destination_lon = destination_row["longitude"] if destination_row is not None else None
    route_type = infer_route_type(origin_lon, destination_lon)

    category = CARGO_TYPE_TO_MODEL_CATEGORY.get(cargo_type, "Consumer Electronics")
    base_vector = build_prediction_vector({
        "modal": "Sea", "route": route_type, "category": category, "weight_kg": gross_weight_kg,
    })
    base_freight_usd = max(float(MODEL.predict(base_vector)[0]), 0.0)

    try:
        month = int(shipment_date.split("-")[1]) if "-" in shipment_date else datetime.utcnow().month
    except (ValueError, IndexError):
        month = datetime.utcnow().month

    thc_origin = 185.0 * container_qty
    thc_destination = 210.0 * container_qty
    baf = base_freight_usd * 0.18
    caf = base_freight_usd * 0.03
    bl_fee = 65.0
    vgm_fee = 35.0

    risk_table = compute_risk_by_route()
    route_risk_score = float(risk_table.loc[route_type, "risk_score"]) if route_type in risk_table.index else 0.0
    port_congestion_surcharge = base_freight_usd * 0.12 if route_risk_score > 7 else 0.0

    war_risk_surcharge = 180.0 * container_qty if route_type == "Suez" else 0.0
    reefer_surcharge = 1200.0 * container_qty if "reefer" in container_type.lower() else 0.0
    hazmat_surcharge = 350.0 * container_qty if cargo_type == "Perigosa" else 0.0
    peak_season_surcharge = base_freight_usd * 0.15 if month in PEAK_SEASON_MONTHS else 0.0

    surcharge_total = (
        thc_origin + thc_destination + baf + caf + bl_fee + vgm_fee
        + port_congestion_surcharge + war_risk_surcharge + reefer_surcharge
        + hazmat_surcharge + peak_season_surcharge
    )
    total_usd = base_freight_usd + surcharge_total

    market_benchmark = get_route_market_benchmark(route_type, "Sea")
    vs_market_pct = ((total_usd - market_benchmark) / market_benchmark * 100.0) if market_benchmark else None
    transit_time_days = estimate_lead_time_days("Sea", route_type, gross_weight_kg)

    line_items = [
        ("Base Freight", base_freight_usd),
        ("THC Origem", thc_origin),
        ("THC Destino", thc_destination),
        ("BAF", baf),
        ("CAF", caf),
        ("BL Fee", bl_fee),
        ("VGM", vgm_fee),
        ("Congestionamento", port_congestion_surcharge),
        ("War Risk", war_risk_surcharge),
        ("Reefer", reefer_surcharge),
        ("Hazmat", hazmat_surcharge),
        ("Peak Season", peak_season_surcharge),
    ]
    breakdown_chart_data = [
        {"label": label, "value": round(value, 2), "pct": round(value / total_usd * 100.0, 1) if total_usd else 0.0}
        for label, value in line_items
    ]

    return {
        "base_freight_usd": round(base_freight_usd, 2),
        "thc_origin": round(thc_origin, 2),
        "thc_destination": round(thc_destination, 2),
        "baf": round(baf, 2),
        "caf": round(caf, 2),
        "bl_fee": round(bl_fee, 2),
        "vgm_fee": round(vgm_fee, 2),
        "port_congestion_surcharge": round(port_congestion_surcharge, 2),
        "war_risk_surcharge": round(war_risk_surcharge, 2),
        "reefer_surcharge": round(reefer_surcharge, 2),
        "hazmat_surcharge": round(hazmat_surcharge, 2),
        "peak_season_surcharge": round(peak_season_surcharge, 2),
        "total_usd": round(total_usd, 2),
        "cost_per_container": round(total_usd / container_qty, 2),
        "cost_per_kg": round(total_usd / gross_weight_kg, 4),
        "confidence_low": round(base_freight_usd * 0.85 + surcharge_total, 2),
        "confidence_high": round(base_freight_usd * 1.15 + surcharge_total, 2),
        "market_benchmark": round(market_benchmark, 2) if market_benchmark else None,
        "vs_market_pct": round(vs_market_pct, 2) if vs_market_pct is not None else None,
        "transit_time_days": round(transit_time_days, 1),
        "route_risk_score": round(route_risk_score, 1),
        "incoterm_scope": INCOTERM_SCOPE.get(incoterm, INCOTERM_SCOPE["FOB"]),
        "currency_rate_usd_brl": round(EXTERNAL_DATA["usd_brl"]["last_value"], 4) if "usd_brl" in EXTERNAL_DATA else None,
        "breakdown_chart_data": breakdown_chart_data,
        "route_type": route_type,
        "volume_cbm": volume_cbm,
        "container_qty": container_qty,
    }


@app.route("/api/predict/maritime", methods=["POST"])
def predict_maritime():
    try:
        if MODEL is None:
            raise ValueError("Model was not loaded")
        payload = request.get_json(silent=True) or {}
        required = ["origin_port", "destination_port", "gross_weight_kg"]
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        result = compute_maritime_quote(payload)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Maritime predict endpoint failed")
        return error_response(str(exc), 500)


def compute_air_quote(payload: dict) -> dict:
    from optimize import estimate_lead_time_days

    origin_iata = str(payload.get("origin_airport_iata", "")).strip()
    destination_iata = str(payload.get("destination_airport_iata", "")).strip()
    gross_weight_kg = max(0.1, float(payload.get("gross_weight_kg", 0.0) or 0.0))
    volume_m3 = max(0.0, float(payload.get("volume_m3", 0.0) or 0.0))
    cargo_type = str(payload.get("cargo_type", "Geral"))
    incoterm = str(payload.get("incoterm", "FOB")).upper()
    service_type = str(payload.get("service_type", "Standard"))
    shipment_date = str(payload.get("shipment_date", "")).strip()

    origin_row = find_airport_row(origin_iata)
    destination_row = find_airport_row(destination_iata)
    origin_lon = origin_row["lon"] if origin_row is not None else None
    destination_lon = destination_row["lon"] if destination_row is not None else None
    route_type = infer_route_type(origin_lon, destination_lon)
    origin_region = classify_region(origin_lon)
    destination_region = classify_region(destination_lon)
    is_regional = origin_region == destination_region and origin_region != "Unknown"

    volumetric_weight_kg = volume_m3 * 167.0
    chargeable_weight_kg = max(gross_weight_kg, volumetric_weight_kg)
    density_kg_m3 = (gross_weight_kg / volume_m3) if volume_m3 > 0 else None

    try:
        month = int(shipment_date.split("-")[1]) if "-" in shipment_date else datetime.utcnow().month
    except (ValueError, IndexError):
        month = datetime.utcnow().month

    base_rate_per_kg = 2.80 if is_regional else 4.20
    base_freight_usd = base_rate_per_kg * chargeable_weight_kg

    # Fuel surcharge references Jet-A1/Brent co-movement: 28% at a $70 Brent baseline,
    # scaled by the live Brent price loaded from the data lake, bounded to a sane range.
    brent_price = EXTERNAL_DATA.get("brent", {}).get("last_value")
    fsc_rate = 0.28 * (brent_price / 70.0) if brent_price else 0.28
    fsc_rate = max(0.15, min(fsc_rate, 0.45))
    fuel_surcharge = base_freight_usd * fsc_rate

    security_surcharge = 0.35 * chargeable_weight_kg
    screening_fee = 0.18 * chargeable_weight_kg
    awb_fee = 45.0
    documentation_fee = 55.0

    cargo_type_surcharge = 0.0
    if cargo_type == "Perigosa":
        cargo_type_surcharge = 1.50 * chargeable_weight_kg
    elif cargo_type == "Perecível":
        cargo_type_surcharge = 0.85 * chargeable_weight_kg
    elif cargo_type == "Farmacêutico":
        cargo_type_surcharge = 1.20 * chargeable_weight_kg

    peak_season_surcharge = base_freight_usd * 0.40 if month in PEAK_SEASON_MONTHS else 0.0

    surcharge_total = (
        fuel_surcharge + security_surcharge + screening_fee + awb_fee
        + documentation_fee + cargo_type_surcharge + peak_season_surcharge
    )
    total_usd = base_freight_usd + surcharge_total

    market_benchmark = get_route_market_benchmark(route_type, "Air")
    vs_market_pct = ((total_usd - market_benchmark) / market_benchmark * 100.0) if market_benchmark else None

    service_transit_days = {"Express": 2.0, "Standard": 5.0, "Economy": 10.0}
    transit_time_days = service_transit_days.get(service_type, 5.0)

    line_items = [
        ("Base Freight", base_freight_usd),
        ("Fuel Surcharge", fuel_surcharge),
        ("Security", security_surcharge),
        ("Screening", screening_fee),
        ("AWB Fee", awb_fee),
        ("Documentation", documentation_fee),
        ("Carga Especial", cargo_type_surcharge),
        ("Peak Season", peak_season_surcharge),
    ]
    breakdown_chart_data = [
        {"label": label, "value": round(value, 2), "pct": round(value / total_usd * 100.0, 1) if total_usd else 0.0}
        for label, value in line_items
    ]

    # Sea comparison: the XGBoost model's weight signal is too weak to compare a small
    # air parcel against a full container fairly (it was trained on a weight proxy, not
    # true per-kg cost). Use the real observed $/kg for Sea on this route instead, which
    # is genuinely weight-sensitive ($0.75-0.85/kg historically vs ~$8.50-9.90/kg for Air).
    sea_cost_per_kg = get_route_cost_per_kg(route_type, "Sea") or 0.8
    sea_total_usd = max(sea_cost_per_kg * gross_weight_kg, 150.0)
    sea_transit_days = estimate_lead_time_days("Sea", route_type, gross_weight_kg)

    air_premium_usd = total_usd - sea_total_usd
    air_premium_pct = (air_premium_usd / sea_total_usd * 100.0) if sea_total_usd else None
    speed_advantage_days = sea_transit_days - transit_time_days
    value_per_kg_breakeven = (air_premium_usd / gross_weight_kg) if gross_weight_kg else None

    return {
        "gross_weight_kg": gross_weight_kg,
        "volume_m3": volume_m3,
        "chargeable_weight_kg": round(chargeable_weight_kg, 2),
        "density_kg_m3": round(density_kg_m3, 2) if density_kg_m3 is not None else None,
        "base_rate_per_kg": base_rate_per_kg,
        "base_freight_usd": round(base_freight_usd, 2),
        "fuel_surcharge": round(fuel_surcharge, 2),
        "security_surcharge": round(security_surcharge, 2),
        "screening_fee": round(screening_fee, 2),
        "awb_fee": round(awb_fee, 2),
        "documentation_fee": round(documentation_fee, 2),
        "cargo_type_surcharge": round(cargo_type_surcharge, 2),
        "peak_season_surcharge": round(peak_season_surcharge, 2),
        "total_usd": round(total_usd, 2),
        "cost_per_kg": round(total_usd / chargeable_weight_kg, 4),
        "cost_per_cbm": round(total_usd / volume_m3, 2) if volume_m3 > 0 else None,
        "confidence_low": round(total_usd * 0.9, 2),
        "confidence_high": round(total_usd * 1.1, 2),
        "market_benchmark": round(market_benchmark, 2) if market_benchmark else None,
        "vs_market_pct": round(vs_market_pct, 2) if vs_market_pct is not None else None,
        "transit_time_days": transit_time_days,
        "service_type": service_type,
        "incoterm_scope": INCOTERM_SCOPE.get(incoterm, INCOTERM_SCOPE["FOB"]),
        "currency_rate_usd_brl": round(EXTERNAL_DATA["usd_brl"]["last_value"], 4) if "usd_brl" in EXTERNAL_DATA else None,
        "breakdown_chart_data": breakdown_chart_data,
        "route_type": route_type,
        "comparison_vs_sea": {
            "sea_total_usd": round(sea_total_usd, 2),
            "air_premium_usd": round(air_premium_usd, 2),
            "air_premium_pct": round(air_premium_pct, 2) if air_premium_pct is not None else None,
            "speed_advantage_days": round(speed_advantage_days, 1),
            "value_per_kg_breakeven_usd": round(value_per_kg_breakeven, 2) if value_per_kg_breakeven is not None else None,
        },
    }


@app.route("/api/predict/air", methods=["POST"])
def predict_air():
    try:
        if MODEL is None:
            raise ValueError("Model was not loaded")
        payload = request.get_json(silent=True) or {}
        required = ["origin_airport_iata", "destination_airport_iata", "gross_weight_kg"]
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        result = compute_air_quote(payload)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Air predict endpoint failed")
        return error_response(str(exc), 500)


COMMODITY_CATEGORIES = {
    "energy": ["BZ=F", "CL=F", "NG=F", "HO=F"],
    "metals": ["GC=F", "SI=F", "HG=F", "SLX"],
    "agriculture": ["ZS=F", "ZC=F", "SB=F", "KC=F", "ZW=F"],
    "shipping": ["ZIM", "MATX", "DAC", "SBLK", "BDRY"],
    "macro": ["DX-Y.NYB", "EURUSD=X", "BRL=X", "CNY=X", "JPY=X", "^VIX", "^GSPC", "^TNX"],
}


@app.route("/api/risk/corridors", methods=["GET"])
def risk_corridors():
    try:
        path = BASE_DIR / "gold" / "corridor_risk_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run src/features/risk_engine.py first")
        df = pd.read_csv(path)
        return jsonify(json_safe(df.to_dict(orient="records")))
    except Exception as exc:
        logger.exception("Risk corridors endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/risk/geopolitical", methods=["GET"])
def risk_geopolitical():
    try:
        path = BASE_DIR / "raw" / "risk" / "gpr_index.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run src/ingestion/fetch_risk_indices.py first")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[["date", "gpr_global", "gpr_threat", "gpr_act"]].sort_values("date").tail(24)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return jsonify(json_safe(df.to_dict(orient="records")))
    except Exception as exc:
        logger.exception("Risk geopolitical endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/risk/climate", methods=["GET"])
def risk_climate():
    try:
        oni_path = BASE_DIR / "raw" / "risk" / "oni_enso.csv"
        panama_path = BASE_DIR / "raw" / "climate" / "panama_enso_risk.csv"
        weather_path = BASE_DIR / "raw" / "climate" / "port_weather.csv"

        oni_current = None
        if oni_path.exists():
            oni_df = pd.read_csv(oni_path)
            last = oni_df.iloc[-1]
            oni_current = {
                "year": int(last["year"]), "month": int(last["month"]), "season": last["season"],
                "oni_value": float(last["oni_value"]), "enso_phase": last["enso_phase"],
            }

        panama_risk = None
        if panama_path.exists():
            panama_df = pd.read_csv(panama_path)
            panama_risk = json_safe(panama_df.iloc[0].to_dict())

        port_weather = []
        if weather_path.exists():
            weather_df = pd.read_csv(weather_path)
            latest_per_port = weather_df.sort_values("date").groupby("port").tail(1)
            port_weather = json_safe(latest_per_port.to_dict(orient="records"))

        return jsonify({
            "oni_current": oni_current,
            "panama_canal_risk": panama_risk,
            "port_weather_latest": port_weather,
        })
    except Exception as exc:
        logger.exception("Risk climate endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/commodities", methods=["GET"])
def commodities():
    try:
        path = BASE_DIR / "gold" / "commodities_master.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run src/ingestion/fetch_commodities.py first")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")

        result = {}
        for category, tickers in COMMODITY_CATEGORIES.items():
            entries = []
            for ticker in tickers:
                if ticker not in df.columns:
                    continue
                series = df[["date", ticker]].dropna()
                if series.empty:
                    continue
                latest_row = series.iloc[-1]
                latest_close = float(latest_row[ticker])
                cutoff_date = latest_row["date"] - pd.Timedelta(days=30)
                past = series[series["date"] <= cutoff_date]
                change_30d_pct = None
                if not past.empty:
                    past_close = float(past.iloc[-1][ticker])
                    change_30d_pct = ((latest_close - past_close) / past_close * 100.0) if past_close else None
                entries.append({
                    "ticker": ticker,
                    "latest_close": round(latest_close, 4),
                    "latest_date": latest_row["date"].strftime("%Y-%m-%d"),
                    "change_30d_pct": round(change_30d_pct, 2) if change_30d_pct is not None else None,
                })
            result[category] = entries

        return jsonify(result)
    except Exception as exc:
        logger.exception("Commodities endpoint failed")
        return error_response(str(exc), 500)


@app.route("/api/freight_rates", methods=["GET"])
def freight_rates():
    try:
        rates_dir = BASE_DIR / "raw" / "freight_rates"
        frames = []

        scfi_path = rates_dir / "scfi_weekly.csv"
        if scfi_path.exists():
            scfi = pd.read_csv(scfi_path)
            comprehensive = scfi[scfi["route"] == "Comprehensive Index"][["date", "index_value"]]
            comprehensive = comprehensive.rename(columns={"index_value": "scfi_composite"})
            frames.append(comprehensive.set_index("date"))

        drewry_path = rates_dir / "drewry_wci_weekly.csv"
        if drewry_path.exists():
            drewry = pd.read_csv(drewry_path)[["date", "index_value_usd_per_40ft"]]
            drewry = drewry.rename(columns={"index_value_usd_per_40ft": "drewry_wci"})
            frames.append(drewry.set_index("date"))

        air_path = rates_dir / "air_freight_index_composite.csv"
        if air_path.exists():
            air = pd.read_csv(air_path)[["date", "index_value"]]
            air = air.rename(columns={"index_value": "air_freight_composite"})
            frames.append(air.set_index("date"))

        if not frames:
            raise FileNotFoundError(f"No freight rate files found in {rates_dir} — run src/ingestion/fetch_freight_rates.py first")

        merged = pd.concat(frames, axis=1, join="outer").sort_index().reset_index()
        merged = merged.rename(columns={"index": "date"})
        merged = merged.tail(200)
        return jsonify(json_safe(merged.to_dict(orient="records")))
    except Exception as exc:
        logger.exception("Freight rates endpoint failed")
        return error_response(str(exc), 500)


if __name__ == "__main__":
    try:
        load_model_and_feature_structure()
        load_external_data()
        load_reference_data()
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
