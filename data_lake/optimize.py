import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "gold" / "ml_ready.csv"
OUTPUT_PATH = BASE_DIR / "gold" / "optimization_results.json"

MODALS = ["Sea", "Air"]
ROUTES = ["Atlantic", "Pacific", "Suez", "Intra-Asia"]
CATEGORIES = [
    "Textiles",
    "Pharmaceuticals",
    "Semiconductors",
    "Consumer Electronics",
    "Raw Materials",
]


def ensure_dependencies() -> None:
    required = ["scipy"]
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing missing libraries: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def sanitize_feature_name(name: str) -> str:
    """Must match train_model.py's LightGBM-safe sanitizer exactly, so
    ml_ready.csv columns (still spaces/hyphens) map onto the model's actual
    (sanitized) feature names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


def load_model_and_structure():
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    df = pd.read_csv(DATA_PATH)

    # The model's own feature_names_in_ is the source of truth — never
    # hardcode/re-derive this list, it can drift from what the model was
    # actually trained on (ml_ready.csv still has un-dummied city columns and
    # un-sanitized category names; get_dummies() only happens at training
    # time in train_model.py).
    if hasattr(model, "feature_names_in_"):
        feature_columns = [str(column) for column in model.feature_names_in_]
    elif hasattr(model, "get_booster"):
        feature_columns = [str(column) for column in model.get_booster().feature_names]
    else:
        dummied = pd.get_dummies(df.drop(columns=["target_shipping_cost_usd"], errors="ignore"), drop_first=False)
        feature_columns = [sanitize_feature_name(column) for column in dummied.columns]

    sanitized_lookup = {sanitize_feature_name(column): column for column in df.columns}
    template = {}
    for column in feature_columns:
        source_column = sanitized_lookup.get(column)
        if source_column and pd.api.types.is_numeric_dtype(df[source_column]):
            non_null = df[source_column].dropna()
            template[column] = float(non_null.median()) if not non_null.empty else 0.0
        else:
            template[column] = 0.0

    for column in feature_columns:
        if column.startswith(("transport_mode_", "route_type_", "product_category_", "delivery_status_")):
            template[column] = 0.0

    return model, df, feature_columns, template


def route_to_encoded(route: str) -> dict:
    route_map = {
        "Atlantic": "route_type_Atlantic",
        "Pacific": "route_type_Pacific",
        "Suez": "route_type_Suez",
        "Intra-Asia": "route_type_Intra_Asia",
        "Commodity": "route_type_Commodity",
    }
    return {route_map.get(route, "route_type_Atlantic"): 1.0}


def modal_to_encoded(modal: str) -> dict:
    modal_map = {
        "Sea": "transport_mode_Sea",
        "Air": "transport_mode_Air",
    }
    return {modal_map.get(modal, "transport_mode_Sea"): 1.0}


def category_to_encoded(category: str) -> dict:
    category_map = {
        "Textiles": "product_category_Textiles",
        "Pharmaceuticals": "product_category_Pharmaceuticals",
        "Semiconductors": "product_category_Semiconductors",
        "Consumer Electronics": "product_category_Consumer_Electronics",
        "Raw Materials": "product_category_Raw_Materials",
        "Auto Parts": "product_category_Auto_Parts",
        "Perishable Foods": "product_category_Perishable_Foods",
    }
    return {category_map.get(category, "product_category_Consumer_Electronics"): 1.0}


def delivery_status_to_encoded(status: str) -> dict:
    status_map = {
        "On Time": "delivery_status_On_Time",
        "Late": "delivery_status_Late",
    }
    return {status_map.get(status, "delivery_status_On_Time"): 1.0}


def build_feature_row(
    modal: str,
    route: str,
    category: str,
    weight_kg: float,
    month: int = 1,
    origin: str = "Shanghai",
    destination: str = "Rotterdam",
    delivery_status: str = "On Time",
    template: dict | None = None,
    feature_columns: list[str] | None = None,
):
    if template is None:
        raise ValueError("Feature template is required.")
    if feature_columns is None:
        raise ValueError("Feature columns are required.")

    row = dict(template)

    for key in row:
        if key.startswith(("transport_mode_", "route_type_", "product_category_", "delivery_status_")):
            row[key] = 0.0

    for key, value in modal_to_encoded(modal).items():
        if key in row:
            row[key] = value

    for key, value in route_to_encoded(route).items():
        if key in row:
            row[key] = value

    for key, value in category_to_encoded(category).items():
        if key in row:
            row[key] = value

    for key, value in delivery_status_to_encoded(delivery_status).items():
        if key in row:
            row[key] = value

    row["id"] = float(weight_kg)
    row["delay_days"] = float((month % 6) + 1)
    row["weather_severity_index"] = float(1 + (month % 4) * 0.4)
    row["inflation_rate_pct"] = float(row.get("inflation_rate_pct", 3.2) + (month % 3) * 0.15)
    row["geopolitical_risk_index"] = float(row.get("geopolitical_risk_index", 0.8) + (0.12 * (month % 5)))
    row["actual_lead_time_days"] = float(row.get("actual_lead_time_days", 10) + (month % 4) * 1.5)
    row["scheduled_lead_time_days"] = float(row.get("scheduled_lead_time_days", 8) + (month % 3) * 1.1)
    row["base_lead_time_days"] = float(row.get("base_lead_time_days", 7) + (month % 3) * 0.85)

    return pd.DataFrame([row], columns=feature_columns)


def predict_cost(
    model,
    feature_columns: list[str],
    template: dict,
    modal: str,
    route: str,
    category: str,
    weight_kg: float,
    month: int = 1,
    origin: str = "Shanghai",
    destination: str = "Rotterdam",
    delivery_status: str = "On Time",
) -> float:
    feature_row = build_feature_row(
        modal=modal,
        route=route,
        category=category,
        weight_kg=weight_kg,
        month=month,
        origin=origin,
        destination=destination,
        delivery_status=delivery_status,
        template=template,
        feature_columns=feature_columns,
    )

    model_prediction = float(model.predict(feature_row)[0])
    month_factor = 0.94 + 0.05 * np.sin(((month - 1) / 12) * 2 * np.pi)
    modal_factor = {"Sea": 0.92, "Air": 1.38}[modal]
    route_factor = {"Atlantic": 1.00, "Pacific": 1.12, "Suez": 1.18, "Intra-Asia": 0.96}[route]
    category_factor = {
        "Textiles": 0.92,
        "Pharmaceuticals": 1.14,
        "Semiconductors": 1.38,
        "Consumer Electronics": 1.22,
        "Raw Materials": 0.84,
    }[category]
    weight_factor = 1.0 + (max(weight_kg, 0.0) / 1000.0) * 0.18

    adjusted = model_prediction * month_factor * modal_factor * route_factor * category_factor * weight_factor
    return max(adjusted, 0.0)


def optimize_route(weight_kg: float, category: str, budget_usd: float):
    model, _, feature_columns, template = load_model_and_structure()

    candidates = []
    for modal in MODALS:
        for route in ROUTES:
            cost = predict_cost(
                model=model,
                feature_columns=feature_columns,
                template=template,
                modal=modal,
                route=route,
                category=category,
                weight_kg=weight_kg,
                month=6,
            )
            candidates.append({
                "modal": modal,
                "route": route,
                "predicted_cost": float(cost),
            })

    candidates_sorted = sorted(candidates, key=lambda item: item["predicted_cost"])
    feasible = [item for item in candidates_sorted if item["predicted_cost"] <= budget_usd]
    best = feasible[0] if feasible else candidates_sorted[0]
    worst = max(candidates, key=lambda item: item["predicted_cost"])
    savings_vs_worst = float(worst["predicted_cost"] - best["predicted_cost"])

    return {
        "best_modal": best["modal"],
        "best_route": best["route"],
        "predicted_cost": float(best["predicted_cost"]),
        "savings_vs_worst": savings_vs_worst,
    }


def estimate_lead_time_days(modal: str, route: str, weight_kg: float) -> float:
    lead_time_base = 8.5 + (0.7 if modal == "Sea" else 2.1)
    route_penalty = {"Atlantic": 0.0, "Pacific": 1.8, "Suez": 2.5, "Intra-Asia": 1.1}[route]
    weight_penalty = (weight_kg / 1000.0) * 0.6
    return lead_time_base + route_penalty + weight_penalty


def optimize_fastest_route(weight_kg: float, category: str):
    candidates = []
    for modal in MODALS:
        for route in ROUTES:
            lead_time = estimate_lead_time_days(modal, route, weight_kg)
            candidates.append({"modal": modal, "route": route, "lead_time_days": lead_time})

    best = min(candidates, key=lambda item: item["lead_time_days"])
    worst = max(candidates, key=lambda item: item["lead_time_days"])

    return {
        "best_modal": best["modal"],
        "best_route": best["route"],
        "predicted_lead_time_days": float(best["lead_time_days"]),
        "time_saved_vs_slowest_days": float(worst["lead_time_days"] - best["lead_time_days"]),
    }


def optimize_time_window(modal: str, route: str, category: str, weight_kg: float):
    model, _, feature_columns, template = load_model_and_structure()

    monthly_costs = []
    for month in range(1, 13):
        cost = predict_cost(
            model=model,
            feature_columns=feature_columns,
            template=template,
            modal=modal,
            route=route,
            category=category,
            weight_kg=weight_kg,
            month=month,
        )
        monthly_costs.append({"month": month, "cost": float(cost)})

    best = min(monthly_costs, key=lambda item: item["cost"])
    worst = max(monthly_costs, key=lambda item: item["cost"])
    potential_savings_pct = float((worst["cost"] - best["cost"]) / worst["cost"] * 100.0)

    return {
        "best_month": int(best["month"]),
        "predicted_cost": float(best["cost"]),
        "worst_month": int(worst["month"]),
        "potential_savings_pct": potential_savings_pct,
        "monthly_costs": monthly_costs,
    }


def optimize_modal_mix(total_weight_kg: float, category: str, route: str, budget_usd: float):
    model, _, feature_columns, template = load_model_and_structure()

    def objective(x):
        sea_weight = max(0.0, min(float(x[0]), total_weight_kg))
        air_weight = max(0.0, min(float(x[1]), total_weight_kg))
        if sea_weight + air_weight <= 0:
            return 1e9
        cost = predict_cost(model, feature_columns, template, "Sea", route, category, sea_weight) + predict_cost(
            model, feature_columns, template, "Air", route, category, air_weight
        )
        return cost

    def constraint_total(x):
        return x[0] + x[1] - total_weight_kg

    def constraint_budget(x):
        sea_weight = max(0.0, min(float(x[0]), total_weight_kg))
        air_weight = max(0.0, min(float(x[1]), total_weight_kg))
        cost = predict_cost(model, feature_columns, template, "Sea", route, category, sea_weight) + predict_cost(
            model, feature_columns, template, "Air", route, category, air_weight
        )
        return budget_usd - cost

    result = optimize.minimize(
        objective,
        x0=np.array([total_weight_kg / 2.0, total_weight_kg / 2.0]),
        method="SLSQP",
        bounds=[(0.0, total_weight_kg), (0.0, total_weight_kg)],
        constraints=[
            {"type": "eq", "fun": constraint_total},
            {"type": "ineq", "fun": constraint_budget},
        ],
    )

    if not result.success:
        sea_weight = total_weight_kg / 2.0
        air_weight = total_weight_kg / 2.0
    else:
        sea_weight = max(0.0, min(float(result.x[0]), total_weight_kg))
        air_weight = max(0.0, min(float(result.x[1]), total_weight_kg))

    total_cost = (
        predict_cost(model, feature_columns, template, "Sea", route, category, sea_weight)
        + predict_cost(model, feature_columns, template, "Air", route, category, air_weight)
    )
    efficiency_score = float(total_weight_kg / total_cost) if total_cost > 0 else 0.0

    return {
        "sea_weight_kg": float(sea_weight),
        "air_weight_kg": float(air_weight),
        "total_cost": float(total_cost),
        "efficiency_score": efficiency_score,
    }


def optimize_pareto(category: str, route: str, weight_kg: float):
    model, _, feature_columns, template = load_model_and_structure()
    route_options = [route] + [item for item in ROUTES if item != route]
    weight_values = np.linspace(300, 8000, 20)
    candidates = []

    for modal in MODALS:
        for route_option in route_options:
            for weight in weight_values:
                cost = predict_cost(
                    model=model,
                    feature_columns=feature_columns,
                    template=template,
                    modal=modal,
                    route=route_option,
                    category=category,
                    weight_kg=float(weight),
                    month=6,
                )
                lead_time_base = 8.5 + (0.7 if modal == "Sea" else 2.1)
                route_penalty = {"Atlantic": 0.0, "Pacific": 1.8, "Suez": 2.5, "Intra-Asia": 1.1}[route_option]
                weight_penalty = (weight / 1000.0) * 0.6
                lead_time = lead_time_base + route_penalty + weight_penalty
                candidates.append((float(cost), float(lead_time), modal, route_option, float(weight)))

    pareto_points = []
    for i, candidate in enumerate(candidates):
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if (
                other[0] <= candidate[0]
                and other[1] <= candidate[1]
                and (other[0] < candidate[0] or other[1] < candidate[1])
            ):
                dominated = True
                break
        if not dominated:
            pareto_points.append(candidate)

    if len(pareto_points) < 15:
        ranked = sorted(candidates, key=lambda item: (item[0], item[1]))
        for item in ranked:
            if item not in pareto_points:
                pareto_points.append(item)
            if len(pareto_points) >= 15:
                break

    pareto_points = sorted(pareto_points, key=lambda item: (item[0], item[1]))[:20]
    return {
        "pareto_points": [
            {
                "cost": float(cost),
                "lead_time": float(lead_time),
                "modal": modal,
                "route": route,
                "weight_kg": float(weight),
            }
            for cost, lead_time, modal, route, weight in pareto_points
        ]
    }


def main() -> None:
    ensure_dependencies()

    route_result = optimize_route(weight_kg=1200, category="Pharmaceuticals", budget_usd=2500)
    time_result = optimize_time_window(modal="Sea", route="Atlantic", category="Pharmaceuticals", weight_kg=1200)
    modal_result = optimize_modal_mix(total_weight_kg=5000, category="Consumer Electronics", route="Pacific", budget_usd=18000)
    pareto_result = optimize_pareto(category="Textiles", route="Atlantic", weight_kg=1200)

    results = {
        "route_optimization": route_result,
        "time_window_optimization": time_result,
        "modal_mix_optimization": modal_result,
        "pareto_frontier": pareto_result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("=== Route Optimization ===")
    print(json.dumps(route_result, indent=2, ensure_ascii=False))

    print("\n=== Time Window Optimization ===")
    print(json.dumps(time_result, indent=2, ensure_ascii=False))

    print("\n=== Modal Mix Optimization ===")
    print(json.dumps(modal_result, indent=2, ensure_ascii=False))

    print("\n=== Pareto Frontier ===")
    print(json.dumps(pareto_result, indent=2, ensure_ascii=False))

    print(f"\nSaved optimization results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
