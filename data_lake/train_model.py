import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "gold" / "ml_ready.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
TARGET_COLUMN = "target_shipping_cost_usd"


def ensure_dependencies() -> None:
    required = ["scikit-learn", "xgboost", "lightgbm"]
    missing = []

    for package in required:
        try:
            if package == "scikit-learn":
                __import__("sklearn")
            else:
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


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = denominator != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(float)
    X = X.dropna(axis=1, how="all")

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            median_value = X[column].median()
            X[column] = X[column].fillna(0 if pd.isna(median_value) else median_value)
        else:
            X[column] = X[column].replace(r"^\s*$", np.nan, regex=True)
            X[column] = X[column].fillna("Missing")

    X = pd.get_dummies(X, drop_first=False)
    # LightGBM rejects JSON special characters (",", "{", "[", ":", ...) in
    # feature names, which one-hot columns like "origin_city_Shanghai, CN"
    # contain. Sanitize for every model — harmless no-op for the others.
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in X.columns]
    return X, y


def build_models() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6),
        "LightGBM": LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6, verbose=-1),
    }


def fresh_model(name: str):
    return build_models()[name]


def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return compute_metrics(y_test.to_numpy(), predictions)


def main() -> None:
    ensure_dependencies()
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} column not found in the dataset.")

    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {DATA_PATH}\n")

    # --- Step 1: 4 algorithms on the full dataset ---
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    all_results = []
    fitted_models = {}
    for name, model in build_models().items():
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        fitted_models[name] = model
        all_results.append({"scope": "Overall", "model": name, **metrics})
        print(f"[Overall] {name}: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} "
              f"MAPE={metrics['MAPE']:.2f}% R²={metrics['R²']:.4f}")

    best_name = min(fitted_models, key=lambda n: next(r["RMSE"] for r in all_results if r["model"] == n and r["scope"] == "Overall"))
    best_model = fitted_models[best_name]
    best_metrics = next(r for r in all_results if r["model"] == best_name and r["scope"] == "Overall")
    print(f"\nBest overall algorithm: {best_name} (RMSE={best_metrics['RMSE']:.2f})\n")

    # --- Step 2: separate Sea/Air models using the best algorithm ---
    for mode in ["Sea", "Air"]:
        column_name = f"transport_mode_{mode}"
        if column_name not in df.columns:
            print(f"[{mode}] SKIPPED — no '{column_name}' column in the dataset")
            continue

        mode_df = df[df[column_name] == 1].copy()
        if mode_df.empty:
            print(f"[{mode}] SKIPPED — no rows for this mode")
            continue

        X_mode, y_mode = prepare_features(mode_df)
        X_mode = X_mode.reindex(columns=X.columns, fill_value=0)  # keep the same feature space as the overall model
        X_mode_train, X_mode_test, y_mode_train, y_mode_test = train_test_split(
            X_mode, y_mode, test_size=0.2, random_state=42
        )

        mode_model = fresh_model(best_name)
        mode_metrics = evaluate(mode_model, X_mode_train, X_mode_test, y_mode_train, y_mode_test)
        all_results.append({"scope": mode, "model": best_name, **mode_metrics})
        print(f"[{mode}] {best_name} ({len(mode_df)} rows): MAE={mode_metrics['MAE']:.2f} "
              f"RMSE={mode_metrics['RMSE']:.2f} MAPE={mode_metrics['MAPE']:.2f}% R²={mode_metrics['R²']:.4f}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        mode_path = MODEL_DIR / f"best_model_{mode.lower()}.pkl"
        with mode_path.open("wb") as file:
            pickle.dump(mode_model, file)
        print(f"       Saved -> {mode_path}")

    # --- Comparison table ---
    print("\n=== Full Comparison Table ===")
    table = pd.DataFrame(all_results)[["scope", "model", "MAE", "RMSE", "MAPE", "R²"]]
    print(table.to_string(index=False))

    # --- Save best overall model ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(best_model, file)
    print(f"\nBest overall model ({best_name}) saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
