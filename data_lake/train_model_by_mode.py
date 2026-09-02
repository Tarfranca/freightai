import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "gold" / "ml_ready.csv"
MODEL_DIR = BASE_DIR / "models"
MODE_ORDER = ["Sea", "Air", "Road", "Rail"]
TARGET_COLUMN = "target_shipping_cost_usd"


def ensure_dependencies() -> None:
    required = ["pandas", "scikit-learn", "xgboost"]
    missing = []

    for package in required:
        try:
            if package == "scikit-learn":
                __import__("sklearn")
            elif package == "xgboost":
                __import__("xgboost")
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
    mask = np.abs(y_true) != 0
    if not np.any(mask):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
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

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            median_value = X[column].median()
            X[column] = X[column].fillna(0 if pd.isna(median_value) else median_value)
        else:
            X[column] = X[column].replace(r"^\s*$", np.nan, regex=True)
            X[column] = X[column].fillna("Missing")

    X = pd.get_dummies(X, drop_first=False)
    return X, y


def train_mode_models(mode_name: str, mode_df: pd.DataFrame) -> tuple[dict, object, pd.DataFrame]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    X, y = prepare_features(mode_df)

    if X.empty or y.empty:
        raise ValueError(f"No usable data for mode: {mode_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6),
    }

    mode_metrics = []
    best_model_name = None
    best_model = None
    best_metrics = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test.to_numpy(), predictions)
        mode_metrics.append({
            "mode": mode_name,
            "model": model_name,
            **metrics,
        })

        if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
            best_model_name = model_name
            best_model = model
            best_metrics = metrics

    summary_df = pd.DataFrame(mode_metrics)
    summary_df = summary_df[["mode", "model", "MAE", "RMSE", "MAPE", "R²"]]
    return summary_df, best_model, best_metrics


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def main() -> None:
    ensure_dependencies()

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {DATA_PATH}.")

    all_summary_rows = []

    for mode_name in MODE_ORDER:
        column_name = f"transport_mode_{mode_name}"
        if column_name not in df.columns:
            print(f"\nMode: {mode_name}")
            print(f"No '{column_name}' column found in dataset; saving a fallback model for this mode.")

            from sklearn.dummy import DummyRegressor
            from sklearn.model_selection import train_test_split

            X_all, y_all = prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            fallback_model = DummyRegressor(strategy="mean")
            fallback_model.fit(X_train, y_train)
            fallback_predictions = fallback_model.predict(X_test)
            fallback_metrics = compute_metrics(y_test.to_numpy(), fallback_predictions)
            save_model(fallback_model, MODEL_DIR / f"best_model_{mode_name.lower()}.pkl")

            print(f"Fallback model saved to: {MODEL_DIR / f'best_model_{mode_name.lower()}.pkl'}")
            for metric_name, value in fallback_metrics.items():
                print(f"{metric_name}: {value:.6f}")

            all_summary_rows.append({
                "mode": mode_name,
                "model": "Fallback",
                **fallback_metrics,
            })
            continue

        mode_df = df[df[column_name] == 1].copy()
        if mode_df.empty:
            print(f"\nMode: {mode_name}")
            print(f"No rows for mode '{mode_name}'. Saving fallback model.")
            fallback_model = __import__("sklearn.dummy", fromlist=["DummyRegressor"]).DummyRegressor(strategy="mean")
            X_all, y_all = prepare_features(df)
            X_train, X_test, y_train, y_test = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
                X_all, y_all, test_size=0.2, random_state=42
            )
            fallback_model.fit(X_train, y_train)
            fallback_predictions = fallback_model.predict(X_test)
            fallback_metrics = compute_metrics(y_test.to_numpy(), fallback_predictions)
            save_model(fallback_model, MODEL_DIR / f"best_model_{mode_name.lower()}.pkl")
            all_summary_rows.append({
                "mode": mode_name,
                "model": "Fallback",
                **fallback_metrics,
            })
            print(f"Fallback model saved to: {MODEL_DIR / f'best_model_{mode_name.lower()}.pkl'}")
            for metric_name, value in fallback_metrics.items():
                print(f"{metric_name}: {value:.6f}")
            continue

        print(f"\n=== Transport mode: {mode_name} ===")
        summary_df, best_model, best_metrics = train_mode_models(mode_name, mode_df)
        all_summary_rows.extend(summary_df.to_dict("records"))

        for _, row in summary_df.iterrows():
            print(f"Model: {row['model']}")
            print(f"MAE: {row['MAE']:.6f}")
            print(f"RMSE: {row['RMSE']:.6f}")
            print(f"MAPE: {row['MAPE']:.6f}")
            print(f"R²: {row['R²']:.6f}")

        save_model(best_model, MODEL_DIR / f"best_model_{mode_name.lower()}.pkl")
        print(f"Best model for {mode_name}: {best_model.__class__.__name__}")
        print(f"Saved to: {MODEL_DIR / f'best_model_{mode_name.lower()}.pkl'}")

    print("\n=== Final Summary Table ===")
    summary_table = pd.DataFrame(all_summary_rows)
    if summary_table.empty:
        print("No mode data was available for training.")
        return

    summary_table = summary_table[["mode", "model", "MAE", "RMSE", "MAPE", "R²"]]
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
