"""
Trains and evaluates forecast models for each horizon (24h/48h/72h):
  1. Naive persistence baseline (tomorrow = today) — the bar any real model must clear
  2. LightGBM regressor per horizon, with SHAP explainability

Uses a time-aware train/test split (NOT random shuffle — that would leak
future information into training, since consecutive hours are correlated).

Run: python src/models/train_forecast.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "src/models/artifacts"
HORIZONS = [24, 48, 72]

NON_FEATURE_COLS = ["city", "timestamp"] + [f"target_{h}h" for h in HORIZONS]


def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def naive_persistence_baseline(test: pd.DataFrame, horizon: int) -> dict:
    valid = test.dropna(subset=[f"target_{horizon}h", "aqi_proxy"])
    y_true = valid[f"target_{horizon}h"]
    y_pred = valid["aqi_proxy"]
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def train_lightgbm(train: pd.DataFrame, test: pd.DataFrame, horizon: int):
    target_col = f"target_{horizon}h"
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]

    train_valid = train.dropna(subset=[target_col])
    test_valid = test.dropna(subset=[target_col])

    X_train, y_train = train_valid[feature_cols], train_valid[target_col]
    X_test, y_test = test_valid[feature_cols], test_valid[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
    }
    return model, metrics, feature_cols, X_test


def train_quantile_models(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                          quantiles=(0.1, 0.9)) -> dict:
    """
    Train quantile regressors so the dashboard can show a real prediction
    interval instead of a made-up error band. p10/p90 gives an 80% interval.

    These are separate models from the point forecast — LightGBM's quantile
    objective optimizes pinball loss, not MSE, so you can't get intervals
    out of the main model for free.
    """
    target_col = f"target_{horizon}h"
    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]

    train_valid = train.dropna(subset=[target_col])
    test_valid = test.dropna(subset=[target_col])
    X_train, y_train = train_valid[feature_cols], train_valid[target_col]
    X_test, y_test = test_valid[feature_cols], test_valid[target_col]

    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        models[q] = m

    # empirical coverage: what fraction of actuals actually fall in the interval.
    # if this is far from the nominal 80%, the intervals are miscalibrated and
    # shouldn't be presented as confidence bands without saying so.
    lo = models[quantiles[0]].predict(X_test)
    hi = models[quantiles[1]].predict(X_test)
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())

    return {"models": models, "coverage": coverage,
            "nominal": quantiles[1] - quantiles[0]}


def compute_shap_summary(model, X_test: pd.DataFrame, top_n: int = 10) -> list:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(500, len(X_test)), random_state=42))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X_test.columns, mean_abs_shap), key=lambda x: -x[1])
    return [{"feature": f, "mean_abs_shap": round(float(v), 3)} for f, v in ranked[:top_n]]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    train, test = time_aware_split(df)

    print(f"Train: {len(train)} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
    print(f"Test:  {len(test)} rows ({test['timestamp'].min()} to {test['timestamp'].max()})\n")

    results = {}
    for horizon in HORIZONS:
        print(f"--- {horizon}h horizon ---")

        baseline_metrics = naive_persistence_baseline(test, horizon)
        print(f"  Naive baseline   MAE={baseline_metrics['mae']:.2f}  RMSE={baseline_metrics['rmse']:.2f}")

        model, model_metrics, feature_cols, X_test = train_lightgbm(train, test, horizon)
        print(f"  LightGBM         MAE={model_metrics['mae']:.2f}  RMSE={model_metrics['rmse']:.2f}")

        improvement = (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
        print(f"  Improvement over baseline: {improvement:.1f}%\n")

        quantile_result = train_quantile_models(train, test, horizon)
        print(f"  Interval coverage: {quantile_result['coverage']:.1%} "
              f"(nominal {quantile_result['nominal']:.0%})")

        top_features = compute_shap_summary(model, X_test)

        model.booster_.save_model(f"{MODEL_DIR}/lgbm_{horizon}h.txt")
        for q, qm in quantile_result["models"].items():
            qm.booster_.save_model(f"{MODEL_DIR}/lgbm_{horizon}h_q{int(q * 100)}.txt")

        # feature order must be saved — LightGBM boosters take a raw array at
        # predict time, so the dashboard has to rebuild columns in this order
        results[f"{horizon}h"] = {
            "baseline": baseline_metrics,
            "model": model_metrics,
            "improvement_pct": improvement,
            "interval_coverage": quantile_result["coverage"],
            "interval_nominal": quantile_result["nominal"],
            "top_features": top_features,
            "feature_cols": feature_cols,
        }

    with open(f"{MODEL_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved models and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
