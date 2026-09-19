"""
Inference: loads the trained models and produces a forecast for the most
recent available row per city, with prediction intervals and per-prediction
SHAP drivers (the "why is it spiking" explanation the dashboard shows).

This is separate from train_forecast.py on purpose — the dashboard should
never trigger training, only read saved artifacts.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
FEATURES_PATH = "data/processed/features.csv"
HORIZONS = [24, 48, 72]

# human-readable names for the feature columns, so the dashboard shows
# "wind speed" rather than "wind_speed_ms" in the explanation panel
FEATURE_LABELS = {
    "wind_speed_ms": "wind speed",
    "wind_deg": "wind direction",
    "temp_c": "temperature",
    "humidity_pct": "humidity",
    "pressure_hpa": "air pressure",
    "clouds_pct": "cloud cover",
    "fire_count": "regional fire activity",
    "fire_mean_frp": "fire intensity",
    "hour": "time of day",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "day_of_week": "day of week",
    "is_weekend": "weekend effect",
    "month": "season",
}


def _pretty(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("aqi_proxy_lag_"):
        return f"AQI {feature.split('_')[-1]} ago"
    if "roll_mean" in feature:
        return f"AQI {feature.split('_')[-1]} average"
    if "roll_std" in feature:
        return f"AQI {feature.split('_')[-1]} volatility"
    return feature.replace("_", " ")


def load_results() -> dict:
    with open(f"{MODEL_DIR}/results.json") as f:
        return json.load(f)


def load_models(horizon: int):
    """Returns (point_model, p10_model, p90_model) as raw LightGBM Boosters."""
    point = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm_{horizon}h.txt")
    p10 = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm_{horizon}h_q10.txt")
    p90 = lgb.Booster(model_file=f"{MODEL_DIR}/lgbm_{horizon}h_q90.txt")
    return point, p10, p90


def latest_row_per_city(features_path: str = FEATURES_PATH) -> pd.DataFrame:
    """Most recent fully-populated feature row for each city — what we forecast from."""
    df = pd.read_csv(features_path, parse_dates=["timestamp"])
    # drop rows where lag features are still NaN (start of each city's series)
    lag_cols = [c for c in df.columns if "_lag_" in c or "roll_" in c]
    df = df.dropna(subset=lag_cols)
    return df.sort_values("timestamp").groupby("city").tail(1).reset_index(drop=True)


def forecast_city(row: pd.Series, results: dict) -> list:
    """
    Produce a forecast per horizon for one city, each with a prediction
    interval and the top SHAP drivers for that specific prediction.
    """
    out = []
    for horizon in HORIZONS:
        meta = results[f"{horizon}h"]
        feature_cols = meta["feature_cols"]
        X = row[feature_cols].to_frame().T.astype(float)

        point, p10, p90 = load_models(horizon)
        pred = float(point.predict(X)[0])
        lo = float(p10.predict(X)[0])
        hi = float(p90.predict(X)[0])

        # per-prediction SHAP: why THIS forecast, not global importance
        explainer = shap.TreeExplainer(point)
        shap_vals = explainer.shap_values(X)[0]
        ranked = sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))[:5]
        drivers = [
            {
                "feature": _pretty(f),
                "raw_feature": f,
                "shap": round(float(v), 2),
                "direction": "raises" if v > 0 else "lowers",
                "value": round(float(row[f]), 2) if pd.notna(row[f]) else None,
            }
            for f, v in ranked
        ]

        out.append({
            "horizon_h": horizon,
            "prediction": round(pred, 1),
            "lower": round(min(lo, hi), 1),
            "upper": round(max(lo, hi), 1),
            "interval_coverage": meta.get("interval_coverage"),
            "drivers": drivers,
        })
    return out


def forecast_all_cities() -> dict:
    results = load_results()
    latest = latest_row_per_city()
    return {
        row["city"]: {
            "as_of": row["timestamp"],
            "current_aqi": round(float(row["aqi_proxy"]), 1),
            "forecasts": forecast_city(row, results),
        }
        for _, row in latest.iterrows()
    }


if __name__ == "__main__":
    all_fc = forecast_all_cities()
    for city, data in all_fc.items():
        print(f"\n{city} (as of {data['as_of']}) — current {data['current_aqi']}")
        for fc in data["forecasts"]:
            print(f"  +{fc['horizon_h']}h: {fc['prediction']} [{fc['lower']}–{fc['upper']}]")
            top = fc["drivers"][0]
            print(f"     top driver: {top['feature']} ({top['direction']} it)")
