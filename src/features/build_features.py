"""
Feature engineering pipeline: joins CPCB pollutant readings with weather and
fire data, then builds the lag/rolling/calendar features the forecast model
trains on.

Input:  data/raw/cpcb_history.csv, data/raw/weather_history.csv, data/raw/firms_history.csv
Output: data/processed/features.csv  (one row per city per hour)

Run: python src/features/build_features.py
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

LAG_HOURS = [1, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [6, 24, 72]
FORECAST_HORIZONS_HOURS = [24, 48, 72]


def load_and_pivot_cpcb(path: str = f"{RAW_DIR}/cpcb_history.csv") -> pd.DataFrame:
    """CPCB data is one row per pollutant per station — pivot to one row per city per hour."""
    df = pd.read_csv(path, parse_dates=["last_update"])
    df["last_update"] = df["last_update"].dt.floor("h")

    pivoted = df.pivot_table(
        index=["city", "last_update"],
        columns="pollutant_id",
        values="pollutant_avg",
        aggfunc="mean",
    ).reset_index()
    pivoted.columns.name = None

    # composite AQI proxy: in a real submission, use the official CPCB sub-index
    # breakpoint formula (piecewise linear per pollutant) rather than this mean —
    # this is a placeholder so the pipeline runs end-to-end
    pollutant_cols = [c for c in pivoted.columns if c not in ("city", "last_update")]
    pivoted["aqi_proxy"] = pivoted[pollutant_cols].mean(axis=1)

    return pivoted.rename(columns={"last_update": "timestamp"})


def load_weather(path: str = f"{RAW_DIR}/weather_history.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["fetched_at"])
    df["fetched_at"] = df["fetched_at"].dt.floor("h")
    return df.rename(columns={"fetched_at": "timestamp"})


def load_firms_daily_counts(path: str = f"{RAW_DIR}/firms_history.csv") -> pd.DataFrame:
    """Aggregate fire detections to a daily count + mean FRP TN-wide.

    NOTE: simplification. A stronger version would use dist_to_<city>_km
    (from firms_client.distance_to_nearest_city) plus wind direction to
    build a directional "is this fire actually upwind" feature.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "fire_count", "fire_mean_frp"])
    df = pd.read_csv(path, parse_dates=["acq_date"])
    daily = df.groupby("acq_date").agg(
        fire_count=("frp", "count"),
        fire_mean_frp=("frp", "mean"),
    ).reset_index().rename(columns={"acq_date": "date"})
    return daily


def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "aqi_proxy") -> pd.DataFrame:
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("city")[target_col]

    for lag in LAG_HOURS:
        df[f"{target_col}_lag_{lag}h"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"{target_col}_roll_mean_{window}h"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=max(1, window // 3)).mean()
        )
        df[f"{target_col}_roll_std_{window}h"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=max(1, window // 3)).std()
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_forecast_targets(df: pd.DataFrame, target_col: str = "aqi_proxy") -> pd.DataFrame:
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    for h in FORECAST_HORIZONS_HOURS:
        df[f"target_{h}h"] = df.groupby("city")[target_col].shift(-h)
    return df


def build_feature_table() -> pd.DataFrame:
    cpcb = load_and_pivot_cpcb()
    weather = load_weather()
    firms_daily = load_firms_daily_counts()

    merged = pd.merge(cpcb, weather, on=["city", "timestamp"], how="inner")

    if not firms_daily.empty:
        merged["date"] = merged["timestamp"].dt.normalize()
        merged = pd.merge(merged, firms_daily, on="date", how="left")
        merged["fire_count"] = merged["fire_count"].fillna(0)
        merged["fire_mean_frp"] = merged["fire_mean_frp"].fillna(0)
        merged = merged.drop(columns=["date"])

    merged = add_lag_and_rolling_features(merged)
    merged = add_calendar_features(merged)
    merged = add_forecast_targets(merged)

    return merged


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    features = build_feature_table()
    out_path = f"{OUT_DIR}/features.csv"
    features.to_csv(out_path, index=False)
    print(f"Built {len(features)} rows x {len(features.columns)} columns -> {out_path}")
    print(f"\nColumns: {list(features.columns)}")
    print(f"\nRows with all 3 targets populated (usable for training): "
          f"{features.dropna(subset=[c for c in features.columns if c.startswith('target_')]).shape[0]}")
