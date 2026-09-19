"""
Generates synthetic CPCB / weather / FIRMS data matching the real API schemas,
so you can build and test the feature engineering + modeling pipeline today,
before your data.gov.in / OpenWeatherMap / FIRMS API keys are approved.

This is NOT for your final model or report — swap it out for real historical
data as soon as you have API access (see README for backfill approach, since
these APIs only give current/recent data, not deep history).

Run: python src/ingestion/generate_synthetic.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CITIES = ["Chennai", "Vellore", "Coimbatore", "Madurai", "Tiruchirappalli"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE"]

# rough baseline pollutant levels per city (Chennai/coastal cities tend lower,
# inland industrial cities like Coimbatore trend higher) — purely illustrative
CITY_BASELINE_PM25 = {
    "Chennai": 55, "Vellore": 62, "Coimbatore": 58, "Madurai": 65, "Tiruchirappalli": 60,
}

rng = np.random.default_rng(seed=42)


def _diurnal_pattern(hour: int) -> float:
    """Traffic-driven double-peak pattern: morning + evening rush hour spikes."""
    morning_peak = np.exp(-((hour - 9) ** 2) / 8)
    evening_peak = np.exp(-((hour - 19) ** 2) / 10)
    return 1.0 + 0.4 * morning_peak + 0.5 * evening_peak


def generate_cpcb_history(days: int = 90, freq_hours: int = 1) -> pd.DataFrame:
    """Hourly synthetic pollutant readings per city, with diurnal + seasonal + noise."""
    timestamps = pd.date_range(end=datetime.now(), periods=days * 24 // freq_hours, freq=f"{freq_hours}h")
    rows = []
    for city in CITIES:
        base = CITY_BASELINE_PM25[city]
        for ts in timestamps:
            seasonal = 1.3 if ts.month in (11, 12, 1) else 1.0
            diurnal = _diurnal_pattern(ts.hour)
            noise = rng.normal(1.0, 0.15)
            pm25 = max(5, base * seasonal * diurnal * noise)

            for pollutant in POLLUTANTS:
                scale = {"PM2.5": 1.0, "PM10": 1.8, "NO2": 0.5, "SO2": 0.2, "CO": 0.05, "OZONE": 0.6}[pollutant]
                value = max(1, pm25 * scale * rng.normal(1.0, 0.2))
                rows.append({
                    "state": "Tamil_Nadu",
                    "city": city,
                    "station": f"{city} - Synthetic Station",
                    "pollutant_id": pollutant,
                    "last_update": ts.isoformat(),
                    "pollutant_min": round(value * 0.85, 1),
                    "pollutant_max": round(value * 1.15, 1),
                    "pollutant_avg": round(value, 1),
                })
    return pd.DataFrame(rows)


def generate_weather_history(days: int = 90, freq_hours: int = 1) -> pd.DataFrame:
    """Hourly synthetic weather matched to the same timestamps as CPCB data."""
    timestamps = pd.date_range(end=datetime.now(), periods=days * 24 // freq_hours, freq=f"{freq_hours}h")
    rows = []
    for city in CITIES:
        for ts in timestamps:
            seasonal_temp = 28 + 5 * np.sin(2 * np.pi * (ts.dayofyear / 365))
            temp = seasonal_temp + 4 * np.sin(2 * np.pi * (ts.hour / 24)) + rng.normal(0, 1.5)
            wind_speed = max(0.2, rng.normal(3.5, 1.8))
            rows.append({
                "city": city,
                "fetched_at": ts.isoformat(),
                "temp_c": round(temp, 1),
                "humidity_pct": round(np.clip(rng.normal(65, 12), 20, 98), 1),
                "pressure_hpa": round(rng.normal(1011, 4), 1),
                "wind_speed_ms": round(wind_speed, 2),
                "wind_deg": round(rng.uniform(0, 360), 1),
                "clouds_pct": round(np.clip(rng.normal(40, 25), 0, 100), 1),
            })
    return pd.DataFrame(rows)


def generate_firms_history(days: int = 90) -> pd.DataFrame:
    """Sparse synthetic fire detections, weighted toward Nov-Jan (stubble burning season)."""
    rows = []
    start = datetime.now() - timedelta(days=days)
    for d in range(days):
        date = start + timedelta(days=d)
        season_multiplier = 8 if date.month in (11, 12, 1) else 1
        n_fires = rng.poisson(1.5 * season_multiplier)
        for _ in range(n_fires):
            rows.append({
                "latitude": round(rng.uniform(8.0, 14.0), 4),
                "longitude": round(rng.uniform(76.0, 81.5), 4),
                "brightness": round(rng.normal(320, 15), 1),
                "confidence": rng.choice(["low", "nominal", "high"], p=[0.2, 0.5, 0.3]),
                "acq_date": date.strftime("%Y-%m-%d"),
                "frp": round(max(1, rng.normal(15, 8)), 1),
                "fetched_at": datetime.now().isoformat(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)

    print("Generating synthetic CPCB data...")
    cpcb = generate_cpcb_history(days=90)
    cpcb.to_csv("data/raw/cpcb_history.csv", index=False)
    print(f"  {len(cpcb)} rows -> data/raw/cpcb_history.csv")

    print("Generating synthetic weather data...")
    weather = generate_weather_history(days=90)
    weather.to_csv("data/raw/weather_history.csv", index=False)
    print(f"  {len(weather)} rows -> data/raw/weather_history.csv")

    print("Generating synthetic FIRMS data...")
    firms = generate_firms_history(days=90)
    firms.to_csv("data/raw/firms_history.csv", index=False)
    print(f"  {len(firms)} rows -> data/raw/firms_history.csv")

    print("\nDone. This synthetic data is for pipeline development only —")
    print("swap in real API data as soon as your keys are approved.")
