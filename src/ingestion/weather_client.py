"""
OpenWeatherMap client for weather features that drive pollutant dispersion:
wind speed/direction (transport), humidity (particle formation),
temperature (inversions trap pollutants near ground), pressure (stagnation).

Free tier: https://openweathermap.org/api (Current Weather + 5-day/3-hour forecast
are both on the free plan; the paid One Call 3.0 gives hourly 48h forecasts
if you want to extend past the free tier later).

Set OPENWEATHERMAP_API_KEY as an environment variable before running.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone

BASE_URL = "https://api.openweathermap.org/data/2.5"

# lat/lon for tracked cities — CPCB station coords should be matched to these
# for the join step in feature engineering
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Vellore": (12.9165, 79.1325),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
}


def fetch_current_weather(city: str) -> dict:
    """Fetch current weather for one city. Returns a flat dict ready for a DataFrame row."""
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set OPENWEATHERMAP_API_KEY. Get a free key at https://openweathermap.org/api"
        )
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Add coords to CITY_COORDS first.")

    lat, lon = CITY_COORDS[city]
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    resp = requests.get(f"{BASE_URL}/weather", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return {
        "city": city,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "temp_c": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"],
        "pressure_hpa": data["main"]["pressure"],
        "wind_speed_ms": data["wind"].get("speed"),
        "wind_deg": data["wind"].get("deg"),
        "clouds_pct": data.get("clouds", {}).get("all"),
    }


def fetch_forecast(city: str) -> pd.DataFrame:
    """5-day / 3-hour forecast — useful as a feature for the forecast horizon itself."""
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENWEATHERMAP_API_KEY")
    lat, lon = CITY_COORDS[city]
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    resp = requests.get(f"{BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    rows = []
    for entry in resp.json().get("list", []):
        rows.append({
            "city": city,
            "forecast_time": entry["dt_txt"],
            "temp_c": entry["main"]["temp"],
            "humidity_pct": entry["main"]["humidity"],
            "pressure_hpa": entry["main"]["pressure"],
            "wind_speed_ms": entry["wind"].get("speed"),
            "wind_deg": entry["wind"].get("deg"),
        })
    return pd.DataFrame(rows)


def fetch_all_cities_current() -> pd.DataFrame:
    rows = [fetch_current_weather(city) for city in CITY_COORDS]
    return pd.DataFrame(rows)


def append_to_history(df: pd.DataFrame, path: str = "data/raw/weather_history.csv") -> None:
    if df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    data = fetch_all_cities_current()
    print(f"Fetched weather for {len(data)} cities")
    append_to_history(data)
