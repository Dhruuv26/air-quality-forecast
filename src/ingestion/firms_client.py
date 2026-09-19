"""
NASA FIRMS active fire hotspot client.

Used as a feature: fire activity upwind of a city (esp. during Oct-Nov stubble
burning season) is a known driver of PM2.5 spikes even far from the fire source.

Free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Docs: https://firms.modaps.eosdis.nasa.gov/api/area/

Rate limit: 5000 transactions / 10-minute window (a multi-day request counts
as multiple transactions, so keep DAY_RANGE small for frequent polling).

Set NASA_FIRMS_MAP_KEY as an environment variable before running.
"""

import os
import requests
import pandas as pd
import io
from datetime import datetime, timezone

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Bounding box covering Tamil Nadu + likely upwind fire regions (min_lon,min_lat,max_lon,max_lat)
TN_BBOX = "76.0,8.0,81.5,14.0"

# VIIRS_SNPP_NRT gives finer resolution (375m) than MODIS; good default for this use case
DEFAULT_SENSOR = "VIIRS_SNPP_NRT"


def fetch_fire_hotspots(day_range: int = 1, bbox: str = TN_BBOX,
                         sensor: str = DEFAULT_SENSOR) -> pd.DataFrame:
    """
    Fetch fire hotspot detections within a bounding box for the last `day_range` days.
    Returns lat/lon/brightness/confidence per detection.
    """
    api_key = os.environ.get("NASA_FIRMS_MAP_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NASA_FIRMS_MAP_KEY. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    url = f"{BASE_URL}/{api_key}/{sensor}/{bbox}/{day_range}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def distance_to_nearest_city(df: pd.DataFrame, city_coords: dict) -> pd.DataFrame:
    """
    Rough haversine distance (km) from each fire detection to each tracked city.
    Adds one column per city, e.g. 'dist_to_Chennai_km'. Use the min-distance
    column plus wind direction as features in the model — a fire 200km upwind
    matters more than one 50km downwind.
    """
    import numpy as np

    for city, (lat, lon) in city_coords.items():
        lat1, lon1, lat2, lon2 = map(np.radians, [df["latitude"], df["longitude"], lat, lon])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df[f"dist_to_{city}_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
    return df


def append_to_history(df: pd.DataFrame, path: str = "data/raw/firms_history.csv") -> None:
    if df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    from weather_client import CITY_COORDS
    data = fetch_fire_hotspots(day_range=1)
    print(f"Fetched {len(data)} fire detections in the last 24h")
    if not data.empty:
        data = distance_to_nearest_city(data, CITY_COORDS)
    append_to_history(data)
