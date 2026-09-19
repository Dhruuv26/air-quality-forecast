"""
CPCB / data.gov.in air quality client.

Data source: "Real time Air Quality Index from various locations"
Resource ID: 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (verified against data.gov.in listing)
Requires a free API key from https://data.gov.in/user/register (instant approval).

Fields returned per record: country, state, city, station, pollutant_id,
last_update, pollutant_min, pollutant_max, pollutant_avg. Note this is
one row PER POLLUTANT per station, not a single composite AQI per station -
you'll need to pivot/aggregate to get a station-level AQI.

Set DATA_GOV_IN_API_KEY as an environment variable before running.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone

CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"  # real-time AQI resource
BASE_URL = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"

TN_CITIES = ["Chennai", "Vellore", "Coimbatore", "Madurai", "Tiruchirappalli"]


def fetch_current_aqi(state: str = "Tamil_Nadu", limit: int = 100) -> pd.DataFrame:
    """
    Fetch current-hour AQI readings for all stations in a state.
    Returns a DataFrame with one row per station per pollutant.
    """
    api_key = os.environ.get("DATA_GOV_IN_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set DATA_GOV_IN_API_KEY. Get a free key at https://data.gov.in/user/register"
        )

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "filters[state]": state,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("records", [])

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def fetch_all_tn_cities() -> pd.DataFrame:
    """Fetch and concatenate readings for every tracked Tamil Nadu city."""
    df = fetch_current_aqi(state="Tamil_Nadu")
    if df.empty:
        return df
    return df[df["city"].isin(TN_CITIES)].reset_index(drop=True)


def append_to_history(df: pd.DataFrame, path: str = "data/raw/cpcb_history.csv") -> None:
    """Append a fetch to the running historical CSV (creates it if absent)."""
    if df.empty:
        return
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    data = fetch_all_tn_cities()
    print(f"Fetched {len(data)} rows")
    append_to_history(data)
