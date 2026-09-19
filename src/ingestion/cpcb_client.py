"""
CPCB / data.gov.in air quality client.

Data source: "Real time Air Quality Index from various locations"
Resource ID: 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (verified against data.gov.in listing)
Requires a free API key from https://data.gov.in/user/register (instant approval).

Fields returned per record (confirmed against a live response, not just the
docs — field names in the docs and in the actual JSON have drifted before):
country, state, city, station, last_update, latitude, longitude,
pollutant_id, min_value, max_value, avg_value. One row PER POLLUTANT per
station — you'll need to pivot/aggregate to get a station-level AQI.

Uses urllib (Python stdlib) instead of the `requests` library. On some
Windows machines `requests`/`urllib3` hangs indefinitely against this
specific API — confirmed via diagnostic where PowerShell's
Invoke-WebRequest and Python's urllib both succeeded instantly while
`requests` timed out every time. Likely an antivirus/security-suite HTTPS
inspection layer that doesn't get along with urllib3's TLS handshake.
If you don't hit that issue, requests would work fine too — this isn't
a data.gov.in requirement, just a compatibility workaround.

Set DATA_GOV_IN_API_KEY as an environment variable before running.
"""

import os
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"  # real-time AQI resource
BASE_URL = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"

TN_CITIES = ["Chennai", "Vellore", "Coimbatore", "Madurai", "Tiruchirappalli"]


def _get(url: str, timeout: int) -> dict:
    """A plain urllib GET with a browser-like User-Agent (some APIs/proxies
    reject or stall on requests missing one) and JSON parsing built in."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_current_aqi(state: str | None = None, limit: int = 2000,
                       max_retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch current-hour AQI readings.

    state: server-side filter value is unconfirmed against this dataset (the
    docs' `filters[state]` example did not reliably return Tamil Nadu rows
    in testing) — left as None by default, filtering happens client-side
    in fetch_all_tn_cities() instead, against the always-correct `city`
    column. Pass a state explicitly only if you've confirmed the exact
    string this dataset expects.

    limit: this resource has 3500+ total records across all of India: a
    limit of 100 can silently truncate before reaching Tamil Nadu rows.
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
    }
    if state:
        params["filters[state]"] = state

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = _get(url, timeout=timeout)
            records = payload.get("records", [])
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df["fetched_at"] = datetime.now(timezone.utc).isoformat()
            return df
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            print(f"  attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt  # 5s, then 10s
                print(f"  retrying in {wait}s...")
                time.sleep(wait)

    raise last_error


def fetch_all_tn_cities() -> pd.DataFrame:
    """Fetch nationwide data (no server-side filter — see note above) and
    keep only rows for tracked Tamil Nadu cities."""
    df = fetch_current_aqi(state=None, limit=2000)
    if df.empty:
        return df
    return df[df["city"].isin(TN_CITIES)].reset_index(drop=True)


def append_to_history(df: pd.DataFrame, path: str = "data/raw/cpcb_history.csv") -> None:
    """Append a fetch to the running historical CSV (creates dir + file if absent)."""
    if df.empty:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    data = fetch_all_tn_cities()
    print(f"Fetched {len(data)} rows")
    append_to_history(data)
