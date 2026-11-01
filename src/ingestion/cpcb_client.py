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


def fetch_current_aqi(state: str | None = None, limit: int = 1000,
                      max_retries: int = 3, timeout: int = 30,
                      max_pages: int = 10) -> pd.DataFrame:
    """
    Fetch current-hour AQI readings, paging through the WHOLE resource.

    `limit` is the page size. The resource has 3500+ records nationwide, so a
    single request can silently miss Tamil Nadu rows; we keep requesting
    pages (offset += limit) until a short page or the reported total.
    """
    api_key = os.environ.get("DATA_GOV_IN_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set DATA_GOV_IN_API_KEY. Get a free key at https://data.gov.in/user/register"
        )

    frames = []
    offset = 0
    for _ in range(max_pages):
        params = {"api-key": api_key, "format": "json",
                  "limit": limit, "offset": offset}
        if state:
            params["filters[state]"] = state
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

        payload, last_error = None, None
        for attempt in range(1, max_retries + 1):
            try:
                payload = _get(url, timeout=timeout)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = e
                print(f"  attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
        if payload is None:
            raise last_error

        records = payload.get("records", [])
        if records:
            frames.append(pd.DataFrame(records))
        offset += limit
        total = int(payload.get("total", 0) or 0)
        if len(records) < limit or (total and offset >= total):
            break

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


def fetch_all_tn_cities() -> pd.DataFrame:
    """Fetch nationwide data (no server-side filter — see note above) and
    keep only rows for tracked Tamil Nadu cities."""
    df = fetch_current_aqi(state=None, limit=1000)
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
