"""
Runs all three ingestion clients in one pass, for use in a scheduled job
(GitHub Actions, cron, etc). Each source fails independently — if FIRMS is
down or rate-limited, CPCB and weather data still get collected. This matters
for unattended scheduled runs where you won't be there to retry manually.

Run: python src/ingestion/collect_all.py
Exit code is 0 if at least one source succeeded, 1 if all three failed.
"""

import sys
import traceback

sys.path.insert(0, "src/ingestion")

from cpcb_client import fetch_all_tn_cities, append_to_history as append_cpcb
from weather_client import fetch_all_cities_current, append_to_history as append_weather, CITY_COORDS
from firms_client import fetch_fire_hotspots, distance_to_nearest_city, append_to_history as append_firms


def run_source(name: str, fetch_fn, append_fn, post_process=None) -> bool:
    """Runs one source's fetch+append, catching and logging errors instead of
    letting one bad source kill the whole scheduled run."""
    try:
        print(f"[{name}] fetching...")
        data = fetch_fn()
        if post_process is not None and not data.empty:
            data = post_process(data)
        append_fn(data)
        print(f"[{name}] OK — {len(data)} rows")
        return True
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    results = {
        "cpcb": run_source("cpcb", fetch_all_tn_cities, append_cpcb),
        "weather": run_source("weather", fetch_all_cities_current, append_weather),
        "firms": run_source(
            "firms",
            lambda: fetch_fire_hotspots(day_range=1),
            append_firms,
            post_process=lambda df: distance_to_nearest_city(df, CITY_COORDS),
        ),
    }

    succeeded = sum(results.values())
    print(f"\n{succeeded}/3 sources collected successfully: {results}")

    if succeeded == 0:
        sys.exit(1)  # signal failure to the scheduler so it's visible in Actions logs


if __name__ == "__main__":
    main()
