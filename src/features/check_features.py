"""
Sanity-check data/processed/features.csv before the daily workflow commits it.
A bad table would be deployed straight to the dashboard, so fail loudly instead.

Run: python src/features/check_features.py
Exit code 1 (and a message) if anything looks wrong.
"""

import sys

import pandas as pd

FEATURES_PATH = "data/processed/features.csv"
MIN_ROWS = 10_000        # ~5 cities x 4+ months of hourly rows
EXPECTED_CITIES = 5
MAX_AGE_HOURS = 72       # newest real reading must be this recent
MAX_NAN_SHARE = 0.20     # share of rows with no AQI (source holes)


def main() -> int:
    f = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    problems = []

    if len(f) < MIN_ROWS:
        problems.append(f"only {len(f)} rows (expected >= {MIN_ROWS})")
    if f["city"].nunique() != EXPECTED_CITIES:
        problems.append(f"{f['city'].nunique()} cities (expected {EXPECTED_CITIES})")

    aqi = f["aqi_proxy"]
    if aqi.isna().mean() > MAX_NAN_SHARE:
        problems.append(f"{aqi.isna().mean():.0%} of rows have no AQI")
    if aqi.dropna().between(0, 500).mean() < 1.0:
        problems.append("AQI values outside 0-500")

    latest = f.loc[aqi.notna(), "timestamp"].max()
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)  # timestamps are IST
    age_h = (now_ist - latest) / pd.Timedelta(hours=1)
    if age_h > MAX_AGE_HOURS:
        problems.append(f"newest reading is {age_h:.0f}h old - data collection looks stuck")

    if problems:
        print("features.csv FAILED checks:\n  - " + "\n  - ".join(problems))
        return 1
    print(f"features.csv OK: {len(f)} rows, {f['city'].nunique()} cities, "
          f"newest reading {latest} ({age_h:.0f}h old)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
