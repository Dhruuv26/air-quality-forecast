"""
Real-data backfill + refresh. Replaces the CPCB feed, which GitHub's runners
cannot reach (data.gov.in refuses their IPs).

  air quality : OpenWeatherMap Air Pollution API (history endpoint).
                MODELLED concentrations, not station measurements.
  weather     : Open-Meteo (archive + recent hours). No API key needed.

Writes the two raw files build_features.py already reads:
  data/raw/cpcb_history.csv     city, last_update, pollutant_id, avg_value
  data/raw/weather_history.csv  city, fetched_at, temp_c, humidity_pct,
                                pressure_hpa, wind_speed_ms, wind_deg, clouds_pct

Everything is on ONE clock: IST (Asia/Kolkata) wall time, hourly, written
without a UTC offset. (The old live collectors mixed IST and UTC, so the two
sources never lined up.)

Alongside the six pollutants, an "AQI" row per city-hour holds the CPCB
breakpoint AQI computed from hourly concentrations; build_features.py uses it
as the forecast target. It is hourly, not the official 24h/8h-averaged AQI.

Usage (run from the project root):
  python src/ingestion/backfill_real.py                   # 2026-07-01 -> now
  python src/ingestion/backfill_real.py --start 2026-08-01
  python src/ingestion/backfill_real.py --recent 7        # cheap refresh

Needs OPENWEATHERMAP_API_KEY in .env (or the environment).
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

# keep in sync with weather_client.CITY_COORDS
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Vellore": (12.9165, 79.1325),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
}

CPCB_PATH = "data/raw/cpcb_history.csv"
WEATHER_PATH = "data/raw/weather_history.csv"
CPCB_COLS = ["city", "last_update", "pollutant_id", "avg_value"]
WEATHER_COLS = ["city", "fetched_at", "temp_c", "humidity_pct", "pressure_hpa",
                "wind_speed_ms", "wind_deg", "clouds_pct"]

OWM_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_RECENT = "https://api.open-meteo.com/v1/forecast"
OM_HOURLY = ("temperature_2m,relative_humidity_2m,surface_pressure,"
             "cloud_cover,wind_speed_10m,wind_direction_10m")
OM_RENAME = {
    "temperature_2m": "temp_c", "relative_humidity_2m": "humidity_pct",
    "surface_pressure": "pressure_hpa", "cloud_cover": "clouds_pct",
    "wind_speed_10m": "wind_speed_ms", "wind_direction_10m": "wind_deg",
}
OWM_TO_CPCB = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2",
               "o3": "OZONE", "so2": "SO2", "co": "CO"}


# ---------------------------------------------------------------- HTTP

def _get_json(url: str, timeout: int = 60, retries: int = 3) -> dict:
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(f"HTTP {e.code}: API key rejected or not yet active "
                                   f"(new OpenWeatherMap keys can take ~2h)") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        print(f"  attempt {attempt}/{retries} failed: {last}")
        if attempt < retries:
            time.sleep(5 * attempt)
    raise last


# ---------------------------------------------------------------- CPCB AQI

# (conc_lo, conc_hi, index_lo, index_hi). ug/m3 except CO in mg/m3.
# Piecewise-linear CPCB breakpoints; the open-ended top band is capped at 500.
_BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
              (90, 120, 201, 300), (120, 250, 301, 400), (250, 380, 401, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
             (250, 350, 201, 300), (350, 430, 301, 400), (430, 510, 401, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
            (180, 280, 201, 300), (280, 400, 301, 400), (400, 500, 401, 500)],
    "OZONE": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
              (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
    "CO": [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
           (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
            (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)],
}


def sub_index(pollutant: str, conc: float) -> float:
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return float("nan")
    for lo, hi, ilo, ihi in _BREAKPOINTS[pollutant]:
        if conc <= hi:
            return ilo + (ihi - ilo) / (hi - lo) * (max(conc, lo) - lo)
    return 500.0


def cpcb_aqi(row: dict) -> float:
    """Max sub-index. CPCB rule: >=3 pollutants and PM2.5 or PM10 present.
    `row` holds CPCB-named concentrations; CO must be in mg/m3."""
    subs = {p: sub_index(p, row.get(p)) for p in _BREAKPOINTS}
    subs = {p: v for p, v in subs.items() if not math.isnan(v)}
    if len(subs) < 3 or not ({"PM2.5", "PM10"} & set(subs)):
        return float("nan")
    return round(min(max(subs.values()), 500.0), 1)


# ---------------------------------------------------------------- fetchers

def _windows(start: datetime, end: datetime, days: int = 30):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def fetch_air_city(city: str, lat: float, lon: float, start: datetime,
                   end: datetime, api_key: str) -> pd.DataFrame:
    """Hourly pollutants for one city -> long CPCB-style rows incl. AQI."""
    rows = []
    for w_start, w_end in _windows(start, end):
        q = {"lat": lat, "lon": lon, "start": int(w_start.timestamp()),
             "end": int(w_end.timestamp()), "appid": api_key}
        payload = _get_json(f"{OWM_URL}?{urllib.parse.urlencode(q)}")
        for item in payload.get("list", []):
            comp = item.get("components", {})
            ts = datetime.fromtimestamp(item["dt"], IST)
            rec = {"ts": ts}
            for owm_key, name in OWM_TO_CPCB.items():
                rec[name] = comp.get(owm_key)
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=CPCB_COLS)

    wide = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
    aqi_in = wide.copy()
    aqi_in["CO"] = aqi_in["CO"] / 1000.0  # ug/m3 -> mg/m3 for the AQI formula
    wide["AQI"] = [cpcb_aqi(r) for r in aqi_in.drop(columns="ts").to_dict("records")]

    long = wide.melt(id_vars="ts", var_name="pollutant_id", value_name="avg_value")
    long = long.dropna(subset=["avg_value"])
    long["city"] = city
    long["last_update"] = long["ts"].dt.strftime("%d-%m-%Y %H:%M:%S")
    return long[CPCB_COLS]


def _openmeteo(base: str, lat: float, lon: float, **extra) -> pd.DataFrame:
    q = {"latitude": lat, "longitude": lon, "hourly": OM_HOURLY,
         "wind_speed_unit": "ms", "timezone": "Asia/Kolkata", **extra}
    h = _get_json(f"{base}?{urllib.parse.urlencode(q)}")["hourly"]
    df = pd.DataFrame(h).rename(columns=OM_RENAME)
    df["time"] = pd.to_datetime(df["time"])  # naive IST wall time
    return df


def fetch_weather_city(city: str, lat: float, lon: float, start: datetime,
                       now: datetime) -> pd.DataFrame:
    parts = []
    archive_end = (now - timedelta(days=8)).date()
    if start.date() <= archive_end:
        parts.append(_openmeteo(OPENMETEO_ARCHIVE, lat, lon,
                                start_date=start.date().isoformat(),
                                end_date=archive_end.isoformat()))
    # last ~10 days come from the live model feed; it wins on overlap
    parts.append(_openmeteo(OPENMETEO_RECENT, lat, lon, past_days=10, forecast_days=1))

    df = pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time")
    now_naive = now.replace(tzinfo=None)
    df = df[(df["time"] >= start.replace(tzinfo=None)) & (df["time"] <= now_naive)]
    df = df.dropna(subset=list(OM_RENAME.values()))
    df["city"] = city
    # naive IST wall time, no UTC offset: build_features.py only floors and joins
    # it against CPCB-style timestamps, and offsets break pandas' CSV parsing
    df["fetched_at"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[WEATHER_COLS]


# ---------------------------------------------------------------- output

def merge_write(new: pd.DataFrame, path: str, key_cols: list, expected_cols: list) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        if list(old.columns) == expected_cols:
            if "fetched_at" in old.columns:
                # the old live weather collector wrote UTC timestamps WITH an
                # offset into this same-shaped file; mixing those with naive IST
                # rows breaks date parsing downstream, so drop them
                ok = old["fetched_at"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
                if (~ok).any():
                    print(f"  dropped {(~ok).sum()} old rows with UTC/offset timestamps")
                old = old[ok]
            new = pd.concat([old, new], ignore_index=True)
        else:  # legacy file from the old collectors: different schema + clock
            os.replace(path, path + ".legacy")
            print(f"  moved old-format {path} -> {path}.legacy")
    new = new.drop_duplicates(subset=key_cols, keep="last")
    new.to_csv(path, index=False)
    return len(new)


def _summarise(cpcb: pd.DataFrame) -> None:
    aqi = cpcb[cpcb["pollutant_id"] == "AQI"].copy()
    if aqi.empty:
        return
    edges = [0, 50, 100, 200, 300, 400, 10_000]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    aqi["band"] = pd.cut(aqi["avg_value"], edges, labels=labels, include_lowest=True)
    print("\nAQI by city (min / median / max):")
    print(aqi.groupby("city")["avg_value"].agg(["min", "median", "max"]).round(1))
    print("\nBand share of city-hours:")
    print((aqi["band"].value_counts(normalize=True).reindex(labels) * 100).round(1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default="2026-07-01", help="YYYY-MM-DD (IST)")
    ap.add_argument("--recent", type=int, default=None,
                    help="only the last N days (overrides --start)")
    args = ap.parse_args(argv)

    key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not key:
        print("Set OPENWEATHERMAP_API_KEY in .env first.")
        return 1

    now = datetime.now(IST).replace(minute=0, second=0, microsecond=0)
    start = (now - timedelta(days=args.recent) if args.recent
             else datetime.fromisoformat(args.start).replace(tzinfo=IST))
    print(f"Window: {start:%Y-%m-%d %H:%M} -> {now:%Y-%m-%d %H:%M} IST")

    failed = False

    print("\n[air] OpenWeatherMap history")
    try:
        parts = []
        for city, (lat, lon) in CITY_COORDS.items():
            df = fetch_air_city(city, lat, lon, start, now, key)
            print(f"  {city}: {len(df)} rows")
            parts.append(df)
        cpcb = pd.concat(parts, ignore_index=True)
        total = merge_write(cpcb, CPCB_PATH, ["city", "last_update", "pollutant_id"], CPCB_COLS)
        print(f"  -> {CPCB_PATH} ({total} rows)")
        _summarise(cpcb)
    except Exception as e:  # noqa: BLE001 - keep the other source going
        failed = True
        print(f"[air] FAILED: {e}")

    print("\n[weather] Open-Meteo")
    try:
        parts = []
        for city, (lat, lon) in CITY_COORDS.items():
            df = fetch_weather_city(city, lat, lon, start, now)
            print(f"  {city}: {len(df)} rows")
            parts.append(df)
        weather = pd.concat(parts, ignore_index=True)
        total = merge_write(weather, WEATHER_PATH, ["city", "fetched_at"], WEATHER_COLS)
        print(f"  -> {WEATHER_PATH} ({total} rows)")
    except Exception as e:  # noqa: BLE001
        failed = True
        print(f"[weather] FAILED: {e}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
