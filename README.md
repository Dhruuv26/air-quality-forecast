# Air Quality Forecasting & Early-Warning Dashboard — Tamil Nadu

Forecasts AQI **24 / 48 / 72 hours ahead** for five Tamil Nadu cities (Chennai,
Vellore, Coimbatore, Madurai, Tiruchirappalli) using modelled pollutant data,
weather, and NASA fire hotspots. Each forecast comes with a calibrated
prediction interval and a SHAP explanation of what is driving it, plus a
Streamlit dashboard and a threshold-based email alert system.

**Live dashboard:** https://air-quality-forecast-yhp5cvmevky6h5ahusgcvg.streamlit.app/

> **Research prototype, not health advice.** The training data is *modelled*
> air quality (not station measurements) covering July–October 2026 only. See
> [Limitations](#limitations) before reading anything into the numbers.

## Results

Held-out test period: 7 Oct – 1 Nov 2026 (~3.5 weeks), all five cities pooled.
Baseline is naive persistence (forecast = current AQI).

| Horizon | Persistence MAE | LightGBM MAE | MAE improvement | 80% interval coverage |
|---|---|---|---|---|
| +24h | 20.93 | 21.75 | **-3.9%** (no gain) | 65.8% |
| +48h | 27.50 | 25.33 | 7.9% | 66.3% |
| +72h | 32.51 | 29.11 | 10.4% | 67.8% |

What these say:

- The model adds value at **48–72h**, not at 24h, where persistence is already
  a strong baseline because AQI follows a smooth daily cycle.
- Intervals are calibrated with conformal prediction but still cover ~67%
  instead of 80%, because conditions shifted between the calibration period
  (19 Sep – 4 Oct) and the test period. The dashboard shows the measured
  coverage rather than the nominal figure.
- These are deliberately conservative numbers. An earlier version of the
  evaluation reported gains of 5–13 points more; that was inflated by
  early-stopping on the test set and by calendar features (`month`) that let
  the model memorise the training period. Both were removed.

## How it works

```
OpenWeatherMap air-pollution history ─┐
Open-Meteo weather ───────────────────┼─► build_features.py ─► LightGBM (x3 horizons)
NASA FIRMS fire hotspots ─────────────┘        │                  + quantile models
                                               │                  + conformal calibration
                                               ▼                  + SHAP
                                    data/processed/features.csv ─► Streamlit dashboard
                                                                 └► email alerts
```

- **Target:** an hourly AQI computed with the CPCB breakpoint formula (max
  sub-index over PM2.5, PM10, NO2, O3, SO2, CO). It is an hourly approximation,
  not the official 24h/8h-averaged AQI.
- **Features:** pollutant concentrations, weather, daily fire counts, AQI
  lags (1–48h), rolling mean/std (6–72h), and hour-of-day. Month, weekday and
  weekend flags are excluded on purpose (see Results).
- **Honest time handling:** every city is reindexed to a complete hourly grid
  before lags and targets are computed, so "24h ago" always means 24 clock
  hours. Hours missing at the source (OpenWeatherMap skips some whole days)
  stay empty and are excluded from training.
- **Evaluation:** chronological fit / calibration / test split with a 72h purge
  gap before each boundary, so no training label comes from a later slice.
  Early stopping uses the calibration slice; the test slice is only used for
  the reported metrics.
- **Intervals:** p10/p90 quantile models widened by a conformal offset
  (conformalized quantile regression) learned on the calibration slice.

## Data sources

| Source | Used for | Notes |
|---|---|---|
| [OpenWeatherMap Air Pollution API](https://openweathermap.org/api/air-pollution) | PM2.5, PM10, NO2, O3, SO2, CO | **Modelled** estimates, hourly, history back to 2020. Free key. |
| [Open-Meteo](https://open-meteo.com/) | Temperature, humidity, pressure, wind, cloud | Reanalysis + recent model data. No key. |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) | Fire hotspots | Collected since 19 Sep 2026 only. Free key. |

**Why not CPCB station data?** It was the original plan (`cpcb_client.py` is
still in the repo and works from Indian IPs). But data.gov.in refuses
connections from GitHub's runners, and the CPCB feed contains only Chennai,
Coimbatore, Thoothukudi and Gummidipoondi for Tamil Nadu, so it cannot cover
all five cities or be backfilled. Station data remains the better ground
truth; it would make a good validation set for Chennai.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your keys
```

Keys: `OPENWEATHERMAP_API_KEY` (required) and `NASA_FIRMS_MAP_KEY` (fires).
`DATA_GOV_IN_API_KEY` is only needed for the legacy CPCB client.

## Run the pipeline

Always run from the **project root** (scripts use relative paths).

```bash
python src/ingestion/backfill_real.py        # air + weather history -> data/raw/   (default start 2026-07-01)
python src/features/build_features.py        # -> data/processed/features.csv
python src/models/train_forecast.py          # trains models, prints metrics -> src/models/artifacts/
streamlit run src/dashboard/app.py           # launch the dashboard
```

`backfill_real.py --recent 7` refreshes only the last week; it is idempotent
(duplicates are dropped), so re-running is always safe.

`src/ingestion/generate_synthetic.py` creates fake data matching the API
schemas for offline testing. **It overwrites `data/raw/*.csv`. Never run it in
a checkout that holds real data.**

## Automation (GitHub Actions)

| Workflow | Schedule | What it does |
|---|---|---|
| `collect_data.yml` | every 6h | Re-fetches the last 7 days of air + weather (idempotent), collects FIRMS fires, commits `data/raw/*.csv`. |
| `rebuild_features.yml` | daily, ~07:00 IST | Rebuilds `features.csv`, sanity-checks it (`check_features.py` fails the run if the table is too small, stale or malformed), commits it so the deployed dashboard refreshes. |
| `send_alerts.yml` | every 6h | Runs the alert logic tests, rebuilds features in-run, sends alerts. |

All three share one concurrency group so their commits never collide. Add
these repository secrets under **Settings → Secrets and variables → Actions**:
`OPENWEATHERMAP_API_KEY`, `NASA_FIRMS_MAP_KEY`, and `SMTP_*` for alerts.

**Models are not retrained automatically.** Retrain by hand when enough new
data has accumulated (especially once winter data exists), review the metrics,
then commit `src/models/artifacts/`:

```bash
git pull --rebase
python src/features/build_features.py
python src/models/train_forecast.py
git add src/models/artifacts && git commit -m "Retrain" && git push
```

## Deployment

The dashboard runs on Streamlit Community Cloud from the `main` branch with
main file `src/dashboard/app.py` and Python 3.11. It only reads saved files
(`data/processed/features.csv` and `src/models/artifacts/`) and never trains,
so both must be committed.

## Alert system

Sends an email when a forecast reaches a subscriber's configured AQI band.

```bash
python src/alerts/run_alerts.py           # dry run, prints only (default)
python src/alerts/run_alerts.py --send    # actually sends
python src/alerts/run_alerts.py --force   # ignore cooldown, for testing
python src/alerts/test_alert_engine.py    # run the logic tests
```

Subscribers are configured in `src/alerts/subscribers.json` (cities and the
band at which each wants a warning). Deduplication has three rules:

- **Threshold:** only alert at or above the subscriber's band.
- **Cooldown:** suppress repeats within N hours (default 12).
- **Escalation override:** still send if conditions worsen into a new band.
  A `Poor` alert followed by `Severe` is new information, not a repeat.

`test_alert_engine.py` covers all three, including a simulation asserting that
72 hourly runs over one episode produce exactly 6 alerts. Alert state persists
in `src/alerts/alert_state.json`, committed back by the workflow. A dry run
deliberately does not write state, so your first real send is not suppressed.

Gmail setup: set `SMTP_USER` and `SMTP_PASSWORD` to an **App Password**
(Google Account → Security → 2-Step Verification → App passwords). Optional:
`SMTP_HOST`, `SMTP_PORT`, `SMTP_FROM`.

During July–October fewer than 4% of city-hours reached Moderate or worse, so
expect alerts to be rare until the pollution season.

## Limitations

- **Modelled, not measured.** Pollutant levels are model estimates and can
  differ from what sensors report, especially in polluted urban air.
- **One season.** Training covers July–October 2026 (monsoon and
  post-monsoon, mostly "Good" and "Satisfactory" AQI). Tree models cannot
  predict above the range they saw, so winter spikes are likely to be
  under-predicted. Retrain as winter data arrives.
- **No gain at 24h**, and moderate gains at 48–72h on a ~3.5-week test window,
  so the estimates are noisy. Walk-forward validation would be more robust.
- **Intervals under-cover** (~67% vs 80% nominal) because of the shift between
  calibration and test periods.
- **Hourly AQI approximation**, not the official CPCB 24h/8h-averaged index.
- **Source gaps:** about 6% of hours per city are missing from the air-quality
  source and are excluded from training.
- **Fire features are unvalidated.** FIRMS collection began 19 Sep 2026, so
  earlier days read as zero fires, and the feature is a single Tamil Nadu-wide
  daily count rather than distance/wind-weighted. They do not rank among the
  top SHAP features, but should not be interpreted.
- **Cities are modelled independently** (no spatial smoothing).
- **The deployed model is a fixed snapshot;** only the input data refreshes
  daily.

## Project structure

```
.github/workflows/    # collect_data, rebuild_features, send_alerts
src/
  ingestion/          # backfill_real.py (air + weather), firms_client.py, legacy cpcb/weather clients,
                      #   synthetic generator
  features/           # build_features.py (joins sources, hourly grid, lags), check_features.py
  models/             # train_forecast.py (fit/calib/test, conformal), predict.py, aqi_bands.py
  dashboard/          # Streamlit app
  alerts/             # threshold alerts: engine, notifiers, subscribers, tests
data/
  raw/                # air, weather and fire histories
  processed/          # features.csv
```

## Roadmap

1. **Retrain through the winter season** and report how the model handles real
   pollution episodes.
2. **Walk-forward validation** and a seasonal-naive baseline for a more robust
   estimate than a single split.
3. **FIRMS backfill** and a directional fire feature: is a fire upwind of the
   city, weighted by distance and wind direction.
4. **Validate against CPCB station data** for Chennai, run from an India-based
   host.
5. **Station-based training data** for cities where the CPCB feed allows it.
6. SMS alerts (e.g. Twilio) alongside `EmailNotifier`.
