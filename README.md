# Air Quality Forecasting & Early-Warning Dashboard — Tamil Nadu

Forecasts AQI 24/48/72 hours ahead for Tamil Nadu cities using CPCB pollutant
data, weather data, and NASA fire-hotspot data, with SHAP-based explainability
("AQI is expected to spike Thursday because wind speed drops and fire
activity is up") and a dashboard + threshold alert system.

## Status

The core pipeline (ingestion → feature engineering → LightGBM forecast model →
SHAP explainability) is built and validated end-to-end on synthetic data.
LightGBM currently beats a naive persistence baseline by ~27-29% MAE across
all three horizons on synthetic data — a real evaluation on real data is the
next step once your API keys are approved and enough history has accumulated.

A GitHub Actions workflow (`.github/workflows/collect_data.yml`) is set up to
run all three ingestion scripts hourly and commit new data back to the repo —
see **Setting up scheduled collection** below to turn it on.

The Streamlit dashboard is built and working (`src/dashboard/app.py`) —
current AQI with CPCB band, 24/48/72h forecasts with 80% prediction
intervals, a per-prediction SHAP driver panel, and a model performance table.

The alert system is built and tested (`src/alerts/`) — threshold-based email
alerts with cooldown deduplication and escalation override, 10 unit tests
covering the dedup logic, and a scheduled workflow. Defaults to dry run.

**Not yet built:** deployment.

## Why synthetic data first

This pipeline was built in an environment without access to CPCB, IMD/OWM, or
NASA FIRMS endpoints, so `src/ingestion/generate_synthetic.py` produces data
matching the *exact schema* each real API returns (verified against current
API docs as of Aug 2026). This means:

- Every downstream script (`build_features.py`, `train_forecast.py`) already
  works and is tested — swapping in real data means editing zero feature/model
  code, just re-running ingestion.
- You can develop and debug the whole pipeline today, in parallel with
  waiting for API key approvals.

**Important caveat on real data:** the CPCB and OpenWeatherMap APIs return
*current/recent* readings, not deep history. To get 90 days of real training
data, the scheduled job needs to run for 90 days — you cannot backfill it on
day one. Start the scheduled collection now, since this is your actual
bottleneck.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # then fill in your API keys
```

Get your free API keys:
- **CPCB / data.gov.in**: https://data.gov.in/user/register (instant approval)
- **OpenWeatherMap**: https://openweathermap.org/api (instant, free tier)
- **NASA FIRMS**: https://firms.modaps.eosdis.nasa.gov/api/map_key/ (instant)

## Quickstart (synthetic data, works right now)

```bash
python src/ingestion/generate_synthetic.py   # generates 90 days of synthetic data
python src/features/build_features.py        # joins sources, builds lag/rolling features
python src/models/train_forecast.py          # trains LightGBM per horizon, prints metrics
streamlit run src/dashboard/app.py           # launches the dashboard
```

Run all four from the **project root** — the scripts use relative paths, so
running `streamlit run app.py` from inside `src/dashboard/` will fail to find
the data and model files.

## Switching to real data (manual, one-off)

Once your API keys are in `.env`, you can run the real ingestion scripts
individually, or all at once with the combined script:

```bash
python src/ingestion/collect_all.py   # runs CPCB + weather + FIRMS in one pass
```

Each source fails independently — if one API is down or rate-limited, the
other two still get collected. This matters for unattended runs where you
won't be there to retry manually.

## Setting up scheduled collection (GitHub Actions)

This is how you actually accumulate 90 days of real history without manually
running the script every hour yourself.

1. Push this repo to GitHub (if you haven't already).
2. In your repo, go to **Settings → Secrets and variables → Actions** and add
   three repository secrets: `DATA_GOV_IN_API_KEY`, `OPENWEATHERMAP_API_KEY`,
   `NASA_FIRMS_MAP_KEY` — same values as your local `.env`.
3. The workflow at `.github/workflows/collect_data.yml` is already set to run
   hourly (`cron: "0 * * * *"`) and commit new rows to `data/raw/*.csv`
   automatically. It also supports a manual trigger — go to the **Actions**
   tab, select "Collect air quality data", and click **Run workflow** to test
   it immediately rather than waiting for the next hour.
4. Check the Actions tab after the first run — if a source fails (e.g. bad
   key, rate limit), the logs will show exactly which one and why, since
   `collect_all.py` reports each source's status independently.
5. Let it run for at least 2-3 weeks before retraining on real data — a
   handful of hours of history won't give the lag/rolling features (which
   look back up to 72h) enough to work with.

**Note on GitHub Actions free tier:** public repos get unlimited Actions
minutes; private repos get a monthly quota (2,000 min/month on the free
plan). An hourly job doing a quick API pull + commit is well within that even
on a private repo, but worth knowing if you add heavier steps later.

## Alert system

Sends an email when a forecast reaches a subscriber's configured AQI band.

```bash
python src/alerts/run_alerts.py           # dry run — prints, sends nothing (default)
python src/alerts/run_alerts.py --send    # actually sends
python src/alerts/run_alerts.py --force   # ignore cooldown, for testing
python src/alerts/test_alert_engine.py    # run the logic tests
```

**Configure subscribers** in `src/alerts/subscribers.json` — each entry sets
which cities they care about and the band at which they want warning
(someone with asthma might set `Moderate`; a general user `Poor`).

**The interesting part is the deduplication.** An hourly job over a 3-day
"Poor" episode would naively send 72 emails. Three rules prevent that:

- **Threshold** — only alert at or above the subscriber's band.
- **Cooldown** — suppress repeats within N hours (default 12).
- **Escalation override** — but DO send if conditions worsen into a new band.
  A `Poor` alert followed by `Severe` two hours later is new information, not
  a repeat. Without this rule the cooldown silently swallows a worsening
  situation, which is the failure mode that actually hurts someone.

`test_alert_engine.py` covers all three, including a simulation asserting
that 72 hourly runs over one episode produce exactly 6 alerts.

**Email setup (Gmail):** set `SMTP_USER` and `SMTP_PASSWORD` — the password
must be an **App Password** (Google Account → Security → 2-Step Verification
→ App passwords), not your normal account password. Optional: `SMTP_HOST`,
`SMTP_PORT`, `SMTP_FROM`.

**Scheduled alerts:** `.github/workflows/send_alerts.yml` runs every 6 hours.
Add the `SMTP_*` values as repo secrets. The workflow runs the logic tests
first and won't send anything if they fail. Trigger it manually from the
Actions tab (dry-run defaults to on) before letting it run live.

Alert state persists in `src/alerts/alert_state.json`, committed back by the
workflow so cooldowns survive across runs. A dry run deliberately does *not*
write state — otherwise your first real send would be suppressed.

## Project structure

```
.github/workflows/  # scheduled data collection (GitHub Actions)
src/
  ingestion/          # CPCB, weather, FIRMS API clients + synthetic generator + collect_all.py
  features/           # joins sources, builds lag/rolling/calendar features
  models/             # LightGBM training, evaluation, SHAP explainability
  dashboard/          # Streamlit dashboard (app.py)
  alerts/             # threshold alerts: engine, notifiers, subscribers, tests
data/
  raw/                # raw pulls from each source, append-only history
  processed/          # features.csv — model-ready table
```

## Roadmap

1. **Let scheduled collection run** (in progress) — 2-3 weeks minimum before
   retraining on real data.
2. **Re-validate model on real data** — the synthetic-data results above are
   a pipeline sanity check, not a real result.
3. **Spatial features** — use `firms_client.distance_to_nearest_city` plus
   wind direction to build a directional "is this fire actually upwind of
   this city" feature, rather than a flat TN-wide fire count. Single
   highest-value technical improvement available, worth highlighting in any
   writeup.
4. ~~**Dashboard**~~ — done (`src/dashboard/app.py`).
5. ~~**Alert system**~~ — done (`src/alerts/`). Possible extension: SMS via
   Twilio, by adding a notifier class alongside `EmailNotifier`.
6. **Deploy** — Streamlit Community Cloud (free) is the fastest path to a
   live, shareable link for your resume. Note it needs `data/processed/` and
   `src/models/artifacts/` committed to the repo, since the deployed app
   reads saved artifacts and never trains.

## Known limitations / honest caveats

- `aqi_proxy` in `build_features.py` is a simple mean of pollutant averages,
  not the official CPCB sub-index breakpoint formula. For a submission-quality
  result, implement the piecewise-linear breakpoint calculation (see the CPCB
  AQI calculator spec) — more accurate and more defensible if presenting this
  work.
- FIRMS integration currently uses a flat daily fire count across all of
  Tamil Nadu, not distance/wind-weighted per city (see Roadmap #3).
- No spatial smoothing between neighboring CPCB stations yet — each city is
  modeled independently.
- **Prediction intervals are slightly under-covered.** The p10/p90 quantile
  models give ~76% empirical coverage against a nominal 80% on the test set,
  so the bands are a bit narrower than they claim. The dashboard shows the
  real coverage number rather than hiding this. Worth recalibrating (e.g.
  conformal prediction) before presenting the intervals as trustworthy.
- **SHAP drivers on synthetic data are dominated by time-of-day**, because
  the synthetic generator bakes in a strong diurnal traffic pattern. On real
  data expect weather (especially wind speed) to matter far more. Don't read
  anything into the current driver rankings.
