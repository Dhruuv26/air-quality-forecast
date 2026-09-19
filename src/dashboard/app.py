"""
Streamlit dashboard for the Tamil Nadu air quality forecast.

Run from the PROJECT ROOT (not from inside src/dashboard):
    streamlit run src/dashboard/app.py

Reads only saved artifacts — never triggers training. If you see a
"no trained models" error, run train_forecast.py first.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from predict import forecast_all_cities, load_results  # noqa: E402
from aqi_bands import classify, AQI_BANDS  # noqa: E402

FEATURES_PATH = "data/processed/features.csv"
HISTORY_HOURS = 168  # one week of context behind the forecast

st.set_page_config(page_title="TN Air Quality Forecast", page_icon="🌫️", layout="wide")


@st.cache_data(ttl=600)
def load_forecasts():
    return forecast_all_cities()


@st.cache_data(ttl=600)
def load_history(city: str) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    df = df[df["city"] == city].sort_values("timestamp")
    return df.tail(HISTORY_HOURS)


@st.cache_data(ttl=600)
def load_metrics():
    return load_results()


def band_chip(aqi: float) -> str:
    b = classify(aqi)
    return (
        f"<span style='background:{b['color']};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{b['label']}</span>"
    )


def forecast_chart(history: pd.DataFrame, forecasts: list, current_aqi: float):
    """History line + forecast points with an 80% prediction interval band."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["timestamp"], y=history["aqi_proxy"],
        mode="lines", name="Observed",
        line=dict(color="#4A90D9", width=2),
    ))

    last_ts = history["timestamp"].iloc[-1]
    fc_times = [last_ts + pd.Timedelta(hours=f["horizon_h"]) for f in forecasts]
    fc_vals = [f["prediction"] for f in forecasts]
    fc_lo = [f["lower"] for f in forecasts]
    fc_hi = [f["upper"] for f in forecasts]

    # anchor the interval band at the last observed point so it doesn't float
    band_x = [last_ts] + fc_times + fc_times[::-1] + [last_ts]
    band_y = [current_aqi] + fc_hi + fc_lo[::-1] + [current_aqi]
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, fill="toself",
        fillcolor="rgba(240,128,64,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="80% interval", hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[last_ts] + fc_times, y=[current_aqi] + fc_vals,
        mode="lines+markers", name="Forecast",
        line=dict(color="#F08040", width=2, dash="dash"),
        marker=dict(size=9),
    ))

    # AQI band reference lines, so the y-axis means something to a non-expert
    for lo, hi, label, color, _ in AQI_BANDS[:4]:
        fig.add_hline(y=hi, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=label, annotation_position="right",
                      annotation_font_size=10, annotation_font_color=color)

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
        yaxis_title="AQI", xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")

try:
    all_forecasts = load_forecasts()
except FileNotFoundError:
    st.error(
        "No trained models or features found. Run this first:\n\n"
        "```\npython src/ingestion/generate_synthetic.py\n"
        "python src/features/build_features.py\n"
        "python src/models/train_forecast.py\n```\n\n"
        "Also make sure you're running streamlit from the project root."
    )
    st.stop()

city = st.sidebar.selectbox("City", sorted(all_forecasts.keys()))
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data source check:** if `data/raw/` was built by "
    "`generate_synthetic.py`, everything below is synthetic and is for "
    "pipeline validation only — not a real air quality forecast."
)

# ── header ─────────────────────────────────────────────────────────────────
data = all_forecasts[city]
current = data["current_aqi"]
band = classify(current)

st.title("Tamil Nadu Air Quality Forecast")
st.caption(f"{city} · latest reading {pd.to_datetime(data['as_of']):%d %b %Y, %H:%M}")

cols = st.columns([1.2, 1, 1, 1])
with cols[0]:
    st.metric("Current AQI", current)
    st.markdown(band_chip(current), unsafe_allow_html=True)
    st.caption(band["note"])

for col, fc in zip(cols[1:], data["forecasts"]):
    with col:
        delta = round(fc["prediction"] - current, 1)
        st.metric(f"+{fc['horizon_h']}h forecast", fc["prediction"], delta=delta,
                  delta_color="inverse")  # rising AQI is bad, so invert the color
        st.markdown(band_chip(fc["prediction"]), unsafe_allow_html=True)
        st.caption(f"range {fc['lower']}–{fc['upper']}")

st.markdown("---")

# ── chart ──────────────────────────────────────────────────────────────────
history = load_history(city)
st.plotly_chart(forecast_chart(history, data["forecasts"], current),
                width='stretch')

# ── explainability ─────────────────────────────────────────────────────────
st.subheader("What's driving the forecast")
st.caption(
    "SHAP values for this specific prediction — how much each input pushed the "
    "forecast up or down, relative to an average day."
)

tabs = st.tabs([f"+{f['horizon_h']}h" for f in data["forecasts"]])
for tab, fc in zip(tabs, data["forecasts"]):
    with tab:
        drivers = pd.DataFrame(fc["drivers"])
        fig = go.Figure(go.Bar(
            x=drivers["shap"], y=drivers["feature"], orientation="h",
            marker_color=["#E03C3C" if v > 0 else "#3BB143" for v in drivers["shap"]],
            text=[f"{v:+.1f}" for v in drivers["shap"]], textposition="outside",
        ))
        fig.update_layout(
            height=260, margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Effect on forecast (AQI points)", yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width='stretch')

        top = fc["drivers"][0]
        st.info(
            f"**Main factor:** {top['feature']} "
            f"(currently {top['value']}) {top['direction']} the forecast by "
            f"{abs(top['shap'])} AQI points."
        )

# ── model performance ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Model performance"):
    metrics = load_metrics()
    rows = []
    for h in [24, 48, 72]:
        m = metrics[f"{h}h"]
        rows.append({
            "Horizon": f"+{h}h",
            "Baseline MAE": round(m["baseline"]["mae"], 2),
            "Model MAE": round(m["model"]["mae"], 2),
            "Improvement": f"{m['improvement_pct']:.1f}%",
            "Interval coverage": f"{m.get('interval_coverage', 0):.1%}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(
        "Baseline is naive persistence (assume AQI stays where it is). "
        "Interval coverage is the share of actuals that fell inside the 80% "
        "band on the test set — currently running a few points under nominal, "
        "so the bands are slightly narrower than advertised."
    )
