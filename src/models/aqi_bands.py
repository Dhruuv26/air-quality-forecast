"""
CPCB AQI band classification — shared by the dashboard and (later) the alert system.

Bands follow the official CPCB National Air Quality Index categories.
Note: these bands apply to a proper CPCB AQI value. Right now the pipeline
feeds them `aqi_proxy` (a mean of pollutant averages), which is NOT the same
thing — see README "Known limitations". Band labels will be directionally
right but not officially correct until the sub-index breakpoint formula is
implemented in build_features.py.
"""

# (lower_bound, upper_bound, label, hex_color, short health note)
AQI_BANDS = [
    (0, 50, "Good", "#3BB143", "Minimal impact."),
    (51, 100, "Satisfactory", "#A3C853", "Minor breathing discomfort to sensitive people."),
    (101, 200, "Moderate", "#F4C430", "Breathing discomfort for asthma, lung and heart conditions."),
    (201, 300, "Poor", "#F08040", "Breathing discomfort on prolonged exposure."),
    (301, 400, "Very Poor", "#E03C3C", "Respiratory illness on prolonged exposure."),
    (401, 10_000, "Severe", "#8B2F8B", "Affects healthy people; serious impact on existing conditions."),
]


def classify(aqi: float) -> dict:
    """Return band info for an AQI value: label, color, health note, and band index."""
    if aqi is None:
        return {"label": "Unknown", "color": "#888888", "note": "", "index": -1}
    for i, (lo, hi, label, color, note) in enumerate(AQI_BANDS):
        if lo <= aqi <= hi:
            return {"label": label, "color": color, "note": note, "index": i}
    return {"label": "Unknown", "color": "#888888", "note": "", "index": -1}


def crosses_band(current_aqi: float, forecast_aqi: float) -> bool:
    """True if the forecast moves into a worse band than current — the alert trigger."""
    return classify(forecast_aqi)["index"] > classify(current_aqi)["index"]
