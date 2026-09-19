"""
Alert decision logic. Deliberately pure — no email sending, no file I/O in the
decision functions — so the rules can be unit tested without a mail server.

The hard part of an alert system is NOT sending mail, it's deciding when NOT
to send it. Three rules handle that:

  1. Threshold   — only alert if a forecast reaches the subscriber's band.
  2. Cooldown    — don't re-alert for an ongoing episode within N hours.
  3. Escalation  — DO override the cooldown if conditions get materially
                   worse (forecast moves into a worse band than what we
                   last alerted about). A "Poor" alert followed by "Severe"
                   six hours later is new information, not a repeat.

Without rule 3, an hourly job silently swallows genuinely worsening
conditions; without rule 2, it sends 72 emails for one three-day episode.
"""

from datetime import datetime, timedelta, timezone
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from aqi_bands import classify, AQI_BANDS  # noqa: E402

BAND_ORDER = {label: i for i, (_, _, label, _, _) in enumerate(AQI_BANDS)}


def band_index(label: str) -> int:
    """Severity rank of a band label. Unknown labels sort as least severe."""
    return BAND_ORDER.get(label, -1)


def worst_forecast(forecasts: list) -> dict:
    """
    The most severe forecast across all horizons — that's what a person
    actually needs warning about. A clean 24h with a Severe 72h still
    warrants an alert.
    """
    return max(forecasts, key=lambda f: f["prediction"])


def should_alert(city: str,
                 forecasts: list,
                 subscriber: dict,
                 state: dict,
                 now: datetime,
                 default_cooldown_h: int = 12) -> tuple:
    """
    Decide whether to alert one subscriber about one city.

    Returns (should_send: bool, reason: str, payload: dict | None).
    `reason` is always populated — including on skips — so the scheduled
    job's logs explain themselves instead of going silent.

    `state` is this subscriber+city's prior alert record:
        {"last_sent_iso": ..., "last_band": ...}
    """
    if city not in subscriber["cities"]:
        return False, "city not subscribed", None

    worst = worst_forecast(forecasts)
    forecast_band = classify(worst["prediction"])["label"]
    threshold_band = subscriber["min_band"]

    if band_index(forecast_band) < band_index(threshold_band):
        return False, (f"forecast {forecast_band} below threshold "
                       f"{threshold_band}"), None

    payload = {
        "city": city,
        "band": forecast_band,
        "horizon_h": worst["horizon_h"],
        "prediction": worst["prediction"],
        "lower": worst["lower"],
        "upper": worst["upper"],
        "drivers": worst.get("drivers", [])[:3],
        "health_note": classify(worst["prediction"])["note"],
    }

    last_sent_iso = state.get("last_sent_iso")
    if not last_sent_iso:
        return True, "first alert for this city", payload

    last_sent = datetime.fromisoformat(last_sent_iso)
    if last_sent.tzinfo is None:
        last_sent = last_sent.replace(tzinfo=timezone.utc)

    cooldown_h = subscriber.get("cooldown_hours", default_cooldown_h)
    cooled_down = now - last_sent >= timedelta(hours=cooldown_h)

    # escalation beats cooldown — worsening conditions are new information
    escalated = band_index(forecast_band) > band_index(state.get("last_band", ""))

    if escalated:
        return True, (f"escalation {state.get('last_band')} -> "
                      f"{forecast_band}"), payload
    if cooled_down:
        return True, f"cooldown of {cooldown_h}h elapsed", payload

    hours_left = cooldown_h - (now - last_sent).total_seconds() / 3600
    return False, (f"within cooldown ({hours_left:.1f}h left), "
                   f"no escalation"), None


def state_key(email: str, city: str) -> str:
    return f"{email}|{city}"


def update_state(state: dict, email: str, city: str,
                 band: str, now: datetime) -> dict:
    """Record that an alert was sent. Returns a new dict, doesn't mutate."""
    new_state = dict(state)
    new_state[state_key(email, city)] = {
        "last_sent_iso": now.isoformat(),
        "last_band": band,
    }
    return new_state


def evaluate_all(all_forecasts: dict,
                 subscribers: list,
                 state: dict,
                 now: datetime,
                 default_cooldown_h: int = 12) -> list:
    """
    Evaluate every subscriber x city pair.
    Returns a list of decision records (both sends and skips, for logging).
    """
    decisions = []
    for sub in subscribers:
        for city, data in all_forecasts.items():
            key = state_key(sub["email"], city)
            send, reason, payload = should_alert(
                city, data["forecasts"], sub, state.get(key, {}),
                now, default_cooldown_h,
            )
            decisions.append({
                "email": sub["email"],
                "name": sub.get("name", ""),
                "city": city,
                "send": send,
                "reason": reason,
                "payload": payload,
                "current_aqi": data.get("current_aqi"),
            })
    return decisions
