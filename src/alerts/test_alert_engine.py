"""
Tests for the alert decision logic.

Run: python -m pytest src/alerts/test_alert_engine.py -v
  or: python src/alerts/test_alert_engine.py   (runs without pytest installed)

These cover the failure modes that actually matter in production: alert
spam during a long episode, and silently swallowing a worsening situation.
"""

from datetime import datetime, timedelta, timezone
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from alert_engine import should_alert, evaluate_all, update_state, state_key  # noqa: E402

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)

SUB = {
    "name": "Test User",
    "email": "test@example.com",
    "cities": ["Chennai"],
    "min_band": "Poor",
    "cooldown_hours": 12,
}


def fc(pred, horizon=24):
    """Minimal forecast record."""
    return {"horizon_h": horizon, "prediction": pred,
            "lower": pred - 10, "upper": pred + 10, "drivers": []}


def test_below_threshold_does_not_alert():
    # 120 = Moderate, subscriber wants Poor (201+)
    send, reason, _ = should_alert("Chennai", [fc(120)], SUB, {}, NOW)
    assert not send, reason
    assert "below threshold" in reason


def test_at_threshold_alerts():
    # 250 = Poor
    send, reason, payload = should_alert("Chennai", [fc(250)], SUB, {}, NOW)
    assert send, reason
    assert payload["band"] == "Poor"


def test_unsubscribed_city_ignored():
    send, reason, _ = should_alert("Madurai", [fc(450)], SUB, {}, NOW)
    assert not send
    assert "not subscribed" in reason


def test_worst_horizon_drives_alert():
    """Clean 24h but Severe 72h should still alert, on the 72h value."""
    forecasts = [fc(80, 24), fc(150, 48), fc(420, 72)]
    send, _, payload = should_alert("Chennai", forecasts, SUB, {}, NOW)
    assert send
    assert payload["horizon_h"] == 72
    assert payload["band"] == "Severe"


def test_cooldown_suppresses_repeat():
    """The core anti-spam case: same episode, 3h later, must not re-send."""
    state = {"last_sent_iso": (NOW - timedelta(hours=3)).isoformat(),
             "last_band": "Poor"}
    send, reason, _ = should_alert("Chennai", [fc(250)], SUB, state, NOW)
    assert not send, reason
    assert "within cooldown" in reason


def test_cooldown_expires():
    state = {"last_sent_iso": (NOW - timedelta(hours=13)).isoformat(),
             "last_band": "Poor"}
    send, reason, _ = should_alert("Chennai", [fc(250)], SUB, state, NOW)
    assert send, reason
    assert "cooldown" in reason and "elapsed" in reason


def test_escalation_overrides_cooldown():
    """Poor -> Severe 2h later is new information and must break the cooldown."""
    state = {"last_sent_iso": (NOW - timedelta(hours=2)).isoformat(),
             "last_band": "Poor"}
    send, reason, payload = should_alert("Chennai", [fc(450)], SUB, state, NOW)
    assert send, reason
    assert "escalation" in reason
    assert payload["band"] == "Severe"


def test_de_escalation_does_not_alert():
    """Severe -> Poor within cooldown should stay quiet; it's improving."""
    state = {"last_sent_iso": (NOW - timedelta(hours=2)).isoformat(),
             "last_band": "Severe"}
    send, reason, _ = should_alert("Chennai", [fc(250)], SUB, state, NOW)
    assert not send, reason


def test_hourly_run_over_long_episode_sends_once():
    """
    Simulates the real failure mode: a 3-day Poor episode with an hourly job.
    With a 12h cooldown, 72 runs should produce 6 alerts, not 72.
    """
    state = {}
    sent = 0
    for hour in range(72):
        now = NOW + timedelta(hours=hour)
        key = state_key(SUB["email"], "Chennai")
        send, _, payload = should_alert(
            "Chennai", [fc(250)], SUB, state.get(key, {}), now)
        if send:
            sent += 1
            state = update_state(state, SUB["email"], "Chennai",
                                 payload["band"], now)
    assert sent == 6, f"expected 6 alerts over 72h with 12h cooldown, got {sent}"


def test_evaluate_all_covers_every_pair():
    forecasts = {
        "Chennai": {"current_aqi": 210, "forecasts": [fc(250)]},
        "Madurai": {"current_aqi": 90, "forecasts": [fc(95)]},
    }
    decisions = evaluate_all(forecasts, [SUB], {}, NOW)
    assert len(decisions) == 2
    by_city = {d["city"]: d for d in decisions}
    assert by_city["Chennai"]["send"] is True
    assert by_city["Madurai"]["send"] is False  # not subscribed


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
