"""
Alert runner — the entry point a scheduled job calls.

Wires: forecasts -> decision engine -> notifier, then persists alert state
so the next run knows what was already sent.

Usage:
    python src/alerts/run_alerts.py              # dry run, prints only (default)
    python src/alerts/run_alerts.py --send       # actually sends email
    python src/alerts/run_alerts.py --force      # ignore cooldown (testing)

Defaults to DRY RUN on purpose — sending is opt-in, so a misconfigured
schedule can't silently email people during development.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from alert_engine import evaluate_all, update_state, state_key  # noqa: E402
from notifiers import get_notifier, format_alert  # noqa: E402
from predict import forecast_all_cities  # noqa: E402

ALERTS_DIR = os.path.dirname(__file__)
SUBSCRIBERS_PATH = os.path.join(ALERTS_DIR, "subscribers.json")
STATE_PATH = os.path.join(ALERTS_DIR, "alert_state.json")


def load_subscribers() -> tuple:
    with open(SUBSCRIBERS_PATH) as f:
        cfg = json.load(f)
    return cfg["subscribers"], cfg.get("defaults", {}).get("cooldown_hours", 12)


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true",
                        help="actually send email (default is dry run)")
    parser.add_argument("--force", action="store_true",
                        help="ignore cooldown/state, for testing")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    subscribers, default_cooldown = load_subscribers()
    state = {} if args.force else load_state()

    print(f"Alert run at {now.isoformat()} "
          f"({'LIVE SEND' if args.send else 'DRY RUN'})")

    try:
        all_forecasts = forecast_all_cities()
    except FileNotFoundError as e:
        print(f"Cannot run alerts — missing model or feature files: {e}")
        print("Run build_features.py and train_forecast.py first.")
        sys.exit(1)

    decisions = evaluate_all(all_forecasts, subscribers, state, now,
                             default_cooldown)

    notifier = get_notifier(dry_run=not args.send)
    sent_count = 0

    for d in decisions:
        if not d["send"]:
            print(f"  skip  {d['email']} / {d['city']}: {d['reason']}")
            continue

        subject, body = format_alert(d["payload"], d["name"])
        ok = notifier.send(d["email"], subject, body)
        if ok:
            sent_count += 1
            print(f"  SENT  {d['email']} / {d['city']}: {d['reason']}")
            state = update_state(state, d["email"], d["city"],
                                 d["payload"]["band"], now)
        else:
            print(f"  ERROR {d['email']} / {d['city']}: delivery failed")

    # only persist state on a real send — a dry run must not mark alerts as
    # delivered, or the first live run would be silently suppressed
    if args.send and not args.force:
        save_state(state)
        print(f"\nState saved to {STATE_PATH}")

    print(f"\n{sent_count} alert(s) "
          f"{'sent' if args.send else 'would be sent'}, "
          f"{len(decisions) - sent_count} skipped")


if __name__ == "__main__":
    main()
