"""
Notification delivery. Two backends:

  ConsoleNotifier — prints what WOULD be sent. Default, and what you should
                    use while testing. No credentials, no risk of emailing
                    real people during development.
  EmailNotifier   — real SMTP delivery.

Gmail note: a normal account password will NOT work. You need an App Password
(Google Account -> Security -> 2-Step Verification -> App passwords), and 2FA
must be on. Set SMTP_USER / SMTP_PASSWORD to that.
"""

import os
import smtplib
from email.message import EmailMessage


def format_alert(payload: dict, name: str = "") -> tuple:
    """Build (subject, plain_text_body) for one alert. Kept separate from
    sending so it can be tested and reused by other backends (SMS, push)."""
    city = payload["city"]
    band = payload["band"]
    horizon = payload["horizon_h"]

    subject = f"Air quality alert: {band} conditions forecast for {city}"

    greeting = f"Hi {name}," if name else "Hi,"
    lines = [
        greeting,
        "",
        f"Air quality in {city} is forecast to reach {band} conditions "
        f"in about {horizon} hours.",
        "",
        f"  Forecast AQI: {payload['prediction']} "
        f"(likely range {payload['lower']}-{payload['upper']})",
        f"  Category:     {band}",
        f"  Health note:  {payload['health_note']}",
    ]

    if payload.get("drivers"):
        lines += ["", "Main factors behind this forecast:"]
        for d in payload["drivers"]:
            lines.append(f"  - {d['feature']} ({d['direction']} the forecast)")

    lines += [
        "",
        "This is an automated forecast from a student-built model, not an "
        "official advisory. For official air quality information, see CPCB "
        "at https://cpcb.nic.in or the Sameer app.",
    ]
    return subject, "\n".join(lines)


class ConsoleNotifier:
    """Dry-run backend — prints instead of sending. Safe default."""

    def __init__(self):
        self.sent = []

    def send(self, to_email: str, subject: str, body: str) -> bool:
        print(f"\n{'=' * 60}")
        print(f"[DRY RUN] would send to: {to_email}")
        print(f"Subject: {subject}")
        print(f"{'-' * 60}")
        print(body)
        print(f"{'=' * 60}\n")
        self.sent.append((to_email, subject))
        return True


class EmailNotifier:
    """Real SMTP delivery. Requires SMTP_* environment variables."""

    def __init__(self):
        self.host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER")
        self.password = os.environ.get("SMTP_PASSWORD")
        self.from_addr = os.environ.get("SMTP_FROM", self.user)

        if not self.user or not self.password:
            raise EnvironmentError(
                "EmailNotifier needs SMTP_USER and SMTP_PASSWORD. "
                "For Gmail, use an App Password, not your account password."
            )
        self.sent = []

    def send(self, to_email: str, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = to_email
        msg.set_content(body)

        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            self.sent.append((to_email, subject))
            return True
        except Exception as e:
            # one bad recipient must not kill the whole scheduled run
            print(f"  [email] FAILED to {to_email}: {e}")
            return False


def get_notifier(dry_run: bool = True):
    return ConsoleNotifier() if dry_run else EmailNotifier()
