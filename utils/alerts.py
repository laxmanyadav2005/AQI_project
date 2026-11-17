"""
utils/alerts.py - Telegram Bot notification engine for AQI health alerts.

Environment variables required (in .env):
  TELEGRAM_BOT_TOKEN — obtained from @BotFather on Telegram
  TELEGRAM_CHAT_ID   — your personal chat ID or a group/channel ID

Usage:
  from utils.alerts import send_aqi_alert
  send_aqi_alert(city='Delhi', aqi=387, category='Very Poor', severity='critical', delta=65)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Emoji map for AQI categories
CATEGORY_EMOJI = {
    "Good":         "🟢",
    "Satisfactory": "🟡",
    "Moderate":     "🟠",
    "Poor":         "🔴",
    "Very Poor":    "🟣",
    "Severe":       "⛔",
}

SEVERITY_EMOJI = {
    "warning":  "⚠️",
    "critical": "🚨",
    "normal":   "ℹ️",
}


def _is_configured() -> bool:
    """Returns True if both Telegram credentials are present."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> bool:
    """
    Sends a plain-text Telegram message using the Bot API.
    Returns True on success, False on failure.
    """
    if not _is_configured():
        print("[ALERTS] Telegram credentials not set — skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[ALERTS] Telegram message sent successfully.")
            return True
        else:
            print(f"[ALERTS] Telegram API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[ALERTS] Failed to send Telegram message: {e}")
        return False


def send_aqi_alert(
    city: str,
    aqi: float,
    category: str,
    severity: str,
    delta: float | None = None,
    z_score: float | None = None,
) -> bool:
    """
    Formats and sends an AQI anomaly health alert via Telegram.

    Args:
        city: City name
        aqi: Current AQI reading
        category: CPCB AQI category label
        severity: 'warning' or 'critical'
        delta: AQI change from previous hour (optional)
        z_score: Statistical z-score (optional)
    """
    cat_emoji = CATEGORY_EMOJI.get(category, "❓")
    sev_emoji = SEVERITY_EMOJI.get(severity, "⚠️")
    title = "CRITICAL AQI Alert" if severity == "critical" else "AQI Health Warning"

    lines = [
        f"{sev_emoji} <b>IND-AQI Platform — {title}</b>",
        f"",
        f"📍 <b>City:</b> {city}",
        f"{cat_emoji} <b>AQI:</b> {aqi:.0f} — <i>{category}</i>",
    ]

    if delta is not None:
        direction = "📈" if delta > 0 else "📉"
        lines.append(f"{direction} <b>Change:</b> {delta:+.0f} points in the last hour")

    if z_score is not None:
        lines.append(f"📊 <b>Z-score:</b> {z_score:.1f}σ from rolling mean")

    health_advice = {
        "Very Poor": "🏠 Stay indoors. Avoid outdoor activity.",
        "Severe":    "🚫 Do NOT go outside. Close all windows. Use air purifiers.",
        "Poor":      "😷 Wear N95 masks outdoors. Limit strenuous activity.",
        "Moderate":  "🚶 Sensitive groups should reduce outdoor exposure.",
    }
    advice = health_advice.get(category, "")
    if advice:
        lines.append(f"")
        lines.append(f"💡 <b>Health Advice:</b> {advice}")

    lines.append(f"")
    lines.append(f"🔗 Monitor live: your-aqi-app.onrender.com")

    message = "\n".join(lines)
    return send_telegram_message(message)


def send_daily_summary(city_summaries: list[dict]) -> bool:
    """
    Sends a daily summary of the top 5 most polluted cities.

    Args:
        city_summaries: List of dicts with keys: city, aqi, category
    """
    if not city_summaries:
        return False

    sorted_cities = sorted(city_summaries, key=lambda x: x['aqi'], reverse=True)[:5]

    lines = [
        "📊 <b>IND-AQI Daily Summary</b>",
        "Top 5 Most Polluted Cities Today:",
        "",
    ]
    for i, c in enumerate(sorted_cities, 1):
        emoji = CATEGORY_EMOJI.get(c['category'], "❓")
        lines.append(f"{i}. {c['city']}: {emoji} <b>{c['aqi']:.0f}</b> ({c['category']})")

    lines.append("")
    lines.append("🔗 Full data: your-aqi-app.onrender.com")

    return send_telegram_message("\n".join(lines))
