"""
utils/anomaly.py - Hybrid anomaly detection for live AQI readings.
Uses two complementary algorithms:
  1. Z-score: flags readings that deviate >3σ from the rolling mean (seasonal baseline)
  2. Rate-of-change: flags a spike if AQI jumps >50 points within 2 consecutive hours

Cold-start: if fewer than 24 readings are available for a city, falls back to
absolute thresholds (AQI > 300 = anomaly, AQI jump > 80 = rate-of-change anomaly).
"""

import math
import statistics


def _zscore_anomaly(current_aqi: float, history: list[float], threshold: float = 3.0) -> dict:
    """
    Returns anomaly info if current_aqi is >threshold standard deviations
    from the historical rolling mean.
    """
    if len(history) < 3:
        # Cold start: use absolute threshold
        is_anomaly = current_aqi > 300
        return {
            "method": "absolute_threshold",
            "is_anomaly": is_anomaly,
            "z_score": None,
            "mean": None,
            "std": None,
        }

    mean = statistics.mean(history)
    std = statistics.stdev(history)

    if std < 1e-6:  # avoid division by zero on perfectly flat signal
        z = 0.0
    else:
        z = (current_aqi - mean) / std

    return {
        "method": "zscore",
        "is_anomaly": abs(z) > threshold,
        "z_score": round(z, 2),
        "mean": round(mean, 1),
        "std": round(std, 1),
    }


def _rate_of_change_anomaly(current_aqi: float, prev_aqi: float, threshold: float = 50.0) -> dict:
    """
    Returns anomaly info if the AQI jumped or crashed by more than `threshold`
    in a single hourly step.
    """
    delta = current_aqi - prev_aqi
    is_anomaly = abs(delta) > threshold
    return {
        "method": "rate_of_change",
        "is_anomaly": is_anomaly,
        "delta": round(delta, 1),
        "prev_aqi": round(prev_aqi, 1),
    }


def detect_anomaly(city: str, current_aqi: float, history: list[float]) -> dict:
    """
    Main anomaly detection entry point. Combines Z-score and rate-of-change signals.

    Args:
        city: City name (for logging/response)
        current_aqi: The latest AQI reading to evaluate
        history: List of recent AQI readings (ordered oldest → newest, NOT including current)

    Returns:
        Dict with:
          - is_anomaly (bool): True if either method triggers
          - severity (str): 'normal' | 'warning' | 'critical'
          - triggers (list): which methods triggered
          - details (dict): per-method breakdown
    """
    zscore_result = _zscore_anomaly(current_aqi, history)

    roc_result = {"method": "rate_of_change", "is_anomaly": False, "delta": 0.0, "prev_aqi": 0.0}
    if history:
        prev_aqi = history[-1]
        roc_result = _rate_of_change_anomaly(current_aqi, prev_aqi)

    triggers = []
    if zscore_result["is_anomaly"]:
        triggers.append("zscore")
    if roc_result["is_anomaly"]:
        triggers.append("rate_of_change")

    is_anomaly = len(triggers) > 0

    # Severity: critical if both methods agree or AQI > 400; warning if one method
    if is_anomaly and (len(triggers) == 2 or current_aqi > 400):
        severity = "critical"
    elif is_anomaly:
        severity = "warning"
    else:
        severity = "normal"

    return {
        "city": city,
        "current_aqi": current_aqi,
        "is_anomaly": is_anomaly,
        "severity": severity,
        "triggers": triggers,
        "details": {
            "zscore": zscore_result,
            "rate_of_change": roc_result,
        }
    }


def should_alert(anomaly_result: dict, last_alert_hours_ago: float | None) -> bool:
    """
    Determines whether a Telegram alert should be sent.
    Rules:
      - Only alert on 'warning' or 'critical' severity
      - Enforce a 4-hour cooldown per city to avoid spam
    """
    if not anomaly_result["is_anomaly"]:
        return False
    if anomaly_result["severity"] == "normal":
        return False
    if last_alert_hours_ago is not None and last_alert_hours_ago < 4.0:
        return False
    return True
