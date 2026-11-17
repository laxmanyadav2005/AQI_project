"""
data_ingestion.py - Ingests live air quality data from OpenWeather Air Pollution API.
Calculates the official CPCB Indian AQI and updates the local SQLite/Supabase database
and the 72-hour cache file used by the LSTM model. Also runs anomaly detection and
fires Telegram alerts when significant AQI spikes are detected.
"""

import os
import datetime
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from utils.aqi_calculator import calculate_aqi, get_aqi_category
from utils.db_manager import insert_reading, get_latest_readings
from utils.anomaly import detect_anomaly, should_alert
from utils.alerts import send_aqi_alert

# Load environment variables
load_dotenv()

# Coordinates for all 26 cities
CITY_COORDINATES = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Hyderabad": [17.3850, 78.4867],
    "Ahmedabad": [23.0225, 72.5714],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462],
    "Patna": [25.5941, 85.1376],
    "Visakhapatnam": [17.6868, 83.2185],
    "Gurugram": [28.4595, 77.0266],
    "Amritsar": [31.6340, 74.8723],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Coimbatore": [11.0168, 76.9558],
    "Guwahati": [26.1158, 91.7278],
    "Bhopal": [23.2599, 77.4126],
    "Chandigarh": [30.7333, 76.7794],
    "Talcher": [20.9525, 85.2285],
    "Jorapokhar": [23.7020, 86.4136],
    "Aizawl": [23.7271, 92.7176],
    "Amaravati": [16.5062, 80.6480],
    "Brajrajnagar": [21.8214, 83.9213],
    "Ernakulam": [9.9816, 76.2999],
    "Kochi": [9.9312, 76.2673],
    "Shillong": [25.5788, 91.8933]
}

# OpenWeather App ID — set OPENWEATHER_API_KEY in your .env file.
# Get a free key at: https://openweathermap.org/appid
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()
if not OPENWEATHER_API_KEY:
    raise EnvironmentError(
        "OPENWEATHER_API_KEY is not set. Add it to your .env file. "
        "Get a free key at https://openweathermap.org/appid"
    )

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lstm", "latest_72h_cache.csv")


def fetch_city_pollution(city, lat, lon):
    """Fetches air pollution data from OpenWeather API for a city."""
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[WARNING] API returned status {response.status_code} for {city}: {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {city}: {e}")
        return None


def process_and_ingest():
    print("=" * 60)
    print(f" Starting Live Ingestion at {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    import time
    new_readings = []

    for city, coords in CITY_COORDINATES.items():
        lat, lon = coords
        print(f"Fetching {city} ({lat}, {lon})...")
        data = fetch_city_pollution(city, lat, lon)
        # Sleep to stay well below the OpenWeather free rate limit (60 requests/min)
        time.sleep(1.2)

        if not data or "list" not in data or len(data["list"]) == 0:
            print(f"  No data returned for {city}.")
            continue

        item = data["list"][0]
        dt = datetime.datetime.utcfromtimestamp(item["dt"])
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        components = item["components"]

        # Parse pollutants
        pm25 = components.get("pm2_5")
        pm10 = components.get("pm10")
        no = components.get("no")
        no2 = components.get("no2")
        so2 = components.get("so2")
        o3 = components.get("o3")
        nh3 = components.get("nh3")

        # OpenWeather CO is in ug/m3. CPCB Caclulator expects mg/m3. We divide by 1000.
        co_raw = components.get("co")
        co = (co_raw / 1000.0) if co_raw is not None else None

        # Build dict for CPCB AQI Calculation
        pollutant_dict = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        }

        # Calculate CPCB AQI
        res = calculate_aqi(pollutant_dict)
        if res["status"] == "success":
            aqi = float(res["aqi"])
            category = res["category"]
            print(f"  Calculated CPCB AQI: {aqi} ({category})")
        else:
            # If calculation fails (e.g. missing PM), we fall back to a simple average sub-index or skip
            print(f"  CPCB AQI Calculation failed: {res['message']}")
            continue

        # Save to database (writes to both local sqlite and production Supabase if set up)
        success = insert_reading(
            city=city,
            dt_str=dt_str,
            aqi=aqi,
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            so2=so2,
            co=co,
            o3=o3,
            source="openweather"
        )

        if success:
            new_readings.append({
                "City": city,
                "Datetime": dt_str,
                "PM2.5": pm25,
                "PM10": pm10,
                "NO": no,
                "NO2": no2,
                "NOx": (no + no2) if (no is not None and no2 is not None) else None,
                "NH3": nh3,
                "CO": co,  # stored in mg/m3 format
                "SO2": so2,
                "O3": o3,
                "Benzene": np.nan,
                "Toluene": np.nan,
                "Xylene": np.nan,
                "AQI": aqi,
                "AQI_Bucket": category
            })

            # --- Anomaly detection ---
            try:
                # Get last alert details to enforce 4-hour cooldown
                from utils.db_manager import get_last_alert_time, insert_alert_log
                last_alert_time_str = get_last_alert_time(city)
                last_alert_hours_ago = None
                if last_alert_time_str:
                    try:
                        # Normalize common datetime formats (e.g. SQLite YYYY-MM-DD HH:MM:SS vs ISO 8601 with T/Z)
                        if 'T' in last_alert_time_str:
                            normalized = last_alert_time_str.split('.')[0].replace('Z', '')
                            last_alert = datetime.datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S")
                        else:
                            last_alert = datetime.datetime.strptime(last_alert_time_str, "%Y-%m-%d %H:%M:%S")
                        
                        elapsed = datetime.datetime.utcnow() - last_alert
                        last_alert_hours_ago = elapsed.total_seconds() / 3600.0
                    except Exception as parse_err:
                        print(f"  [WARNING] Failed to parse last alert time '{last_alert_time_str}' for {city}: {parse_err}")

                recent = get_latest_readings(city, limit=24)
                history = [r["aqi"] for r in recent if r.get("aqi") is not None]
                anomaly = detect_anomaly(city, aqi, history)

                if anomaly["is_anomaly"]:
                    print(f"  [ANOMALY] {city}: severity={anomaly['severity']}, "
                          f"triggers={anomaly['triggers']}")

                # Check if we should send an alert (considers cooldown)
                if should_alert(anomaly, last_alert_hours_ago):
                    delta = anomaly["details"]["rate_of_change"].get("delta")
                    z = anomaly["details"]["zscore"].get("z_score")
                    
                    alert_sent = send_aqi_alert(
                        city=city,
                        aqi=aqi,
                        category=category,
                        severity=anomaly["severity"],
                        delta=delta,
                        z_score=z,
                    )
                    
                    if alert_sent:
                        insert_alert_log(city, dt_str, aqi, anomaly["severity"])
                        print(f"  [ALERT SENT] Alert recorded for {city} at {dt_str}")
            except Exception as ae:
                print(f"  [WARNING] Anomaly detection error for {city}: {ae}")

    if not new_readings:
        print("\n[WARNING] No new readings were successfully processed. Cache will not be updated.")
        return

    # Update latest 72h cache file
    if os.path.exists(CACHE_PATH):
        try:
            cache_df = pd.read_csv(CACHE_PATH)
            new_df = pd.DataFrame(new_readings)

            # Combine and remove duplicates
            combined = pd.concat([cache_df, new_df], ignore_index=True)
            combined["Datetime"] = pd.to_datetime(combined["Datetime"])

            # Sort and remove duplicate city + datetime pairs (keep latest)
            combined = combined.sort_values(["City", "Datetime"])
            combined = combined.drop_duplicates(subset=["City", "Datetime"], keep="last")

            # Retain only the last 72 hours (rows) per city
            updated_cache = combined.groupby("City").tail(72).reset_index(drop=True)

            updated_cache.to_csv(CACHE_PATH, index=False)
            print(f"\n[SUCCESS] Updated cache saved to {CACHE_PATH}")
            print(f"Total rows in cache: {len(updated_cache)} (72 rows per city)")
        except Exception as e:
            print(f"\n[ERROR] Failed to update cache file: {e}")
    else:
        # Create new cache file if not present
        try:
            new_df = pd.DataFrame(new_readings)
            new_df.to_csv(CACHE_PATH, index=False)
            print(f"\n[SUCCESS] Created new cache file at {CACHE_PATH}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save new cache file: {e}")

    print("\nLive Ingestion Cycle Complete.")


if __name__ == "__main__":
    process_and_ingest()
