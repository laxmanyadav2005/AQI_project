"""
utils/db_manager.py - Manages live AQI data storage and retrieval.
Supports dual-mode storage: Supabase REST API in production (with auto-fallback)
and local SQLite (aqi_live.db) for local development and testing.
"""

import os
import sqlite3
import requests
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'aqi_live.db')

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Read credentials from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()


def init_local_db():
    """Initializes local SQLite database for fallback or dev use."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hourly_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            datetime TEXT NOT NULL,
            aqi REAL,
            pm25 REAL,
            pm10 REAL,
            no2 REAL,
            so2 REAL,
            co REAL,
            o3 REAL,
            source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, datetime)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            alert_time TEXT NOT NULL,
            aqi REAL,
            severity TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Initialize database table locally on import
init_local_db()


def seed_db_from_cache():
    """Seeds the local SQLite database from the 72h historical cache if it is empty."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hourly_readings")
        count = cursor.fetchone()[0]
        conn.close()

        if count >= 100:
            return  # Already seeded or has user data

        print("[DB] Seeding local database from 72h historical cache...")
        cache_file = os.path.join(BASE_DIR, 'data', 'lstm', 'latest_72h_cache.csv')
        if not os.path.exists(cache_file):
            print(f"[DB WARNING] Cache file {cache_file} not found for database seeding.")
            return

        import pandas as pd
        df = pd.read_csv(cache_file)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        seeded_count = 0
        for _, row in df.iterrows():
            city = row['City']
            dt_str = row['Datetime']
            aqi = row['AQI']
            pm25 = row.get('PM2.5', None)
            pm10 = row.get('PM10', None)
            no2 = row.get('NO2', None)
            so2 = row.get('SO2', None)
            co = row.get('CO', None)
            o3 = row.get('O3', None)

            # Handle NaN values
            aqi = float(aqi) if pd.notna(aqi) else None
            pm25 = float(pm25) if pd.notna(pm25) else None
            pm10 = float(pm10) if pd.notna(pm10) else None
            no2 = float(no2) if pd.notna(no2) else None
            so2 = float(so2) if pd.notna(so2) else None
            co = float(co) if pd.notna(co) else None
            o3 = float(o3) if pd.notna(o3) else None

            cursor.execute("""
                INSERT OR IGNORE INTO hourly_readings 
                (city, datetime, aqi, pm25, pm10, no2, so2, co, o3, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'historical_cache')
            """, (city, dt_str, aqi, pm25, pm10, no2, so2, co, o3))
            if cursor.rowcount > 0:
                seeded_count += 1
                
        conn.commit()
        conn.close()
        print(f"[DB SUCCESS] Seeded {seeded_count} historical readings into local SQLite.")
    except Exception as e:
        print(f"[DB ERROR] Database seeding failed: {e}")


# Run seed script
seed_db_from_cache()


def is_supabase_enabled():
    """Returns True if Supabase credentials are configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


def insert_reading_local(city, dt_str, aqi, pm25, pm10, no2, so2, co, o3, source):
    """Inserts a reading into the local SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO hourly_readings 
            (city, datetime, aqi, pm25, pm10, no2, so2, co, o3, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (city, dt_str, aqi, pm25, pm10, no2, so2, co, o3, source))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save to local SQLite: {e}")
        return False


def insert_reading(city, dt_str, aqi, pm25, pm10, no2, so2, co, o3, source="live"):
    """
    Inserts a single hourly reading.
    Tries Supabase first; falls back to local SQLite on failure or if not configured.
    """
    # Always write to local database first as a cached copy
    insert_reading_local(city, dt_str, aqi, pm25, pm10, no2, so2, co, o3, source)

    if not is_supabase_enabled():
        return True

    # Attempt Supabase write
    url = f"{SUPABASE_URL}/rest/v1/hourly_readings"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"  # UPSERT support on primary key constraint
    }
    payload = {
        "city": city,
        "datetime": dt_str,
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "co": co,
        "o3": o3,
        "source": source
    }

    try:
        # We timeout fast to avoid blocking the caller
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"[WARNING] Supabase write failed (status {response.status_code}): {response.text}")
            return False
    except Exception as e:
        print(f"[WARNING] Supabase connection error: {e}. Fallback to SQLite cached data.")
        return False


def get_latest_readings_local(city, limit=72):
    """Retrieves the latest readings from local SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT city, datetime, aqi, pm25, pm10, no2, so2, co, o3, source
            FROM hourly_readings
            WHERE city = ?
            ORDER BY datetime DESC
            LIMIT ?
        """, (city, limit))
        rows = cursor.fetchall()
        conn.close()

        # Convert to list of dicts, reversing to get chronological order
        result = [dict(row) for row in rows]
        result.reverse()
        return result
    except Exception as e:
        print(f"[ERROR] Failed to query local SQLite: {e}")
        return []


def get_latest_readings(city, limit=72):
    """
    Retrieves the latest hourly readings for a city.
    Tries Supabase first; falls back to local SQLite.
    """
    if not is_supabase_enabled():
        return get_latest_readings_local(city, limit)

    url = f"{SUPABASE_URL}/rest/v1/hourly_readings"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "city": f"eq.{city}",
        "order": "datetime.desc",
        "limit": limit
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # reverse to chronological order
            data.reverse()
            return data
        else:
            print(f"[WARNING] Supabase fetch failed (status {response.status_code}): {response.text}")
            return get_latest_readings_local(city, limit)
    except Exception as e:
        print(f"[WARNING] Supabase fetch connection error: {e}. Fallback to SQLite.")
        return get_latest_readings_local(city, limit)


def insert_alert_log_local(city, alert_time_str, aqi, severity):
    """Inserts an alert log into the local SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alert_logs (city, alert_time, aqi, severity)
            VALUES (?, ?, ?, ?)
        """, (city, alert_time_str, aqi, severity))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save alert to local SQLite: {e}")
        return False


def insert_alert_log(city, alert_time_str, aqi, severity):
    """
    Inserts an alert log into database.
    Tries Supabase first; falls back to local SQLite.
    """
    # Cache locally
    insert_alert_log_local(city, alert_time_str, aqi, severity)

    if not is_supabase_enabled():
        return True

    url = f"{SUPABASE_URL}/rest/v1/alert_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "city": city,
        "alert_time": alert_time_str,
        "aqi": aqi,
        "severity": severity
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"[WARNING] Supabase alert log write error: {e}")
        return False


def get_last_alert_time_local(city):
    """Returns the last alert time string (or None) from local SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT alert_time FROM alert_logs
            WHERE city = ?
            ORDER BY alert_time DESC
            LIMIT 1
        """, (city,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"[ERROR] Failed to read alert time from local SQLite: {e}")
        return None


def get_last_alert_time(city):
    """
    Retrieves the last alert time for a city.
    Tries Supabase first; falls back to local SQLite.
    """
    if not is_supabase_enabled():
        return get_last_alert_time_local(city)

    url = f"{SUPABASE_URL}/rest/v1/alert_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "city": f"eq.{city}",
        "order": "alert_time.desc",
        "limit": 1
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data[0]["alert_time"] if data else None
        else:
            return get_last_alert_time_local(city)
    except Exception as e:
        print(f"[WARNING] Supabase alert log read error: {e}")
        return get_last_alert_time_local(city)


def get_recent_alerts_local(limit=30):
    """Retrieves the latest alert logs from local SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT city, alert_time, aqi, severity
            FROM alert_logs
            ORDER BY alert_time DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[ERROR] Failed to query alert logs from local SQLite: {e}")
        return []


def get_recent_alerts(limit=30):
    """
    Retrieves recent alert logs.
    Tries Supabase first; falls back to local SQLite.
    """
    if not is_supabase_enabled():
        return get_recent_alerts_local(limit)

    url = f"{SUPABASE_URL}/rest/v1/alert_logs"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    params = {
        "order": "alert_time.desc",
        "limit": limit
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return get_recent_alerts_local(limit)
    except Exception as e:
        print(f"[WARNING] Supabase alerts fetch connection error: {e}. Fallback to SQLite.")
        return get_recent_alerts_local(limit)
