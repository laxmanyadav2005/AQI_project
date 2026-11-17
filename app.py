"""
app.py - Flask backend for the Indian AQI ML Project.
Loads the trained ML models and provides an interactive web dashboard
for AQI calculation, next-day forecasting, and city comparisons.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from utils.aqi_calculator import calculate_aqi, get_aqi_category

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Predefined coordinates for Indian cities to render on the dashboard map
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

# Load models and preprocessing assets
try:
    model_pred = joblib.load(os.path.join(MODELS_DIR, 'xgboost_aqi_predictor.pkl'))
    model_cls = joblib.load(os.path.join(MODELS_DIR, 'xgboost_aqi_classifier.pkl'))
    le_aqi = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    le_city = joblib.load(os.path.join(MODELS_DIR, 'city_encoder.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("[SUCCESS] All ML models loaded successfully.")
except Exception as e:
    print(f"[WARNING] ML Models could not be loaded: {e}")
    model_pred = None
    model_cls = None

# Load Keras LSTM model and metadata for hourly forecasts
# Note: We use direct Keras inference instead of TFLite because TF's LSTM
# layer generates dynamic TensorList ops that cannot be converted to pure
# TFLite builtins. The Keras model is only ~440 KB and loads instantly.
import json

lstm_model = None
lstm_scalers = None
lstm_city_map = None
lstm_residual_stats = None

try:
    import tensorflow as tf
    keras_path = os.path.join(MODELS_DIR, 'lstm_best.keras')
    if os.path.exists(keras_path):
        lstm_model = tf.keras.models.load_model(keras_path)
        print(f"[SUCCESS] LSTM Keras model loaded ({os.path.getsize(keras_path) / 1024:.0f} KB).")
    else:
        print("[WARNING] lstm_best.keras not found in models/.")
except Exception as e:
    print(f"[WARNING] LSTM Keras model could not be loaded: {e}")

try:
    scalers_path = os.path.join(BASE_DIR, 'data', 'lstm', 'scalers.pkl')
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            lstm_scalers = joblib.load(f)

    city_map_path = os.path.join(BASE_DIR, 'data', 'lstm', 'city_map.json')
    if os.path.exists(city_map_path):
        with open(city_map_path, 'r') as f:
            lstm_city_map = json.load(f)

    residual_stats_path = os.path.join(MODELS_DIR, 'lstm_residual_stats.json')
    if os.path.exists(residual_stats_path):
        with open(residual_stats_path, 'r') as f:
            lstm_residual_stats = json.load(f)
    print("[SUCCESS] LSTM assets loaded successfully.")
except Exception as e:
    print(f"[WARNING] LSTM metadata assets could not be loaded: {e}")

# Load dataset for dashboard historical views
try:
    dataset_path = os.path.join(DATA_DIR, 'cleaned_aqi_data.csv')
    if os.path.exists(dataset_path):
        df_clean = pd.read_csv(dataset_path)
        print("[SUCCESS] Cleaned dataset loaded for dashboard usage.")
    else:
        df_clean = None
except Exception as e:
    print(f"[WARNING] Dataset could not be loaded: {e}")
    df_clean = None

# Load 72h cache for hourly forecasts
try:
    cache_path = os.path.join(BASE_DIR, 'data', 'lstm', 'latest_72h_cache.csv')
    if os.path.exists(cache_path):
        df_cache = pd.read_csv(cache_path)
        df_cache['Datetime'] = pd.to_datetime(df_cache['Datetime'])
        print("[SUCCESS] 72-hour hourly cache loaded.")
    else:
        df_cache = None
except Exception as e:
    print(f"[WARNING] 72-hour hourly cache could not be loaded: {e}")
    df_cache = None

@app.route('/')
def home():
    # Gather statistics for dashboard cards and map markers
    dashboard_data = []
    if df_clean is not None:
        recent_data = df_clean.sort_values(by='Date').groupby('City').last().reset_index()
        for _, row in recent_data.iterrows():
            city_name = row['City']
            aqi_val = int(row['AQI'])
            cat_info = get_aqi_category(aqi_val)
            color_hex = cat_info['color'].lstrip('#')
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            bg_color = f"rgba({r}, {g}, {b}, 0.15)"
            
            coords = CITY_COORDINATES.get(city_name, [20.5937, 78.9629])
            
            dashboard_data.append({
                'city': city_name,
                'aqi': aqi_val,
                'category': cat_info['label'],
                'color': cat_info['color'],
                'bg_color': bg_color,
                'pm25': round(row['PM2.5'], 1),
                'pm10': round(row['PM10'], 1),
                'date': row['Date'],
                'lat': coords[0],
                'lng': coords[1]
            })
        dashboard_data = sorted(dashboard_data, key=lambda x: x['aqi'], reverse=True)
    return render_template('index.html', cities=dashboard_data)

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    if request.method == 'POST':
        try:
            pollutants = {}
            for param in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
                val_str = request.form.get(param, '').strip()
                if val_str:
                    pollutants[param] = float(val_str)
                    
            if not pollutants:
                return jsonify({'status': 'error', 'message': 'Please enter concentration values for at least 3 pollutants.'})
                
            res = calculate_aqi(pollutants)
            return jsonify(res)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Calculation error: {str(e)}'})
            
    return render_template('calculator.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    cities_list = list(le_city.classes_) if le_city else []
    
    if request.method == 'POST':
        if model_pred is None:
            return jsonify({'status': 'error', 'message': 'ML Models are not initialized.'})
            
        try:
            city_name = request.form.get('city')
            
            # Helper function for safe float parsing of optional inputs
            def get_float(name, default=0.0):
                val_str = request.form.get(name, '').strip()
                return float(val_str) if val_str else default
            
            today_aqi = get_float('today_aqi', 100.0)
            today_pm25 = get_float('today_pm25', 35.0)
            temp = get_float('temp', 25.0)
            humidity = get_float('humidity', 60.0)
            wind_speed = get_float('wind_speed', 10.0)
            pressure = get_float('pressure', 1010.0)
            
            from datetime import datetime
            now = datetime.now()
            month = now.month
            day = now.day
            dayofweek = now.weekday()
            is_weekend = 1 if dayofweek in [5, 6] else 0
            
            input_dict = {c: 0.0 for c in feature_cols}
            
            input_dict['PM2.5'] = today_pm25
            input_dict['Month'] = month
            input_dict['Day'] = day
            input_dict['DayOfWeek'] = dayofweek
            input_dict['IsWeekend'] = is_weekend
            
            input_dict['AQI_lag_1'] = today_aqi
            input_dict['PM25_lag_1'] = today_pm25
            input_dict['AQI_lag_2'] = today_aqi
            input_dict['AQI_lag_3'] = today_aqi
            input_dict['AQI_lag_7'] = today_aqi
            input_dict['PM25_lag_3'] = today_pm25
            
            input_dict['AQI_rolling_3d'] = today_aqi
            input_dict['AQI_rolling_7d'] = today_aqi
            input_dict['AQI_rolling_14d'] = today_aqi
            
            if le_city:
                try:
                    input_dict['City_encoded'] = le_city.transform([city_name])[0]
                except ValueError:
                    input_dict['City_encoded'] = 0
            
            input_dict['PM_ratio'] = 0.55
            input_dict['NOx_ratio'] = 0.8
            
            if month in [12, 1, 2]:
                if 'Season_Winter' in input_dict:
                    input_dict['Season_Winter'] = 1.0
            elif month in [6, 7, 8, 9]:
                if 'Season_Monsoon' in input_dict:
                    input_dict['Season_Monsoon'] = 1.0
            elif month in [3, 4, 5]:
                if 'Season_Spring' in input_dict:
                    input_dict['Season_Spring'] = 1.0
                    
            input_features = [input_dict[col] for col in feature_cols]
            input_features = np.array(input_features).reshape(1, -1)
            
            predicted_val = float(model_pred.predict(input_features)[0])
            predicted_val = max(0.0, min(predicted_val, 500.0))
            
            predicted_class_encoded = model_cls.predict(input_features)[0]
            predicted_category = le_aqi.classes_[predicted_class_encoded]
            
            cat_info = get_aqi_category(predicted_val)
            
            return jsonify({
                'status': 'success',
                'predicted_aqi': round(predicted_val, 1),
                'predicted_category': predicted_category,
                'color': cat_info['color'],
                'description': cat_info['description']
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Prediction error: {str(e)}'})
            
    return render_template('predict.html', cities=cities_list)

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    if lstm_model is None or lstm_scalers is None or lstm_city_map is None:
        return jsonify({'status': 'error', 'message': 'LSTM Model assets are not initialized.'})

    try:
        city_name = request.form.get('city')
        if not city_name:
            return jsonify({'status': 'error', 'message': 'City name is required.'})

        if city_name not in lstm_city_map:
            return jsonify({'status': 'error', 'message': f'City {city_name} is not supported by LSTM model.'})

        cache_path = os.path.join(BASE_DIR, 'data', 'lstm', 'latest_72h_cache.csv')
        if not os.path.exists(cache_path):
            return jsonify({'status': 'error', 'message': '72-hour hourly cache is not loaded.'})
        df_local_cache = pd.read_csv(cache_path)
        df_local_cache['Datetime'] = pd.to_datetime(df_local_cache['Datetime'])

        # Get city-specific data from cache
        city_data = df_local_cache[df_local_cache['City'] == city_name].sort_values(by='Datetime')
        if len(city_data) < 72:
            return jsonify({'status': 'error', 'message': f'Insufficient historical data for {city_name} (Got {len(city_data)} hours, need 72).'})

        # Take the last 72 hours
        latest_72h = city_data.tail(72).copy()

        # Check for user overrides (for "what-if" testing on the UI)
        def get_override(name, df_val):
            val_str = request.form.get(name, '').strip()
            return float(val_str) if val_str else df_val

        last_idx = latest_72h.index[-1]
        latest_72h.at[last_idx, 'AQI'] = get_override('today_aqi', latest_72h.at[last_idx, 'AQI'])
        latest_72h.at[last_idx, 'PM2.5'] = get_override('today_pm25', latest_72h.at[last_idx, 'PM2.5'])

        # Prepare feature columns in exact order
        feature_cols_list = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        features_matrix = latest_72h[feature_cols_list].copy()

        # Dynamically calculate temporal features for the sequence
        hours = latest_72h['Datetime'].dt.hour
        features_matrix['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        features_matrix['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Scale input features
        feature_scaler = lstm_scalers['feature_scaler']
        scaled_features = feature_scaler.transform(features_matrix)

        # Prepare inputs for Keras model
        city_id = lstm_city_map[city_name]
        seq_data = np.expand_dims(scaled_features, axis=0).astype(np.float32)
        city_data_input = np.array([[city_id]], dtype=np.int32)

        # Run Keras inference (direct model.predict)
        pred_scaled = lstm_model.predict(
            [seq_data, city_data_input], verbose=0
        )[0][0]

        # Inverse-transform to real AQI units
        target_scaler = lstm_scalers['target_scaler']
        pred_aqi = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
        pred_aqi = max(0.0, min(pred_aqi, 500.0))

        cat_info = get_aqi_category(pred_aqi)

        # Compute confidence interval from residual standard deviations
        ci_lower = pred_aqi
        ci_upper = pred_aqi
        if lstm_residual_stats:
            band_stds = lstm_residual_stats.get('validation', {}).get('residual_stds_by_band', {})
            if pred_aqi <= 50:
                band_label = "Good (0-50)"
            elif pred_aqi <= 100:
                band_label = "Satisfactory (51-100)"
            elif pred_aqi <= 200:
                band_label = "Moderate (101-200)"
            elif pred_aqi <= 300:
                band_label = "Poor (201-300)"
            elif pred_aqi <= 400:
                band_label = "Very Poor (301-400)"
            else:
                band_label = "Severe (401+)"

            std = band_stds.get(band_label, 15.0)
            ci_lower = max(0.0, pred_aqi - std)
            ci_upper = min(500.0, pred_aqi + std)

        return jsonify({
            'status': 'success',
            'predicted_aqi': round(pred_aqi, 1),
            'predicted_category': cat_info['label'],
            'color': cat_info['color'],
            'description': cat_info['description'],
            'ci_lower': round(ci_lower, 1),
            'ci_upper': round(ci_upper, 1)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'LSTM Prediction error: {str(e)}'})

@app.route('/compare')
def compare():
    cities_list = list(df_clean['City'].unique()) if df_clean is not None else []
    return render_template('compare.html', cities=cities_list)

@app.route('/api/city_stats/<city>')
def city_stats(city):
    if df_clean is not None:
        city_data = df_clean[df_clean['City'] == city].sort_values(by='Date')
        if not city_data.empty:
            recent_30 = city_data.tail(30)
            return jsonify({
                'status': 'success',
                'dates': list(recent_30['Date']),
                'aqi': list(recent_30['AQI']),
                'pm25': list(recent_30['PM2.5']),
                'pm10': list(recent_30['PM10']),
                'so2': list(recent_30['SO2']),
                'no2': list(recent_30['NO2']),
                'co': list(recent_30['CO']),
                'o3': list(recent_30['O3']),
                'avg_aqi': round(city_data['AQI'].mean(), 1)
            })
    return jsonify({'status': 'error', 'message': 'City data not found.'})

@app.route('/api/v1/anomalies')
def get_anomalies():
    """
    Returns anomaly status for all monitored cities.
    Reads the last 24 readings per city from the live DB and runs hybrid detection.
    """
    from utils.db_manager import get_latest_readings
    from utils.anomaly import detect_anomaly

    results = []
    city_map_path = os.path.join(BASE_DIR, 'data', 'lstm', 'city_map.json')
    if os.path.exists(city_map_path):
        cities = json.load(open(city_map_path)).keys()
    else:
        cities = list(CITY_COORDINATES.keys())

    for city in cities:
        try:
            recent = get_latest_readings(city, limit=25)
            if not recent:
                continue
            history = [r['aqi'] for r in recent[:-1] if r.get('aqi') is not None]
            current_aqi = recent[-1].get('aqi')
            if current_aqi is None:
                continue
            anomaly = detect_anomaly(city, current_aqi, history)
            if anomaly['is_anomaly']:
                results.append({
                    'city': city,
                    'current_aqi': current_aqi,
                    'severity': anomaly['severity'],
                    'triggers': anomaly['triggers'],
                    'delta': anomaly['details']['rate_of_change'].get('delta'),
                    'z_score': anomaly['details']['zscore'].get('z_score'),
                })
        except Exception as e:
            print(f"[WARNING] Anomaly check failed for {city}: {e}")

    # Sort: critical first, then by AQI descending
    results.sort(key=lambda x: (0 if x['severity'] == 'critical' else 1, -x['current_aqi']))
    return jsonify({'status': 'success', 'anomalies': results, 'count': len(results)})


@app.route('/alerts')
def alerts_page():
    from utils.db_manager import get_recent_alerts
    from utils.alerts import _is_configured
    
    bot_configured = _is_configured()
    recent_logs = get_recent_alerts(limit=30)
    cities_list = sorted(list(CITY_COORDINATES.keys()))
    
    return render_template(
        'alerts.html',
        bot_configured=bot_configured,
        logs=recent_logs,
        cities=cities_list
    )


@app.route('/api/v1/send_test_alert', methods=['POST'])
def send_test_alert():
    from utils.alerts import send_aqi_alert, _is_configured
    from utils.db_manager import insert_alert_log
    import datetime

    if not _is_configured():
        return jsonify({'status': 'error', 'message': 'Telegram Bot is not configured.'}), 400

    try:
        data = request.json or {}
        city = data.get('city', 'Delhi')
        aqi = float(data.get('aqi', 325))
        severity = data.get('severity', 'critical')
        delta = data.get('delta', None)
        if delta is not None:
            delta = float(delta)

        # Get category matching AQI
        cat_info = get_aqi_category(aqi)
        category = cat_info['label']

        # Send test alert
        success = send_aqi_alert(
            city=city,
            aqi=aqi,
            category=category,
            severity=severity,
            delta=delta
        )

        if success:
            dt_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            insert_alert_log(city, dt_str, aqi, severity)
            return jsonify({'status': 'success', 'message': 'Test alert sent.'})
        else:
            return jsonify({'status': 'error', 'message': 'API delivery failure.'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/insights')
def insights_page():
    from utils.policy_analysis import get_feature_importance
    
    importances = get_feature_importance()
    cities_list = sorted(list(CITY_COORDINATES.keys()))
    
    return render_template(
        'insights.html',
        importances=importances,
        cities=cities_list
    )


@app.route('/api/v1/policy_insights')
def policy_insights():
    from utils.policy_analysis import get_temporal_insights, get_policy_recommendations
    
    city = request.args.get('city', 'Delhi')
    temporal = get_temporal_insights(city)
    recommendations = get_policy_recommendations(city)
    
    return jsonify({
        'status': 'success',
        'temporal': temporal,
        'recommendations': recommendations
    })


@app.route('/api/v1/live_summary')
def live_summary():
    """
    Returns a real-time summary of live AQI data.
    Reads from Supabase when configured, falls back to local SQLite.
    """
    from utils.db_manager import (
        DB_PATH, is_supabase_enabled,
        SUPABASE_URL, SUPABASE_KEY
    )
    import sqlite3, requests as req_lib

    rows = []
    total = 0
    last_ts = "N/A"
    source = "supabase" if is_supabase_enabled() else "sqlite"

    try:
        if is_supabase_enabled():
            # ── Supabase path ──────────────────────────────────────────────
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
            # Latest per city using Supabase PostgREST ordering
            r = req_lib.get(
                f"{SUPABASE_URL}/rest/v1/hourly_readings",
                headers=headers,
                params={"order": "datetime.desc", "limit": 10000},
                timeout=8
            )
            if r.status_code == 200:
                all_data = r.json()
                # Deduplicate: keep latest per city
                seen = {}
                for rec in all_data:
                    city = rec.get("city")
                    if city and city not in seen and rec.get("aqi") is not None:
                        seen[city] = rec
                rows = sorted(seen.values(), key=lambda x: x.get("aqi", 0), reverse=True)
                total = len(all_data)
                last_ts = all_data[0]["datetime"] if all_data else "N/A"
            else:
                source = "sqlite (supabase error)"
                raise Exception(f"Supabase returned {r.status_code}")
        else:
            raise Exception("Supabase not configured")

    except Exception as e:
        # ── SQLite fallback ────────────────────────────────────────────────
        source = "sqlite"
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT city, aqi, pm25, datetime
                FROM hourly_readings
                WHERE datetime IN (
                    SELECT MAX(datetime) FROM hourly_readings GROUP BY city
                )
                AND aqi IS NOT NULL
                ORDER BY aqi DESC
            """)
            rows = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) FROM hourly_readings")
            total = cur.fetchone()[0]
            cur.execute("SELECT MAX(datetime) FROM hourly_readings")
            last_ts = cur.fetchone()[0] or "N/A"
            conn.close()
        except Exception as db_err:
            return jsonify({'status': 'error', 'message': str(db_err)}), 500

    cities_live = []
    for r in rows:
        aqi_val = r.get('aqi') or 0
        cat = get_aqi_category(aqi_val)
        cities_live.append({
            'city': r['city'],
            'aqi': round(float(aqi_val), 0),
            'category': cat['label'],
            'color': cat['color'],
            'datetime': r.get('datetime', 'N/A')
        })

    return jsonify({
        'status': 'success',
        'source': source,
        'top_cities': cities_live[:5],
        'all_cities': cities_live,
        'total_readings': total,
        'last_updated': last_ts
    })


@app.route('/api/v1/health_advice')
def health_advice():
    """Returns health advisory text for a given AQI value."""
    aqi = request.args.get('aqi', 0, type=float)
    advisories = [
        (50,  'Good',         '#29f1c3', 'Air quality is satisfactory. Outdoor activities are safe for everyone.'),
        (100, 'Satisfactory', '#9dff00', 'Unusually sensitive individuals may experience minor discomfort. Most people can enjoy outdoor activities.'),
        (200, 'Moderate',     '#f5a623', 'People with respiratory or heart disease, the elderly, and children should limit prolonged outdoor exertion.'),
        (300, 'Poor',         '#ff6b6b', 'Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion.'),
        (400, 'Very Poor',    '#c084fc', 'Health alert: everyone may experience more serious health effects. Wear N95 masks outdoors.'),
        (500, 'Severe',       '#ff2d2d', 'Health warning of emergency conditions. Entire population is more likely to be affected. Stay indoors.'),
    ]
    for limit, label, color, advice in advisories:
        if aqi <= limit:
            return jsonify({'label': label, 'color': color, 'advice': advice, 'aqi': aqi})
    return jsonify({'label': 'Severe', 'color': '#ff2d2d',
                    'advice': 'Hazardous. Remain indoors and keep windows closed.', 'aqi': aqi})


@app.route('/api/v1/ingest', methods=['POST'])
def trigger_ingest():
    token = request.headers.get('X-Ingest-Token')
    expected_token = os.environ.get("INGEST_TOKEN", "aqi-forecaster-secure-token-2026")
    
    if not token or token != expected_token:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
    from threading import Thread
    from data_ingestion import process_and_ingest
    
    # Start ingestion asynchronously in a background thread to prevent HTTP timeouts
    thread = Thread(target=process_and_ingest)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Live data ingestion triggered successfully in the background.'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
