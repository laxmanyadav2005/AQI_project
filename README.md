# 🌍 IND-AQI Platform — Air Quality Index Monitor for Indian Cities

An end-to-end, production-ready AQI forecasting and monitoring system for **26 major Indian cities**, built with Flask, XGBoost, Keras LSTM, and Supabase. It follows the official **CPCB (Central Pollution Control Board) AQI standard**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| **CPCB AQI Calculator** | Official piecewise linear interpolation for PM2.5, PM10, NO₂, SO₂, CO, O₃ |
| **Daily AQI Prediction** | XGBoost regression + classification model (2015–2020 Kaggle dataset) |
| **Hourly LSTM Forecast** | Multi-city Keras LSTM with station embeddings, 72-hour rolling window |
| **Live Ingestion** | Pulls real-time pollution data from OpenWeather API for all 26 cities |
| **Anomaly Detection** | Z-score + rate-of-change spike detection with alert cooldowns |
| **Supabase Cloud DB** | PostgreSQL via Supabase REST API, with SQLite local fallback |
| **Interactive Dashboard** | Live status bar, city rankings, health advisories, AQI health scale |
| **Policy Insights** | Feature importance, health advice, comparative city analysis |
| **Telegram Alerts** | Optional bot alerts when anomalous AQI spikes are detected |
| **Render-ready** | `Procfile` + `render.yaml` for one-click Render.com deployment |

---

## 🗺️ Covered Cities (26)

Delhi · Mumbai · Bengaluru · Kolkata · Chennai · Hyderabad · Ahmedabad · Jaipur · Lucknow · Patna · Visakhapatnam · Gurugram · Amritsar · Thiruvananthapuram · Coimbatore · Guwahati · Bhopal · Chandigarh · Nagpur · Vadodara · Surat · Indore · Kochi · Shillong · Amaravati · Faridabad

---

## 📁 Project Structure

```
aqi_project/
├── app.py                    # Flask app — routes, API endpoints, model inference
├── data/
│   ├── processed/
│   │   ├── cleaned_aqi_data.csv   # Pre-cleaned daily dataset (committed)
│   │   └── aqi_live.db            # SQLite live readings (committed as seed)
│   └── lstm/
│       ├── latest_72h_cache.csv   # 72-hour rolling window for LSTM (committed)
│       ├── city_map.json          # City → index mapping (committed)
│       └── scalers.pkl            # Feature scalers for inference (committed)
├── models/
│   ├── lstm_aqi.tflite            # Lightweight LSTM (committed, 0.1 MB)
│   ├── lstm_residual_stats.json   # Residual correction stats (committed)
│   ├── city_encoder.pkl           # City label encoder (committed)
│   ├── feature_cols.pkl           # Feature column list (committed)
│   ├── label_encoder.pkl          # AQI category encoder (committed)
│   └── scaler.pkl                 # Daily model scaler (committed)
├── utils/
│   ├── aqi_calculator.py          # Official CPCB AQI formula
│   ├── db_manager.py              # Dual-mode DB: Supabase + SQLite fallback
│   ├── anomaly.py                 # Spike detection logic
│   ├── alerts.py                  # Telegram alert dispatcher
│   └── policy_analysis.py        # Feature importance & insights
├── templates/                # Jinja2 HTML templates (7 pages)
├── static/                   # CSS, JS, icons, PWA manifest
├── preprocess.py             # Daily dataset cleaning pipeline
├── lstm_preprocess.py        # Hourly LSTM sequence preparation
├── train_models.py           # XGBoost training pipeline
├── train_lstm.py             # Keras LSTM training pipeline
├── data_ingestion.py         # Live OpenWeather ingestion + alert loop
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Dev/training dependencies (adds pytest)
├── Procfile                  # Gunicorn startup for Render/Heroku
├── render.yaml               # Render.com deployment config
└── setup_env.ps1             # Windows venv setup helper
```

> **Large files not committed** (re-train locally):
> `models/xgboost_aqi_predictor.pkl` (0.65 MB), `models/lstm_aqi_final.keras` (0.43 MB),
> `models/xgboost_aqi_classifier.pkl` (2.35 MB), `data/raw/` (270+ MB Kaggle CSVs),
> `data/lstm/*.npz` sequence arrays.

---

## 🚀 Quick Start

### 1 — Clone and set up environment

```bash
git clone https://github.com/laxmanyadav2005/AQI_project.git
cd AQI_project
```

**Windows (PowerShell):**
```powershell
.\setup_env.ps1
.\aqi_env\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv aqi_env
source aqi_env/bin/activate
pip install -r requirements.txt
```

### 2 — Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENWEATHER_API_KEY (required for live ingestion)
```

### 3 — Run the dashboard

```bash
python app.py
# → Open http://127.0.0.1:5000
```

The app starts immediately using the **pre-committed** cleaned dataset, 72-hour cache, LSTM TFLite model, and scalers — no retraining needed.

---

## 🧠 Retraining Models (optional)

Only needed if you want to retrain on new data. Requires the Kaggle dataset files in `data/raw/`:
- [`city_day.csv` and `city_hour.csv`](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

```bash
# 1. Clean daily data
python preprocess.py

# 2. Train XGBoost models (creates xgboost_aqi_predictor.pkl, xgboost_aqi_classifier.pkl)
python train_models.py

# 3. Prepare LSTM sequences (creates train/val/test .npz files)
python lstm_preprocess.py

# 4. Train LSTM (creates lstm_aqi_final.keras, lstm_aqi.tflite)
python train_lstm.py
```

---

## 📡 Live Data Ingestion

```bash
python data_ingestion.py
```

Fetches live pollutant readings from the [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution) for all 26 cities, calculates CPCB AQI, writes to SQLite (or Supabase if configured), updates the 72-hour cache, and fires anomaly alerts.

Can also be triggered via the Flask endpoint:
```bash
curl -X POST http://127.0.0.1:5000/api/v1/ingest \
  -H "Authorization: Bearer YOUR_INGEST_TOKEN"
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard |
| `/predict` | GET/POST | Daily AQI prediction form (XGBoost) |
| `/predict_lstm` | POST | Hourly LSTM forecast (JSON) |
| `/compare` | GET | City-to-city comparison charts |
| `/insights` | GET | Policy insights & feature importance |
| `/alerts` | GET | Recent anomaly alert log |
| `/calculator` | GET | CPCB AQI calculator tool |
| `/api/city_stats/<city>` | GET | City stats JSON |
| `/api/v1/live_summary` | GET | Top polluted cities from Supabase/SQLite |
| `/api/v1/anomalies` | GET | Current anomaly detections |
| `/api/v1/health_advice` | GET | Health advisory for a given AQI (`?aqi=N`) |
| `/api/v1/policy_insights` | GET | Policy insights for a city (`?city=Delhi`) |
| `/api/v1/ingest` | POST | Trigger live ingestion (token-protected) |

---

## ☁️ Deployment

### Render.com (recommended free tier)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect this repo.
3. Render auto-detects `render.yaml`. Add environment variables in the Render dashboard:
   - `OPENWEATHER_API_KEY` (required)
   - `SUPABASE_URL` and `SUPABASE_KEY` (optional, for cloud DB)
   - `INGEST_TOKEN` (a long random string)
4. Deploy. The `Procfile` runs: `gunicorn app:app --workers 2 --timeout 120`

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENWEATHER_API_KEY` | ✅ Yes (for ingestion) | OpenWeather API key |
| `SUPABASE_URL` | ⬜ Optional | Supabase project URL |
| `SUPABASE_KEY` | ⬜ Optional | Supabase anon/service key |
| `INGEST_TOKEN` | ⬜ Optional | Token to protect `/api/v1/ingest` |
| `TELEGRAM_BOT_TOKEN` | ⬜ Optional | Telegram bot for AQI spike alerts |
| `TELEGRAM_CHAT_ID` | ⬜ Optional | Telegram chat/group ID |

---

## 📦 Dependencies

**Production (`requirements.txt`):**
`flask`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `plotly`, `requests`, `joblib`, `python-dotenv`, `gunicorn`, `python-docx`

**Dev/training only (`requirements-dev.txt`):** adds `matplotlib`, `seaborn`, `pytest`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Air Quality Data in India — Kaggle (Rohan Rao)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)
- [Central Pollution Control Board (CPCB) AQI Standard](https://cpcb.nic.in/)
- [Supabase](https://supabase.com/) for the cloud PostgreSQL backend
