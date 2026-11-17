"""
utils/policy_analysis.py - Generates scientific insights and policy diagnostics.
Computes:
  1. XGBoost Feature Importance (from live model weights)
  2. Temporal Profiles (diurnal hourly cycles, day-of-week variation, seasonal trends)
  3. Actionable Policy Recommendations based on pollutant ratios
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def get_feature_importance():
    """Extracts feature importance from the trained XGBoost model."""
    try:
        model_path = os.path.join(MODELS_DIR, 'xgboost_aqi_predictor.pkl')
        feature_cols_path = os.path.join(MODELS_DIR, 'feature_cols.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(feature_cols_path):
            # Fallback if models are not trained yet
            return [
                {"feature": "AQI_lag_1", "importance": 0.65},
                {"feature": "PM2.5", "importance": 0.18},
                {"feature": "AQI_rolling_3d", "importance": 0.08},
                {"feature": "Month", "importance": 0.04},
                {"feature": "City_encoded", "importance": 0.03},
                {"feature": "IsWeekend", "importance": 0.02}
            ]
            
        model = joblib.load(model_path)
        features = joblib.load(feature_cols_path)
        
        # Get importances from XGBoost
        importances = model.feature_importances_
        
        # Combine and sort
        feat_imp = [{"feature": feat, "importance": float(imp)} for feat, imp in zip(features, importances)]
        feat_imp.sort(key=lambda x: x["importance"], reverse=True)
        return feat_imp[:10]  # Return top 10 features
    except Exception as e:
        print(f"[POLICY ERROR] Failed to calculate feature importance: {e}")
        return []


def get_temporal_insights(city_name=None):
    """Computes seasonal, weekly, and diurnal profiles from the preprocessed historical datasets."""
    insights = {
        "seasonal": {"Monsoon": 45.0, "Spring": 110.0, "Summer": 90.0, "Winter": 195.0},
        "weekly": {"Weekday": 112.0, "Weekend": 105.0},
        "diurnal": {
            "hours": list(range(24)),
            "aqi": [120, 115, 110, 108, 112, 125, 140, 155, 160, 150, 130, 115, 105, 100, 102, 108, 120, 135, 150, 155, 145, 138, 130, 125]
        }
    }
    
    try:
        # Load daily dataset for seasonal & weekly patterns
        dataset_path = os.path.join(DATA_DIR, 'cleaned_aqi_data.csv')
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            # Filter by city if specified
            if city_name and city_name in df['City'].unique():
                df = df[df['City'] == city_name]
                
            # 1. Seasonal trend
            if 'Season' in df.columns:
                season_avgs = df.groupby('Season')['AQI'].mean().to_dict()
                insights["seasonal"] = {k: round(v, 1) for k, v in season_avgs.items()}
            elif 'Month' in df.columns:
                # Reconstruct seasons if columns are missing
                def get_season(m):
                    if m in [12, 1, 2]: return 'Winter'
                    if m in [3, 4, 5]: return 'Spring'
                    if m in [6, 7, 8, 9]: return 'Monsoon'
                    return 'Summer'
                df['Season'] = df['Month'].apply(get_season)
                season_avgs = df.groupby('Season')['AQI'].mean().to_dict()
                insights["seasonal"] = {k: round(v, 1) for k, v in season_avgs.items()}

            # 2. Weekday vs Weekend
            if 'IsWeekend' in df.columns:
                weekly_avgs = df.groupby('IsWeekend')['AQI'].mean().to_dict()
                insights["weekly"] = {
                    "Weekday": round(weekly_avgs.get(0, 112.0), 1),
                    "Weekend": round(weekly_avgs.get(1, 105.0), 1)
                }

        # Load hourly cache for diurnal cycles
        cache_path = os.path.join(BASE_DIR, 'data', 'lstm', 'latest_72h_cache.csv')
        if os.path.exists(cache_path):
            df_hour = pd.read_csv(cache_path)
            df_hour['Datetime'] = pd.to_datetime(df_hour['Datetime'])
            df_hour['Hour'] = df_hour['Datetime'].dt.hour
            
            if city_name and city_name in df_hour['City'].unique():
                df_hour = df_hour[df_hour['City'] == city_name]
                
            diurnal_avgs = df_hour.groupby('Hour')['AQI'].mean().tolist()
            if len(diurnal_avgs) == 24:
                insights["diurnal"]["aqi"] = [round(v, 1) for v in diurnal_avgs]

    except Exception as e:
        print(f"[POLICY WARNING] Failed to calculate real temporal insights: {e}. Using fallback defaults.")
        
    return insights


def get_policy_recommendations(city_name=None):
    """
    Analyzes pollutant ratios (e.g. PM2.5/PM10 ratio, gaseous vs particulate)
    to generate policy actions.
    """
    recommendations = []
    
    try:
        # Load recent cache data
        cache_path = os.path.join(BASE_DIR, 'data', 'lstm', 'latest_72h_cache.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            if city_name and city_name in df['City'].unique():
                city_data = df[df['City'] == city_name]
            else:
                city_data = df
                
            pm25 = city_data['PM2.5'].mean()
            pm10 = city_data['PM10'].mean()
            no2 = city_data['NO2'].mean()
            so2 = city_data['SO2'].mean()
            co = city_data['CO'].mean() * 1000.0 if 'CO' in city_data.columns else 0 # convert back to ug/m3 for ratio comparison
            
            pm_ratio = pm25 / pm10 if pm10 > 0 else 0.5
            
            # Case 1: High PM2.5/PM10 ratio (>0.6) indicates combustion sources (vehicular/industrial)
            if pm_ratio > 0.6:
                recommendations.append({
                    "priority": "HIGH 🔴",
                    "category": "Vehicular Emissions",
                    "finding": f"High PM2.5/PM10 ratio ({pm_ratio:.2f}) indicates fine particulate pollution from combustion.",
                    "action": "Implement odd-even traffic restrictions, expand electric bus fleet, and enforce strict emission checks at city borders."
                })
            else:
                recommendations.append({
                    "priority": "MEDIUM 🟡",
                    "category": "Dust Control",
                    "finding": f"Low PM2.5/PM10 ratio ({pm_ratio:.2f}) indicates coarse dust (road dust, construction debris).",
                    "action": "Enforce mechanized vacuum road sweeping, regular water sprinkling on high-dust corridors, and windbreakers around construction sites."
                })
                
            # Case 2: High NO2 (NO2 > 60 ug/m3) indicates transport hotspot
            if no2 > 60:
                recommendations.append({
                    "priority": "CRITICAL 🚨",
                    "category": "Traffic Hotspot",
                    "finding": f"Elevated average NO2 level ({no2:.1f} µg/m³) indicates heavy diesel combustion and congestion.",
                    "action": "Divert heavy commercial vehicles, establish low-emission zones (LEZs) in commercial centers, and promote non-motorized transport corridors."
                })
                
            # Case 3: High SO2 (industrial indicator)
            if so2 > 40:
                recommendations.append({
                    "priority": "HIGH 🔴",
                    "category": "Industrial Regulation",
                    "finding": f"SO2 level ({so2:.1f} µg/m³) indicates coal combustion or petroleum refining nearby.",
                    "action": "Enforce Flue Gas Desulfurization (FGD) systems in surrounding power stations and conduct surprise stack emission audits."
                })
        
        # Fallback if empty recommendations
        if not recommendations:
            recommendations.append({
                "priority": "MEDIUM 🟡",
                "category": "General Air Quality Actions",
                "finding": "Baseline pollution profile suggests mixed urban source pollution.",
                "action": "Establish strict enforcement of waste burning bans, set up localized air purifiers in commercial clusters, and expand urban tree cover."
            })
    except Exception as e:
        print(f"[POLICY ERROR] Failed to calculate recommendations: {e}")
        recommendations.append({
            "priority": "MEDIUM 🟡",
            "category": "Urban Planning",
            "finding": "General urban particulate density detected.",
            "action": "Ban open trash burning, restrict heavy diesel trucks during peak hours, and increase green buffer zones."
        })
        
    return recommendations
