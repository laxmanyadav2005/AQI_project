"""
preprocess.py - Preprocessing pipeline for the Indian AQI ML Project.
Loads the raw Kaggle dataset, performs cleaning, handles missing values,
merges weather features, and splits the data for machine learning models.
"""

import os
import pandas as pd
import numpy as np
import requests

def preprocess_data(raw_data_dir, processed_data_dir):
    print("--------------------------------------------------")
    print("   Starting Data Preprocessing Pipeline...   ")
    print("--------------------------------------------------")
    
    city_day_path = os.path.join(raw_data_dir, "city_day.csv")
    if not os.path.exists(city_day_path):
        print(f"[ERROR] Could not find {city_day_path}.")
        print("Please download the dataset from Kaggle and place it in the data/raw folder.")
        return
        
    print("[INFO] Loading city_day.csv...")
    df = pd.read_csv(city_day_path)
    
    # 1. Date conversion and feature extraction
    print("[INFO] Extracting time and seasonal features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Define Indian seasons
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Autumn', 11: 'Autumn'
    })
    
    # 2. Handle missing values
    print("[INFO] Cleaning and imputing missing pollutant values...")
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    
    # Forward fill missing values within each city, then fill remaining with city median
    for col in pollutant_cols:
        df[col] = df.groupby('City')[col].transform(
            lambda x: x.ffill().bfill().fillna(x.median())
        )
        # Ensure no negative values
        df[col] = df[col].clip(lower=0)
        
    # Drop rows where target variable (AQI) is still missing
    df = df.dropna(subset=['AQI'])
    
    # 3. Fetch/Merge weather data if available
    weather_path = os.path.join(raw_data_dir, "weather_india.csv")
    if os.path.exists(weather_path):
        print("[INFO] Merging historical weather data...")
        weather_df = pd.read_csv(weather_path)
        weather_df['Date'] = pd.to_datetime(weather_df['Date'])
        df = df.merge(weather_df, on=['Date', 'City'], how='left')
    else:
        print("[INFO] Weather dataset not found. Skipping merge step.")
        
    # 4. Create lag and rolling features
    print("[INFO] Engineering lag and rolling average features...")
    # Sort values to ensure correct rolling calculations
    df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    
    for lag in [1, 2, 3, 7]:
        df[f'AQI_lag_{lag}'] = df.groupby('City')['AQI'].shift(lag)
        df[f'PM25_lag_{lag}'] = df.groupby('City')['PM2.5'].shift(lag)
        
    for window in [3, 7, 14]:
        df[f'AQI_rolling_{window}d'] = df.groupby('City')['AQI'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
    # Ratios
    df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1)
    df['NOx_ratio'] = df['NO2'] / (df['NO'] + 1)
    
    # 5. Drop NaN rows resulting from shifts/lags
    df = df.dropna()
    
    # Save cleaned and engineered dataset
    os.makedirs(processed_data_dir, exist_ok=True)
    processed_path = os.path.join(processed_data_dir, "cleaned_aqi_data.csv")
    df.to_csv(processed_path, index=False)
    
    print(f"[SUCCESS] Preprocessed dataset saved to: {processed_path}")
    print(f"Shape of cleaned dataset: {df.shape}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    preprocess_data("data/raw", "data/processed")
