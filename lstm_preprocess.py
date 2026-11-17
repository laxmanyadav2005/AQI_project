"""
lstm_preprocess.py - Hourly data windowing pipeline for LSTM model.
Loads city_hour.csv, handles missing values, engineers temporal features,
creates sliding windows per-city (no cross-city leakage), and saves
train/val/test .npz files plus scalers for inference.

Features (9 total):
  [AQI, PM2.5, PM10, NO2, SO2, CO, O3, hour_sin, hour_cos]

Split strategy (chronological, no temporal leakage):
  Training:    2015-01-01 → 2018-12-31  (~70%)
  Validation:  2019-01-01 → 2019-09-30  (~15%)
  Test:        2019-10-01 → 2020-07-01  (~15%)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
LOOKBACK = 72   # hours (3 full diurnal cycles)
HORIZON = 1     # predict 1 hour ahead


def create_sequences_for_city(data, city_id, lookback=LOOKBACK, horizon=HORIZON):
    """Create sliding window sequences for a SINGLE city's contiguous data.

    Must be called per-city, never on a multi-city concatenated array —
    otherwise windows silently mix the tail of one city's series with the
    head of the next city's, corrupting both X and y.

    Args:
        data:    np.ndarray of shape (T, n_features), scaled.
        city_id: integer city identifier.
        lookback: number of past timesteps per window.
        horizon:  number of steps ahead for target.

    Returns:
        X: list of (lookback, n_features) arrays
        y: list of scalar targets (AQI at t+horizon, col 0)
        cities: list of city_id repeated per window
    """
    X, y, cities = [], [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i + horizon - 1, 0])  # target = AQI (col 0)
        cities.append(city_id)
    return X, y, cities


def preprocess_hourly(raw_path, output_dir):
    """Full preprocessing pipeline: load → clean → engineer → window → save.

    Args:
        raw_path:   Path to city_hour.csv
        output_dir: Directory for output .npz, scalers, city_map

    Returns:
        feature_scaler, target_scaler, city_map
    """
    print("=" * 60)
    print("  LSTM Preprocessing Pipeline")
    print("=" * 60)

    # --- 1. Load raw hourly data ---
    print("\n[INFO] Loading city_hour.csv...")
    df = pd.read_csv(raw_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(['City', 'Datetime']).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows across {df['City'].nunique()} cities")
    print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

    # --- 2. Imputation: ffill -> bfill -> city median -> clip ---
    print("\n[INFO] Imputing missing pollutant values...")
    for col in POLLUTANT_COLS:
        missing_before = df[col].isna().sum()
        df[col] = df.groupby('City')[col].transform(
            lambda x: x.ffill().bfill().fillna(x.median())
        ).clip(lower=0)
        # If entire city is NaN for a pollutant, fill with global median
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        missing_after = df[col].isna().sum()
        print(f"  {col}: {missing_before:,} missing -> {missing_after:,} remaining")

    # --- 3. AQI target imputation ---
    print("\n[INFO] Imputing AQI target values...")
    aqi_missing_before = df['AQI'].isna().sum()
    df['AQI'] = df.groupby('City')['AQI'].transform(
        lambda x: x.interpolate(method='linear', limit=6).ffill().bfill()
    )
    # Drop rows where AQI is still NaN (cities with no AQI data at all)
    df = df.dropna(subset=['AQI'])
    df['AQI'] = df['AQI'].clip(lower=0)
    aqi_missing_after = df['AQI'].isna().sum()
    print(f"  AQI: {aqi_missing_before:,} missing -> {aqi_missing_after:,} remaining")
    print(f"  Rows after AQI cleanup: {len(df):,}")

    # --- 4. Temporal features (cyclical encoding) ---
    # Both sin AND cos are required: sin alone cannot distinguish
    # e.g. 6 AM from 6 PM (their sin values are nearly identical).
    print("\n[INFO] Engineering temporal features (hour_sin, hour_cos)...")
    hour = df['Datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # --- 5. Encode cities ---
    print("\n[INFO] Encode city identifiers...")
    city_map = {c: i for i, c in enumerate(sorted(df['City'].unique()))}
    df['city_id'] = df['City'].map(city_map)
    n_cities = len(city_map)
    print(f"  {n_cities} cities encoded: {list(city_map.keys())}")

    # --- 6. Define feature columns ---
    # Order matters: AQI must be column 0 (used as target in windowing)
    feature_cols = ['AQI'] + POLLUTANT_COLS + ['hour_sin', 'hour_cos']
    n_features = len(feature_cols)
    print(f"\n[INFO] Feature columns ({n_features}): {feature_cols}")

    # --- 7. Time-based split ---
    print("\n[INFO] Splitting data chronologically...")
    train_df = df[df['Datetime'] < '2019-01-01'].copy()
    val_df = df[(df['Datetime'] >= '2019-01-01') & (df['Datetime'] < '2019-10-01')].copy()
    test_df = df[df['Datetime'] >= '2019-10-01'].copy()

    print(f"  Training:   {len(train_df):>8,} rows  (2015-01 to 2018-12)")
    print(f"  Validation: {len(val_df):>8,} rows  (2019-01 to 2019-09)")
    print(f"  Test:       {len(test_df):>8,} rows  (2019-10 to 2020-07)")

    # --- 8. Fit scalers on TRAINING data only ---
    print("\n[INFO] Fitting scalers on training data...")
    # Feature scaler: scales all 9 features for model input
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols])

    # Target scaler: scales only AQI for inverse-transforming predictions
    # back to real AQI units (not scaled [0,1] values)
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[['AQI']])

    # --- 9. Window EACH CITY SEPARATELY (no cross-city leakage) ---
    print("\n[INFO] Creating sliding windows per city...")
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        all_X, all_y, all_c = [], [], []
        city_counts = {}

        for city_id, city_df in split_df.groupby('city_id'):
            if len(city_df) < LOOKBACK + HORIZON:
                city_name = [k for k, v in city_map.items() if v == city_id][0]
                print(f"  [WARN] Skipping {city_name} in {split_name} "
                      f"(only {len(city_df)} rows, need {LOOKBACK + HORIZON})")
                continue

            scaled = feature_scaler.transform(city_df[feature_cols])
            X_c, y_c, c_c = create_sequences_for_city(scaled, city_id)
            all_X.extend(X_c)
            all_y.extend(y_c)
            all_c.extend(c_c)

            city_name = [k for k, v in city_map.items() if v == city_id][0]
            city_counts[city_name] = len(X_c)

        X_arr = np.array(all_X, dtype=np.float32)
        y_arr = np.array(all_y, dtype=np.float32)
        c_arr = np.array(all_c, dtype=np.int32)

        save_path = os.path.join(output_dir, f'{split_name}_sequences.npz')
        np.savez_compressed(save_path, X=X_arr, y=y_arr, cities=c_arr)

        print(f"\n  {split_name.upper()} set:")
        print(f"    Shape: X={X_arr.shape}, y={y_arr.shape}, cities={c_arr.shape}")
        print(f"    File:  {save_path} ({os.path.getsize(save_path) / 1024 / 1024:.1f} MB)")
        print(f"    Per-city windows: {city_counts}")

    # --- 10. Save scalers and city map ---
    scalers_path = os.path.join(output_dir, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': feature_cols
        }, f)
    print(f"\n[INFO] Scalers saved to {scalers_path}")

    city_map_path = os.path.join(output_dir, 'city_map.json')
    with open(city_map_path, 'w') as f:
        json.dump(city_map, f, indent=2)
    print(f"[INFO] City map saved to {city_map_path}")

    # --- 11. Summary statistics ---
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Cities:      {n_cities}")
    print(f"  Features:    {n_features}")
    print(f"  Lookback:    {LOOKBACK} hours")
    print(f"  Horizon:     {HORIZON} hour(s)")
    print(f"  Output dir:  {output_dir}")
    print("=" * 60)

    return feature_scaler, target_scaler, city_map


if __name__ == "__main__":
    preprocess_hourly("data/raw/city_hour.csv", "data/lstm")
