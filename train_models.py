"""
train_models.py - Model training script for the Indian AQI ML Project.
Loads the preprocessed data, splits it into train/test based on time,
trains regression (AQI prediction) and classification (AQI category) models,
evaluates them, and saves the final models to the models/ directory.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score

def train_and_evaluate(processed_data_path, models_dir):
    print("--------------------------------------------------")
    print("        Starting Model Training Pipeline...       ")
    print("--------------------------------------------------")
    
    if not os.path.exists(processed_data_path):
        print(f"[ERROR] Cleaned dataset not found at {processed_data_path}. Please run preprocess.py first.")
        return
        
    os.makedirs(models_dir, exist_ok=True)
    
    print("[INFO] Loading cleaned dataset...")
    df = pd.read_csv(processed_data_path)
    
    # 1. Define feature columns and target variables
    # Exclude non-numeric or future-leakage target columns
    feature_cols = [
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene',
        'Month', 'Day', 'DayOfWeek', 'IsWeekend',
        'AQI_lag_1', 'AQI_lag_2', 'AQI_lag_3', 'AQI_lag_7',
        'PM25_lag_1', 'PM25_lag_3',
        'AQI_rolling_3d', 'AQI_rolling_7d', 'AQI_rolling_14d',
        'PM_ratio', 'NOx_ratio'
    ]
    
    # Include city encoding if city is in the features
    print("[INFO] Encoding categorical features...")
    le_city = LabelEncoder()
    df['City_encoded'] = le_city.fit_transform(df['City'])
    feature_cols.append('City_encoded')
    
    # Encode Season if exists
    if 'Season' in df.columns:
        df = pd.get_dummies(df, columns=['Season'], drop_first=True)
        # Add generated season dummy columns
        season_cols = [c for c in df.columns if c.startswith('Season_')]
        feature_cols.extend(season_cols)
        
    # Targets
    target_reg = 'AQI'
    target_cls = 'AQI_Bucket'
    
    X = df[feature_cols]
    y_reg = df[target_reg]
    y_cls = df[target_cls]
    
    # 2. Time-Based Train/Test Split
    # Since dataset covers 2015-2020, we use 2015-2019 for training, 2020 for testing
    print("[INFO] Performing time-based data splitting...")
    train_mask = df['Year'] < 2020
    test_mask = df['Year'] == 2020
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train_reg, y_test_reg = y_reg[train_mask], y_reg[test_mask]
    y_train_cls, y_test_cls = y_cls[train_mask], y_cls[test_mask]
    
    print(f"Training set: {X_train.shape[0]} samples (Years 2015-2019)")
    print(f"Test set:     {X_test.shape[0]} samples (Year 2020)")
    
    if X_test.shape[0] == 0:
        print("[WARNING] Test set is empty! Defaulting to random split (80/20).")
        X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
            X, y_reg, y_cls, test_size=0.2, random_state=42
        )
        
    # Scale features (Standard scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. AQI REGRESSION (Predict AQI Value)
    print("\n[INFO] Training Regression Models for AQI Prediction...")
    
    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg) # Tree-based models don't strictly require scaling
    rf_preds = rf_reg.predict(X_test)
    
    # XGBoost Regressor
    xgb_reg = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train, y_train_reg)
    xgb_preds = xgb_reg.predict(X_test)
    
    # Regression Evaluation
    print("\n=== REGRESSION EVALUATION ===")
    print("Random Forest Regressor:")
    print(f"  MAE:  {mean_absolute_error(y_test_reg, rf_preds):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, rf_preds)):.2f}")
    print(f"  R2:   {r2_score(y_test_reg, rf_preds):.4f}")
    
    print("XGBoost Regressor:")
    print(f"  MAE:  {mean_absolute_error(y_test_reg, xgb_preds):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_reg, xgb_preds)):.2f}")
    print(f"  R2:   {r2_score(y_test_reg, xgb_preds):.4f}")
    
    # 4. AQI CLASSIFICATION (Predict AQI Category)
    print("\n[INFO] Training Classification Models for AQI Categories...")
    
    # Encode categories
    le_aqi = LabelEncoder()
    # Fit on all categories
    y_train_cls_encoded = le_aqi.fit_transform(y_train_cls)
    y_test_cls_encoded = le_aqi.transform(y_test_cls)
    
    # XGBoost Classifier
    xgb_cls = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb_cls.fit(X_train, y_train_cls_encoded)
    cls_preds = xgb_cls.predict(X_test)
    
    print("\n=== CLASSIFICATION EVALUATION ===")
    print(f"XGBoost Classifier Accuracy: {accuracy_score(y_test_cls_encoded, cls_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls_encoded, cls_preds, target_names=le_aqi.classes_))
    
    # 5. Save Final Models
    print("\n[INFO] Saving models and encoders...")
    joblib.dump(xgb_reg, os.path.join(models_dir, 'xgboost_aqi_predictor.pkl'))
    joblib.dump(xgb_cls, os.path.join(models_dir, 'xgboost_aqi_classifier.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(le_aqi, os.path.join(models_dir, 'label_encoder.pkl'))
    joblib.dump(le_city, os.path.join(models_dir, 'city_encoder.pkl'))
    
    # Save the feature columns list so Web App knows the exact order
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.pkl'))
    
    print("[SUCCESS] All models saved successfully to the 'models/' directory!")
    print("--------------------------------------------------")

if __name__ == "__main__":
    train_and_evaluate("data/processed/cleaned_aqi_data.csv", "models")
