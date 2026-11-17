"""
run_eda.py - Exploratory Data Analysis (EDA) for the Indian AQI ML Project.
Loads the raw data, cleans it, generates visualization charts, and saves them to the outputs/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(raw_data_dir, outputs_dir):
    print("--------------------------------------------------")
    print("      Starting Exploratory Data Analysis...       ")
    print("--------------------------------------------------")
    
    city_day_path = os.path.join(raw_data_dir, "city_day.csv")
    if not os.path.exists(city_day_path):
        print(f"[ERROR] Could not find {city_day_path}.")
        return
        
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("[INFO] Loading dataset...")
    df = pd.read_csv(city_day_path)
    
    # Pre-cleaning for EDA
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Autumn', 11: 'Autumn'
    })
    
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    
    # Impute missing values for plotting
    for col in pollutant_cols:
        df[col] = df.groupby('City')[col].transform(
            lambda x: x.ffill().bfill().fillna(x.median())
        ).clip(lower=0)
        
    # Drop rows where AQI is missing
    df = df.dropna(subset=['AQI'])
    
    # Plot 1: Top 15 Polluted Indian Cities (Average AQI)
    print("[INFO] Plotting Top 15 Polluted Cities...")
    plt.figure(figsize=(10, 6))
    city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=city_aqi.values, y=city_aqi.index, hue=city_aqi.index, legend=False, palette='coolwarm')
    plt.title('Top 15 Most Polluted Indian Cities (Average AQI, 2015-2020)', fontsize=14)
    plt.xlabel('Average AQI', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'top_15_polluted_cities.png'), dpi=150)
    plt.close()
    
    # Plot 2: Average AQI Trend Over Years
    print("[INFO] Plotting Yearly AQI Trend...")
    plt.figure(figsize=(10, 5))
    yearly_aqi = df.groupby('Year')['AQI'].mean()
    plt.plot(yearly_aqi.index, yearly_aqi.values, marker='o', color='#0078D4', linewidth=2.5)
    plt.title('Average AQI Trend Over Years in India', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average AQI', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(yearly_aqi.index)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'yearly_aqi_trend.png'), dpi=150)
    plt.close()
    
    # Plot 3: Seasonal AQI Distribution
    print("[INFO] Plotting Seasonal AQI Distribution...")
    plt.figure(figsize=(9, 6))
    season_order = ['Spring', 'Monsoon', 'Autumn', 'Winter']
    sns.boxplot(x='Season', y='AQI', data=df, order=season_order, palette='Set2', showfliers=False)
    plt.title('Seasonal AQI Distribution in India (Outliers Hidden)', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'seasonal_aqi_distribution.png'), dpi=150)
    plt.close()
    
    # Plot 4: Pollutant Correlation Heatmap
    print("[INFO] Plotting Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df[pollutant_cols + ['AQI']].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt='.2f', square=True, linewidths=0.5)
    plt.title('Pollutant Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'pollutant_correlation_matrix.png'), dpi=150)
    plt.close()
    
    # Plot 5: AQI Category distribution (AQI Bucket)
    print("[INFO] Plotting AQI Category distribution...")
    plt.figure(figsize=(8, 8))
    # Standard CPCB order
    category_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    bucket_counts = df['AQI_Bucket'].value_counts()
    # Reindex to order correctly
    bucket_counts = bucket_counts.reindex([c for c in category_order if c in bucket_counts.index])
    
    # Colors matching CPCB
    cpcb_colors = ['#00E400', '#98F51F', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']
    colors_map = [cpcb_colors[category_order.index(c)] for c in bucket_counts.index]
    
    plt.pie(bucket_counts.values, labels=bucket_counts.index, autopct='%1.1f%%', startangle=140, colors=colors_map, wedgeprops={'edgecolor': 'white'})
    plt.title('Distribution of AQI Categories in India', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'aqi_category_distribution.png'), dpi=150)
    plt.close()
    
    print("[SUCCESS] All EDA visualizations saved to the 'outputs/' directory!")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_eda("data/raw", "outputs")
