"""
recover_lstm.py - Loads the best checkpoint from interrupted training,
evaluates on val/test sets, exports TFLite, and saves residual stats.
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

# Add project root to path for utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    return np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100


def get_aqi_category_simple(aqi):
    """Simplified CPCB category lookup."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def evaluate_model(model, X, cities, y_scaled, target_scaler, city_map, split_name="Test"):
    y_pred_scaled = model.predict([X, cities], batch_size=512, verbose=0).flatten()

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    y_pred = np.clip(y_pred, 0, 500)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {split_name} Set Evaluation (Real AQI Units)")
    print(f"{'=' * 60}")
    print(f"  RMSE:  {rmse:.2f} AQI points  (target: < 25)")
    print(f"  MAE:   {mae:.2f} AQI points  (target: < 18)")
    print(f"  SMAPE: {smape:.2f}%           (target: < 15%)")

    # Category Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred)
                  if get_aqi_category_simple(yt) == get_aqi_category_simple(yp))
    cat_accuracy = correct / len(y_true) * 100
    print(f"  Category Accuracy: {cat_accuracy:.2f}%  (target: > 85%)")

    # Per-city metrics
    inv_city_map = {v: k for k, v in city_map.items()}
    print(f"\n  Per-City Breakdown:")
    print(f"  {'City':<25} {'RMSE':>8} {'MAE':>8} {'SMAPE':>8} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    city_metrics = {}
    for cid in np.unique(cities):
        mask = cities == cid
        if mask.sum() < 10:
            continue
        city_name = inv_city_map.get(cid, f"ID_{cid}")
        c_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        c_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        c_smape = compute_smape(y_true[mask], y_pred[mask])
        print(f"  {city_name:<25} {c_rmse:>8.2f} {c_mae:>8.2f} {c_smape:>7.2f}% {mask.sum():>8,}")
        city_metrics[city_name] = {'rmse': float(c_rmse), 'mae': float(c_mae), 'smape': float(c_smape)}

    # Residual std by severity band
    print(f"\n  Residual Std by AQI Severity Band:")
    residuals = y_pred - y_true
    bands = [
        ("Good (0-50)", 0, 50),
        ("Satisfactory (51-100)", 51, 100),
        ("Moderate (101-200)", 101, 200),
        ("Poor (201-300)", 201, 300),
        ("Very Poor (301-400)", 301, 400),
        ("Severe (401+)", 401, 999),
    ]
    residual_stds = {}
    for label, lo, hi in bands:
        band_mask = (y_true >= lo) & (y_true <= hi)
        if band_mask.sum() > 10:
            band_std = np.std(residuals[band_mask])
            print(f"  {label:<25} +/-{band_std:.2f} AQI  (n={band_mask.sum():,})")
            residual_stds[label] = float(band_std)
        else:
            print(f"  {label:<25} insufficient data (n={band_mask.sum()})")

    return {
        'rmse': float(rmse), 'mae': float(mae), 'smape': float(smape),
        'category_accuracy': float(cat_accuracy),
        'city_metrics': city_metrics,
        'residual_stds': residual_stds,
        'y_true': y_true, 'y_pred': y_pred
    }


def main():
    print("=" * 60)
    print("  LSTM Recovery: Evaluate + Export from Best Checkpoint")
    print("=" * 60)

    data_dir = 'data/lstm'
    models_dir = 'models'
    outputs_dir = 'outputs'
    os.makedirs(outputs_dir, exist_ok=True)

    # Load checkpoint
    checkpoint_path = os.path.join(models_dir, 'lstm_best.keras')
    print(f"\n[INFO] Loading best checkpoint: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
    model.summary()

    # Load data
    print("\n[INFO] Loading preprocessed sequences...")
    val_data = np.load(os.path.join(data_dir, 'val_sequences.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_sequences.npz'))

    X_val, y_val, cities_val = val_data['X'], val_data['y'], val_data['cities']
    X_test, y_test, cities_test = test_data['X'], test_data['y'], test_data['cities']

    print(f"  Val:    X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:   X={X_test.shape}, y={y_test.shape}")

    # Load city map and scalers
    with open(os.path.join(data_dir, 'city_map.json')) as f:
        city_map = json.load(f)

    with open(os.path.join(data_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    target_scaler = scalers['target_scaler']

    # Evaluate
    val_results = evaluate_model(model, X_val, cities_val, y_val, target_scaler, city_map, "Validation")
    test_results = evaluate_model(model, X_test, cities_test, y_test, target_scaler, city_map, "Test")

    # Plot predictions
    print("\n[INFO] Generating prediction plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_true, y_pred = test_results['y_true'], test_results['y_pred']

    sample_idx = np.random.choice(len(y_true), min(500, len(y_true)), replace=False)
    axes[0].scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, color='#00f2fe')
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect prediction')
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title('LSTM: Predicted vs Actual AQI', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    n_show = min(200, len(y_true))
    axes[1].plot(range(n_show), y_true[:n_show], label='Actual', color='#00f2fe', linewidth=1)
    axes[1].plot(range(n_show), y_pred[:n_show], label='Predicted', color='#FF7E00', linewidth=1, alpha=0.8)
    axes[1].set_xlabel('Time Step (hours)')
    axes[1].set_ylabel('AQI')
    axes[1].set_title('LSTM Forecast vs Ground Truth (First 200 Hours)', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, 'lstm_predictions.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved to {plot_path}")

    # Save final Keras model
    keras_path = os.path.join(models_dir, 'lstm_aqi_final.keras')
    model.save(keras_path)
    print(f"\n[INFO] Keras model saved to {keras_path} ({os.path.getsize(keras_path) / 1024:.1f} KB)")

    # Convert to TFLite
    print("\n[INFO] Converting to TFLite (float16 quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = os.path.join(models_dir, 'lstm_aqi.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite model saved to {tflite_path} ({os.path.getsize(tflite_path) / 1024:.1f} KB)")

    # Save residual statistics
    residual_stats_path = os.path.join(models_dir, 'lstm_residual_stats.json')
    with open(residual_stats_path, 'w') as f:
        json.dump({
            'validation': {
                'rmse': val_results['rmse'],
                'mae': val_results['mae'],
                'smape': val_results['smape'],
                'category_accuracy': val_results['category_accuracy'],
                'residual_stds_by_band': val_results['residual_stds']
            },
            'test': {
                'rmse': test_results['rmse'],
                'mae': test_results['mae'],
                'smape': test_results['smape'],
                'category_accuracy': test_results['category_accuracy'],
                'residual_stds_by_band': test_results['residual_stds']
            }
        }, f, indent=2)
    print(f"\n[INFO] Residual statistics saved to {residual_stats_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("  RECOVERY COMPLETE")
    print("=" * 60)
    print(f"  Test RMSE:  {test_results['rmse']:.2f} AQI points")
    print(f"  Test MAE:   {test_results['mae']:.2f} AQI points")
    print(f"  Test SMAPE: {test_results['smape']:.2f}%")
    print(f"  Test Category Accuracy: {test_results['category_accuracy']:.2f}%")
    print(f"\n  Model files:")
    print(f"    Keras:  {keras_path}")
    print(f"    TFLite: {tflite_path}")
    print(f"    Stats:  {residual_stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
