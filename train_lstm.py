"""
train_lstm.py - LSTM model training script for the Indian AQI Forecaster.
Builds a multi-station LSTM with station embeddings using the Keras Functional API,
trains on preprocessed hourly sequences, evaluates in REAL AQI units (not scaled),
and exports the model as both .keras and TFLite format for production inference.

Architecture:
  Sequence Branch:  LSTM(64) → LSTM(32)
  Embedding Branch: city_id → Embedding(8)
  Merge:            Concatenate → Dense(32) → Dropout → Dense(1)

Total Parameters: ~35K
TFLite Model Size: ~150 KB
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_lstm_model(n_features=9, lookback=72, n_cities=26, embed_dim=8):
    """Build multi-station LSTM with station embedding.

    n_cities is derived dynamically from city_map.json — never hardcoded.
    This ensures scaling from any subset to all 26 cities requires no
    code changes, only rerunning preprocessing + retraining.

    Args:
        n_features: Number of input features per timestep (9: AQI + 6 pollutants + hour_sin/cos)
        lookback:   Number of past timesteps in each window
        n_cities:   Number of unique stations (from city_map.json)
        embed_dim:  Dimension of station embedding vector

    Returns:
        Compiled Keras Model
    """
    # Sequence input branch
    seq_input = layers.Input(shape=(lookback, n_features), name='sequence')
    x = layers.LSTM(64, return_sequences=True, dropout=0.2,
                    recurrent_dropout=0.0, name='lstm_1')(seq_input)
    x = layers.LSTM(32, return_sequences=False, dropout=0.2,
                    recurrent_dropout=0.0, name='lstm_2')(x)

    # Station embedding branch
    city_input = layers.Input(shape=(1,), dtype='int32', name='city_id')
    city_embed = layers.Embedding(n_cities, embed_dim, name='city_embedding')(city_input)
    city_embed = layers.Flatten(name='city_flatten')(city_embed)

    # Merge branches
    merged = layers.Concatenate(name='merge')([x, city_embed])
    merged = layers.Dense(32, activation='relu', name='dense_1')(merged)
    merged = layers.Dropout(0.15, name='dropout_merge')(merged)
    output = layers.Dense(1, activation='linear', name='aqi_pred')(merged)

    model = Model(inputs=[seq_input, city_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',       # Robust to outliers in AQI spikes
        metrics=['mae']
    )
    return model


def compute_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    return np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100


def evaluate_model(model, X, cities, y_scaled, target_scaler, city_map, split_name="Test"):
    """Evaluate model in REAL AQI units after inverse-transforming.

    Metrics computed on scaled [0,1] values would look artificially good
    and are not comparable to the target thresholds (e.g. "<25 AQI points").
    """
    # Predict
    y_pred_scaled = model.predict([X, cities], batch_size=512, verbose=0).flatten()

    # Inverse-transform to real AQI units
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    # Clip predictions to valid AQI range
    y_pred = np.clip(y_pred, 0, 500)

    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {split_name} Set Evaluation (Real AQI Units)")
    print(f"{'=' * 60}")
    print(f"  RMSE:  {rmse:.2f} AQI points  (target: < 25)")
    print(f"  MAE:   {mae:.2f} AQI points  (target: < 18)")
    print(f"  SMAPE: {smape:.2f}%           (target: < 15%)")

    # CPCB Category Accuracy
    from utils.aqi_calculator import get_aqi_category
    correct = 0
    total = len(y_true)
    for yt, yp in zip(y_true, y_pred):
        cat_true = get_aqi_category(yt)
        cat_pred = get_aqi_category(yp)
        if cat_true and cat_pred and cat_true['label'] == cat_pred['label']:
            correct += 1
    cat_accuracy = correct / total * 100
    print(f"  Category Accuracy: {cat_accuracy:.2f}%  (target: > 85%)")

    # Per-city metrics
    inv_city_map = {v: k for k, v in city_map.items()}
    print(f"\n  Per-City Breakdown:")
    print(f"  {'City':<25} {'RMSE':>8} {'MAE':>8} {'SMAPE':>8} {'Count':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    unique_cities = np.unique(cities)
    city_metrics = {}
    for cid in unique_cities:
        mask = cities == cid
        if mask.sum() < 10:
            continue
        city_name = inv_city_map.get(cid, f"ID_{cid}")
        c_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        c_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        c_smape = compute_smape(y_true[mask], y_pred[mask])
        print(f"  {city_name:<25} {c_rmse:>8.2f} {c_mae:>8.2f} {c_smape:>7.2f}% {mask.sum():>8,}")
        city_metrics[city_name] = {'rmse': c_rmse, 'mae': c_mae, 'smape': c_smape}

    # Per-severity-band residual std (for heteroscedastic confidence intervals)
    print(f"\n  Residual Std by AQI Severity Band (for confidence intervals):")
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
        'rmse': rmse, 'mae': mae, 'smape': smape,
        'category_accuracy': cat_accuracy,
        'city_metrics': city_metrics,
        'residual_stds': residual_stds,
        'y_true': y_true, 'y_pred': y_pred
    }


def plot_training_curves(history, output_dir):
    """Save training loss and MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history.history['loss'], label='Train Loss', color='#00f2fe')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#FF7E00')
    axes[0].set_title('Training & Validation Loss (Huber)', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curve
    axes[1].plot(history.history['mae'], label='Train MAE', color='#00f2fe')
    axes[1].plot(history.history['val_mae'], label='Val MAE', color='#FF7E00')
    axes[1].set_title('Training & Validation MAE (Scaled)', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'lstm_training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[INFO] Training curves saved to {path}")


def plot_predictions(y_true, y_pred, output_dir, n_points=500):
    """Save a scatter plot of predicted vs actual AQI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: predicted vs actual
    sample_idx = np.random.choice(len(y_true), min(n_points, len(y_true)), replace=False)
    axes[0].scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, color='#00f2fe')
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect prediction')
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title('LSTM: Predicted vs Actual AQI', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Time series: first 200 points
    n_show = min(200, len(y_true))
    axes[1].plot(range(n_show), y_true[:n_show], label='Actual', color='#00f2fe', linewidth=1)
    axes[1].plot(range(n_show), y_pred[:n_show], label='Predicted', color='#FF7E00', linewidth=1, alpha=0.8)
    axes[1].set_xlabel('Time Step (hours)')
    axes[1].set_ylabel('AQI')
    axes[1].set_title('LSTM Forecast vs Ground Truth (First 200 Hours)', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'lstm_predictions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Prediction plots saved to {path}")


def main():
    print("=" * 60)
    print("  LSTM Model Training Pipeline")
    print("=" * 60)

    # --- Paths ---
    data_dir = 'data/lstm'
    models_dir = 'models'
    outputs_dir = 'outputs'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # --- Load preprocessed data ---
    print("\n[INFO] Loading preprocessed sequences...")
    train_data = np.load(os.path.join(data_dir, 'train_sequences.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_sequences.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_sequences.npz'))

    X_train, y_train, cities_train = train_data['X'], train_data['y'], train_data['cities']
    X_val, y_val, cities_val = val_data['X'], val_data['y'], val_data['cities']
    X_test, y_test, cities_test = test_data['X'], test_data['y'], test_data['cities']

    print(f"  Train:  X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:    X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:   X={X_test.shape}, y={y_test.shape}")

    # --- Load city map for dynamic n_cities ---
    with open(os.path.join(data_dir, 'city_map.json')) as f:
        city_map = json.load(f)
    n_cities = len(city_map)
    n_features = X_train.shape[2]
    lookback = X_train.shape[1]
    print(f"\n[INFO] n_cities={n_cities}, n_features={n_features}, lookback={lookback}")

    # --- Load scalers ---
    with open(os.path.join(data_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    target_scaler = scalers['target_scaler']

    # --- Build model ---
    print("\n[INFO] Building LSTM model...")
    model = build_lstm_model(
        n_features=n_features,
        lookback=lookback,
        n_cities=n_cities,
        embed_dim=8
    )
    model.summary()

    # --- Training ---
    print("\n[INFO] Starting training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5,
            factor=0.5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(models_dir, 'lstm_best.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    history = model.fit(
        [X_train, cities_train], y_train,
        validation_data=([X_val, cities_val], y_val),
        epochs=100,
        batch_size=1024,
        callbacks=callbacks,
        verbose=1
    )

    # --- Plot training curves ---
    plot_training_curves(history, outputs_dir)

    # --- Evaluate on VALIDATION set ---
    val_results = evaluate_model(
        model, X_val, cities_val, y_val,
        target_scaler, city_map, split_name="Validation"
    )

    # --- Evaluate on TEST set ---
    test_results = evaluate_model(
        model, X_test, cities_test, y_test,
        target_scaler, city_map, split_name="Test"
    )

    # --- Plot predictions ---
    plot_predictions(test_results['y_true'], test_results['y_pred'], outputs_dir)

    # --- Save Keras model ---
    keras_path = os.path.join(models_dir, 'lstm_aqi_final.keras')
    model.save(keras_path)
    print(f"\n[INFO] Keras model saved to {keras_path}")
    print(f"  Size: {os.path.getsize(keras_path) / 1024:.1f} KB")

    # --- Convert to TFLite for production inference ---
    print("\n[INFO] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Use float16 quantization for smaller model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = os.path.join(models_dir, 'lstm_aqi.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  TFLite model saved to {tflite_path}")
    print(f"  Size: {os.path.getsize(tflite_path) / 1024:.1f} KB")

    # --- Save residual statistics for confidence intervals ---
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

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
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
