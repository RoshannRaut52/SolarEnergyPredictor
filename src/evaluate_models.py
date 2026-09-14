# src/evaluate_models.py
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_models')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'models', 'model_metrics.json')

# Feature columns
FEATURES = [
    'Hour', 'Day', 'Month', 'Year', 'Cloud.coverage', 'Visibility',
    'Temperature', 'Dew.point', 'Relative.humidity', 'Wind.speed',
    'Station.pressure', 'Altimeter'
]


def load_test_data():
    """Load test data"""
    print("📂 Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'weather_test.csv'), sep=";", header=None)
    
    col_names = FEATURES + ['Solar.energy']
    test_df.columns = col_names
    
    X = test_df[FEATURES].values
    y = test_df['Solar.energy'].values
    
    print(f"✅ Test samples: {len(X)}")
    return X, y


def evaluate_simple_model(model, scaler, X, y, name):
    """Evaluate a scikit-learn model"""
    print(f"\n🔍 Evaluating {name}...")
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # R² as percentage (some call this "accuracy" for regression)
    accuracy_pct = max(0, r2 * 100)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f} ({accuracy_pct:.1f}%)")
    
    return {
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'accuracy': round(accuracy_pct, 1),
        'test_samples': len(y)
    }


def evaluate_lstm(model, scaler, X, y, name):
    """Evaluate LSTM model with sequences"""
    print(f"\n🔍 Evaluating {name}...")
    
    SEQ_LENGTH = 24
    X_scaled = scaler.transform(X)
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(SEQ_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQ_LENGTH:i])
        y_seq.append(y[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Predict
    y_pred = model.predict(X_seq, verbose=0).flatten()
    
    # Reverse scale if needed
    config_file = os.path.join(MODEL_PATH, 'lstm_config.pkl')
    if os.path.exists(config_file):
        config = joblib.load(config_file)
        y_min = config.get('y_min', 0)
        y_max = config.get('y_max', 1)
        y_pred = y_pred * (y_max - y_min) + y_min
    
    mse = mean_squared_error(y_seq, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_seq, y_pred)
    r2 = r2_score(y_seq, y_pred)
    accuracy_pct = max(0, r2 * 100)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f} ({accuracy_pct:.1f}%)")
    
    return {
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'accuracy': round(accuracy_pct, 1),
        'test_samples': len(y_seq)
    }


def main():
    print("=" * 60)
    print("📊 EVALUATING ALL MODELS ON TEST SET")
    print("=" * 60)
    
    # Load test data
    X, y = load_test_data()
    
    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    
    # Results dict
    results = {}
    
    # Random Forest
    try:
        rf = joblib.load(os.path.join(MODEL_PATH, 'random_forest.pkl'))
        results['random_forest'] = evaluate_simple_model(rf, scaler, X, y, 'Random Forest')
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")
        results['random_forest'] = {'error': str(e)}
    
    # Gradient Boosting
    try:
        gb = joblib.load(os.path.join(MODEL_PATH, 'gradient_boosting.pkl'))
        results['gradient_boosting'] = evaluate_simple_model(gb, scaler, X, y, 'Gradient Boosting')
    except Exception as e:
        print(f"❌ Gradient Boosting failed: {e}")
        results['gradient_boosting'] = {'error': str(e)}
    
    # LSTM
    try:
        lstm = keras.models.load_model(
            os.path.join(MODEL_PATH, 'lstm_model.keras'),
            compile=False
        )
        results['lstm'] = evaluate_lstm(lstm, scaler, X, y, 'LSTM Neural Network')
    except Exception as e:
        print(f"❌ LSTM failed: {e}")
        results['lstm'] = {'error': str(e)}
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Metrics saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()