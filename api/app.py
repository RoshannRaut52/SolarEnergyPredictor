# api/app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_models')
HISTORY_FILE = os.path.join(PROJECT_ROOT, 'models', 'prediction_history.json')

# ============================================================
# FLASK APP
# ============================================================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
CORS(app, origins=['*'])

# ============================================================
# GLOBAL STATE
# ============================================================
models = {}
scaler = None
feature_names = None
prediction_history = []

# ============================================================
# MODEL LOADING
# ============================================================
def load_models():
    """Load all trained models from disk"""
    global models, scaler, feature_names

    print("📂 Loading models...")
    print(f"📁 Model path: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model directory not found: {MODEL_PATH}")
        return

    print(f"📁 Files in model directory:")
    for f in os.listdir(MODEL_PATH):
        print(f"   - {f}")

    # Scaler
    try:
        scaler_file = os.path.join(MODEL_PATH, 'scaler.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            print("✅ Scaler loaded")
    except Exception as e:
        print(f"❌ Scaler failed: {e}")

    # Feature names
    try:
        features_file = os.path.join(MODEL_PATH, 'feature_names.pkl')
        if os.path.exists(features_file):
            feature_names = joblib.load(features_file)
            print(f"✅ Feature names loaded: {len(feature_names)} features")
        else:
            feature_names = [
                'Hour', 'Day', 'Month', 'Year', 'Cloud.coverage', 'Visibility',
                'Temperature', 'Dew.point', 'Relative.humidity', 'Wind.speed',
                'Station.pressure', 'Altimeter'
            ]
            print("⚠️ Using default feature names")
    except Exception as e:
        print(f"❌ Feature names failed: {e}")

    # Random Forest (independent try)
    try:
        rf_file = os.path.join(MODEL_PATH, 'random_forest.pkl')
        if os.path.exists(rf_file):
            models['random_forest'] = joblib.load(rf_file)
            print("✅ Random Forest loaded")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    # Gradient Boosting (independent try)
    try:
        gb_file = os.path.join(MODEL_PATH, 'gradient_boosting.pkl')
        if os.path.exists(gb_file):
            models['gradient_boosting'] = joblib.load(gb_file)
            print("✅ Gradient Boosting loaded")
    except Exception as e:
        print(f"❌ Gradient Boosting failed: {e}")

    # LSTM (independent try)
    try:
        lstm_keras = os.path.join(MODEL_PATH, 'lstm_model.keras')
        lstm_h5 = os.path.join(MODEL_PATH, 'lstm_model.h5')

        if os.path.exists(lstm_keras):
            models['lstm'] = keras.models.load_model(lstm_keras, compile=False)
            print("✅ LSTM loaded (.keras)")
        elif os.path.exists(lstm_h5):
            models['lstm'] = keras.models.load_model(lstm_h5, compile=False)
            print("✅ LSTM loaded (.h5)")
    except Exception as e:
        print(f"❌ LSTM failed: {e}")

    print("=" * 50)
    print(f"✅ Models loaded: {list(models.keys())}")
    print(f"✅ Total models: {len(models)}")
    print("=" * 50)


# ============================================================
# HISTORY MANAGEMENT
# ============================================================
def load_history():
    """Load prediction history from JSON file"""
    global prediction_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                prediction_history = json.load(f)
            print(f"✅ Loaded {len(prediction_history)} history entries")
        except Exception as e:
            print(f"⚠️ Could not load history: {e}")
            prediction_history = []
    else:
        prediction_history = []
        print("📝 Starting with empty history")


def save_history():
    """Save prediction history to JSON file"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        # Keep only the last 500 entries
        with open(HISTORY_FILE, 'w') as f:
            json.dump(prediction_history[-500:], f, indent=2)
    except Exception as e:
        print(f"❌ Error saving history: {e}")


# ============================================================
# FEATURE PREPARATION
# ============================================================
def prepare_features(data, model_type='simple'):
    """Prepare feature vector for prediction"""
    if isinstance(data, dict):
        features = []
        for name in feature_names:
            features.append(float(data.get(name, 0)))
        features = np.array(features).reshape(1, -1)
    else:
        features = np.array(data).reshape(1, -1)

    # Apply scaler
    if scaler is not None:
        features = scaler.transform(features)

    # LSTM needs 24-hour sequence
    if model_type == 'lstm':
        SEQ_LENGTH = 24
        features = np.tile(features, (SEQ_LENGTH, 1))          # (24, 12)
        features = features.reshape(1, SEQ_LENGTH, -1)         # (1, 24, 12)

    return features


# ============================================================
# WEB PAGE ROUTES
# ============================================================
@app.route('/')
def home():
    """Dashboard page"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction page"""
    # Preferred order: Random Forest first (most reliable), then others
    preferred = ['random_forest', 'gradient_boosting', 'lstm']
    ordered = [m for m in preferred if m in models]
    return render_template('predict.html', models=ordered)


METRICS_FILE = os.path.join(PROJECT_ROOT, 'models', 'model_metrics.json')

def load_metrics():
    """Load real model metrics from JSON"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}


# Update /models route
@app.route('/models')
def models_page():
    """Models page with real metrics"""
    metrics = load_metrics()
    return render_template('models.html', metrics=metrics)


@app.route('/history')
def history_page():
    """History page"""
    return render_template('history.html')


@app.route('/settings')
def settings_page():
    """Settings page"""
    return render_template('settings.html')


@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')


# ============================================================
# API ROUTES
# ============================================================
@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'models': list(models.keys()),
        'features': feature_names
    })


@app.route('/api/models', methods=['GET'])
def api_models():
    """List available models and features"""
    return jsonify({
        'models': list(models.keys()),
        'features': feature_names,
        'feature_count': len(feature_names) if feature_names else 0
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        features = data.get('features', {})
        model_name = data.get('model', 'random_forest')  # Default to RF!

        print(f"\n🔍 Prediction request:")
        print(f"   Model: {model_name}")
        print(f"   Available: {list(models.keys())}")

        if model_name not in models:
            return jsonify({
                'success': False,
                'error': f'Model "{model_name}" not found',
                'available_models': list(models.keys())
            }), 400

        # LSTM branch with extra safety
        if model_name == 'lstm':
            try:
                X = prepare_features(features, 'lstm')
                print(f"   LSTM input shape: {X.shape}")

                prediction_scaled = models['lstm'].predict(X, verbose=0, batch_size=1)

                if len(prediction_scaled.shape) == 3:
                    prediction_scaled = float(prediction_scaled[0][0][0])
                else:
                    prediction_scaled = float(prediction_scaled[0][0])

                config_file = os.path.join(MODEL_PATH, 'lstm_config.pkl')
                if os.path.exists(config_file):
                    config = joblib.load(config_file)
                    y_min = config.get('y_min', 0)
                    y_max = config.get('y_max', 1)
                    prediction = prediction_scaled * (y_max - y_min) + y_min
                else:
                    prediction = prediction_scaled

                print(f"   📈 LSTM: {prediction:.2f} kWh")

            except Exception as lstm_error:
                print(f"❌ LSTM failed: {lstm_error}")
                return jsonify({
                    'success': False,
                    'error': 'LSTM model failed (may be out of memory on free tier). Try Random Forest.',
                    'suggestion': 'Use random_forest model instead'
                }), 500

        else:
            X = prepare_features(features, 'simple')
            prediction = float(models[model_name].predict(X)[0])
            print(f"   📈 Prediction: {prediction:.2f} kWh")

        # Save history
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'features': features,
            'prediction': round(float(prediction), 2)
        }
        prediction_history.append(entry)
        save_history()

        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'model_used': model_name,
            'features_used': feature_names,
            'timestamp': entry['timestamp']
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """Get prediction history"""
    return jsonify({
        'total': len(prediction_history),
        'history': list(reversed(prediction_history[-100:]))
    })


@app.route('/api/history', methods=['DELETE'])
def api_clear_history():
    """Clear prediction history"""
    global prediction_history
    prediction_history = []
    save_history()
    return jsonify({'success': True, 'message': 'History cleared'})


@app.route('/api/history/export', methods=['GET'])
def api_export_history():
    """Export history as JSON (frontend can convert to CSV)"""
    return jsonify({
        'total': len(prediction_history),
        'history': prediction_history
    })


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================
# STARTUP - LOAD MODELS AT IMPORT TIME
# ============================================================
# This MUST run at import so gunicorn picks up models in production.
# gunicorn imports this file, so this code executes even when
# __name__ != '__main__'.
load_models()
load_history()

print()
print("=" * 55)
print("🚀 Solar Energy Predictor - Server Ready")
print("=" * 55)
print(f"📊 Models loaded: {len(models)}")
print(f"📈 History entries: {len(prediction_history)}")
print("=" * 55)
print()


# ============================================================
# DEV SERVER
# ============================================================
if __name__ == '__main__':
    # Only run the dev server when this file is run directly
    # (Production uses gunicorn which imports `app` from this module)
    app.run(host='0.0.0.0', port=5000, debug=True)