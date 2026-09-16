# api/app.py
# Limit TensorFlow threads BEFORE importing TF (critical for Render free tier)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
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
_lstm_load_attempted = False  # Track if LSTM load was tried


# ============================================================
# LAZY LOADER FOR LSTM
# ============================================================
def get_lstm_model():
    """Lazy-load LSTM model only when needed (saves memory at startup)"""
    global _lstm_load_attempted

    # Already loaded — return it
    if 'lstm' in models:
        return models['lstm']

    # Already tried and failed — don't retry
    if _lstm_load_attempted:
        return None

    _lstm_load_attempted = True

    try:
        lstm_keras = os.path.join(MODEL_PATH, 'lstm_model.keras')
        lstm_h5 = os.path.join(MODEL_PATH, 'lstm_model.h5')

        if os.path.exists(lstm_keras):
            print("🔄 Loading LSTM on demand...")
            models['lstm'] = keras.models.load_model(lstm_keras, compile=False)
            print("✅ LSTM loaded (lazy)")
            return models['lstm']
        elif os.path.exists(lstm_h5):
            print("🔄 Loading LSTM on demand (.h5)...")
            models['lstm'] = keras.models.load_model(lstm_h5, compile=False)
            print("✅ LSTM loaded (lazy)")
            return models['lstm']
        else:
            print(f"❌ LSTM file not found")
            return None

    except Exception as e:
        print(f"❌ LSTM lazy-load failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# MODEL LOADING (RF + GB only at startup)
# ============================================================
def load_models():
    """Load RF and GB at startup. LSTM is lazy-loaded."""
    global models, scaler, feature_names

    print("📂 Loading models...")
    print(f"📁 Model path: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model directory not found: {MODEL_PATH}")
        return

    print("📁 Files in model directory:")
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

    # Random Forest
    try:
        rf_file = os.path.join(MODEL_PATH, 'random_forest.pkl')
        if os.path.exists(rf_file):
            models['random_forest'] = joblib.load(rf_file)
            print("✅ Random Forest loaded")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    # Gradient Boosting
    try:
        gb_file = os.path.join(MODEL_PATH, 'gradient_boosting.pkl')
        if os.path.exists(gb_file):
            models['gradient_boosting'] = joblib.load(gb_file)
            print("✅ Gradient Boosting loaded")
    except Exception as e:
        print(f"❌ Gradient Boosting failed: {e}")

    # LSTM - DON'T load here. It will be lazy-loaded on first request.
    print("⏳ LSTM will be lazy-loaded on first request (saves memory)")

    print("=" * 50)
    print(f"✅ Models loaded at startup: {list(models.keys())}")
    print("=" * 50)


# ============================================================
# HISTORY
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

    if scaler is not None:
        features = scaler.transform(features)

    # LSTM needs 24-hour sequence
    if model_type == 'lstm':
        SEQ_LENGTH = 24
        features = np.tile(features, (SEQ_LENGTH, 1))
        features = features.reshape(1, SEQ_LENGTH, -1)

    return features


# ============================================================
# WEB PAGE ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction page - include LSTM in dropdown"""
    # Include all models that could be loaded
    all_available = list(models.keys())
    # Always show LSTM as an option (it may lazy-load on first use)
    if 'lstm' not in all_available:
        all_available.append('lstm')
    
    # Order: reliable first
    preferred = ['random_forest', 'gradient_boosting', 'lstm']
    ordered = [m for m in preferred if m in all_available]
    
    return render_template('predict.html', models=ordered)


@app.route('/models')
def models_page():
    return render_template('models.html')


@app.route('/history')
def history_page():
    return render_template('history.html')


@app.route('/settings')
def settings_page():
    return render_template('settings.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


# ============================================================
# API ROUTES
# ============================================================
@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'models': list(models.keys()),
        'features': feature_names
    })


@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({
        'models': list(models.keys()),
        'features': feature_names,
        'feature_count': len(feature_names) if feature_names else 0
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    """Make a prediction"""
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        features = data.get('features', {})
        model_name = data.get('model', 'random_forest')

        print(f"\n🔍 Prediction request:")
        print(f"   Model: {model_name}")
        print(f"   Available: {list(models.keys())}")

        # ============================================
        # LSTM
        # ============================================
        if model_name == 'lstm':
            lstm_model = get_lstm_model()
            if lstm_model is None:
                return jsonify({
                    'success': False,
                    'error': 'LSTM model could not be loaded. Try Random Forest or Gradient Boosting.',
                    'available_models': [m for m in models.keys() if m != 'lstm']
                }), 503

            try:
                X = prepare_features(features, 'lstm')
                print(f"   LSTM input shape: {X.shape}")

                prediction_scaled = lstm_model.predict(X, verbose=0, batch_size=1)

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

            except Exception as lstm_err:
                print(f"❌ LSTM predict failed: {lstm_err}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': 'LSTM inference failed. Try Random Forest or Gradient Boosting.',
                    'available_models': [m for m in models.keys() if m != 'lstm']
                }), 500

        # ============================================
        # RF / GB
        # ============================================
        elif model_name in models:
            X = prepare_features(features, 'simple')
            prediction = float(models[model_name].predict(X)[0])
            print(f"   📈 Prediction: {prediction:.2f} kWh")

        else:
            return jsonify({
                'success': False,
                'error': f'Model "{model_name}" not found',
                'available_models': list(models.keys())
            }), 400

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
    return jsonify({
        'total': len(prediction_history),
        'history': list(reversed(prediction_history[-100:]))
    })


@app.route('/api/history', methods=['DELETE'])
def api_clear_history():
    global prediction_history
    prediction_history = []
    save_history()
    return jsonify({'success': True, 'message': 'History cleared'})


@app.route('/api/history/export', methods=['GET'])
def api_export_history():
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
# STARTUP
# ============================================================
load_models()
load_history()

print()
print("=" * 55)
print("🚀 Solar Energy Predictor - Server Ready")
print("=" * 55)
print(f"📊 Models loaded: {list(models.keys())}")
print(f"📈 History entries: {len(prediction_history)}")
print("=" * 55)
print()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)