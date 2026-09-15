# src/retrain_gb.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("🔄 Retraining GB and RF with sklearn 1.4.2")
print("=" * 60)

# Paths
DATA_PATH = 'data/processed'
MODEL_PATH = 'models/saved_models'

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv(os.path.join(DATA_PATH, 'weather_train.csv'), sep=";", header=None)
test_df = pd.read_csv(os.path.join(DATA_PATH, 'weather_test.csv'), sep=";", header=None)

col_names = ['Hour', 'Day', 'Month', 'Year', 'Cloud.coverage', 'Visibility',
             'Temperature', 'Dew.point', 'Relative.humidity', 'Wind.speed',
             'Station.pressure', 'Altimeter', 'Solar.energy']

train_df.columns = col_names
test_df.columns = col_names

features = col_names[:-1]
target = 'Solar.energy'

X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# Load existing scaler
print("\n🔧 Loading existing scaler...")
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Retrain Gradient Boosting
# ==========================================
print("\n🌳 Training Gradient Boosting...")
gb = GradientBoostingRegressor(
    n_estimators=35,
    learning_rate=0.5,
    max_depth=15,
    random_state=42
)
gb.fit(X_train_scaled, y_train)

y_pred = gb.predict(X_test_scaled)
print(f"   MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"   R²:   {r2_score(y_test, y_pred):.4f}")

# Save
joblib.dump(gb, os.path.join(MODEL_PATH, 'gradient_boosting.pkl'))
print("✅ Gradient Boosting saved")

# ==========================================
# Also retrain Random Forest (for consistency)
# ==========================================
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
print(f"   MSE:  {mean_squared_error(y_test, y_pred_rf):.4f}")
print(f"   R²:   {r2_score(y_test, y_pred_rf):.4f}")

# Save
joblib.dump(rf, os.path.join(MODEL_PATH, 'random_forest.pkl'))
print("✅ Random Forest saved")

print("\n" + "=" * 60)
print("✅ Both models retrained with sklearn 1.4.2")
print("=" * 60)