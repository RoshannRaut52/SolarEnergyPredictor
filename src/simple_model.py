# src/simple_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_simple_models():
    """Train simpler models for comparison"""
    print("📂 Loading data...")
    
    # Load your data files
    train_df = pd.read_csv('data/processed/weather_train.csv', sep=";", header=None)
    dev_df = pd.read_csv('data/processed/weather_dev.csv', sep=";", header=None)
    test_df = pd.read_csv('data/processed/weather_test.csv', sep=";", header=None)
    
    # Define column names (based on your R code)
    col_names = ['Hour', 'Day', 'Month', 'Year', 'Cloud.coverage', 'Visibility', 
                 'Temperature', 'Dew.point', 'Relative.humidity', 'Wind.speed', 
                 'Station.pressure', 'Altimeter', 'Solar.energy']
    
    train_df.columns = col_names
    dev_df.columns = col_names
    test_df.columns = col_names
    
    # Prepare features and target
    feature_cols = col_names[:-1]  # All except Solar.energy
    X_train = train_df[feature_cols]
    y_train = train_df['Solar.energy']
    X_test = test_df[feature_cols]
    y_test = test_df['Solar.energy']
    
    print(f"✅ Data loaded: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models directory
    os.makedirs('models/saved_models', exist_ok=True)
    
    # 1. Random Forest
    print("🌳 Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/saved_models/random_forest.pkl')
    
    # 2. Gradient Boosting
    print("📈 Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    joblib.dump(gb, 'models/saved_models/gradient_boosting.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_cols, 'models/saved_models/feature_names.pkl')
    
    # Evaluate
    for name, model in [('Random Forest', rf), ('Gradient Boosting', gb)]:
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"📊 {name} - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    print("✅ Models saved successfully!")

if __name__ == "__main__":
    train_simple_models()