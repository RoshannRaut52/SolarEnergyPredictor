# src/train_lstm_fixed.py
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def train_lstm_fixed():
    print("=" * 60)
    print("🧠 TRAINING FIXED LSTM MODEL")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv('data/processed/weather_train.csv', sep=";", header=None)
    dev_df = pd.read_csv('data/processed/weather_dev.csv', sep=";", header=None)
    test_df = pd.read_csv('data/processed/weather_test.csv', sep=";", header=None)
    
    col_names = ['Hour', 'Day', 'Month', 'Year', 'Cloud.coverage', 'Visibility',
                 'Temperature', 'Dew.point', 'Relative.humidity', 'Wind.speed',
                 'Station.pressure', 'Altimeter', 'Solar.energy']
    
    train_df.columns = col_names
    dev_df.columns = col_names
    test_df.columns = col_names
    
    print(f"✅ Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    feature_cols = col_names[:-1]
    target_col = 'Solar.energy'
    
    # Scale features
    print("\n🔧 Scaling features...")
    scaler = joblib.load('models/saved_models/scaler.pkl')
    X_train = scaler.transform(train_df[feature_cols].values)
    X_dev = scaler.transform(dev_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    # Scale target (IMPORTANT!)
    print("🔧 Scaling target...")
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(train_df[[target_col]].values).flatten()
    y_dev = y_scaler.transform(dev_df[[target_col]].values).flatten()
    y_test = y_scaler.transform(test_df[[target_col]].values).flatten()
    
    print(f"✅ Target scaled: {y_train.min():.3f} to {y_train.max():.3f}")
    
    # Create sequences
    print("\n🔨 Creating sequences...")
    SEQ_LENGTH = 24
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_dev_seq, y_dev_seq = create_sequences(X_dev, y_dev, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)
    
    print(f"✅ Sequences: Train {X_train_seq.shape}, Dev {X_dev_seq.shape}, Test {X_test_seq.shape}")
    
    # Build model
    print("\n🧠 Building LSTM model...")
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LENGTH, len(feature_cols))),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    model.summary()
    
    # Train
    print("\n⏳ Training LSTM...")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_data=(X_dev_seq, y_dev_seq),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating...")
    train_loss = model.evaluate(X_train_seq, y_train_seq, verbose=0)
    dev_loss = model.evaluate(X_dev_seq, y_dev_seq, verbose=0)
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    
    print(f"\n📈 Final Results:")
    print(f"   Train MSE: {float(train_loss):.6f}")
    print(f"   Dev MSE:   {float(dev_loss):.6f}")
    print(f"   Test MSE:  {float(test_loss):.6f}")
    
    # Sample predictions
    print(f"\n🔍 Sample Predictions (Scaled):")
    test_pred = model.predict(X_test_seq[:5], verbose=0)
    for i in range(5):
        print(f"   Actual: {y_test_seq[i]:.4f} | Predicted: {test_pred[i][0]:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    model.save('models/saved_models/lstm_model.keras')
    
    # Save config with target scaling info
    joblib.dump({
        'seq_length': SEQ_LENGTH,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'y_min': float(y_scaler.data_min_[0]),
        'y_max': float(y_scaler.data_max_[0])
    }, 'models/saved_models/lstm_config.pkl')
    
    print("✅ LSTM model saved successfully!")
    print(f"📁 Saved: models/saved_models/lstm_model.keras")
    print(f"📁 Config: y_min={y_scaler.data_min_[0]:.2f}, y_max={y_scaler.data_max_[0]:.2f}")
    print("=" * 60)

if __name__ == '__main__':
    train_lstm_fixed()