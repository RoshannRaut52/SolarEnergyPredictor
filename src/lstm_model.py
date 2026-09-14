# src/lstm_model.py
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras import optimizers
import joblib
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def normalize_data(dataset, data_min, data_max):
    """Normalize data between 0 and 1"""
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

def import_data(train_dataframe, dev_dataframe, test_dataframe):
    """Import and pre-process data"""
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')
    
    max_test = np.max(dataset[:,12])
    min_test = np.min(dataset[:,12])
    scale_factor = max_test - min_test
    max_val = np.empty(13)
    min_val = np.empty(13)
    
    # Create training dataset
    for i in range(0,13):
        min_val[i] = np.amin(dataset[:,i], axis=0)
        max_val[i] = np.amax(dataset[:,i], axis=0)
        dataset[:,i] = normalize_data(dataset[:, i], min_val[i], max_val[i])
    
    train_data = dataset[:,0:12]
    train_labels = dataset[:,12]
    
    # Create dev dataset
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')
    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min_val[i], max_val[i])
    
    dev_data = dataset[:,0:12]
    dev_labels = dataset[:,12]
    
    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')
    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min_val[i], max_val[i])
    
    test_data = dataset[:, 0:12]
    test_labels = dataset[:, 12]
    
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor, min_val, max_val

def build_model(init_type='glorot_uniform', optimizer='adam', num_features=12):
    """Build LSTM model - Fixed for newer Keras versions"""
    model = Sequential()
    
    # First LSTM layer with Input layer
    model.add(keras.layers.Input(shape=(None, num_features)))
    model.add(LSTM(num_features, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(64, kernel_initializer=init_type, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(64, activation='tanh', kernel_initializer=init_type))
    model.add(Dense(1))
    model.add(Activation("relu"))
    
    # Use learning_rate instead of lr
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=0.001)
    else:
        opt = optimizers.RMSprop(learning_rate=0.002, rho=0.9, epsilon=1e-08)
    
    model.compile(loss="mean_squared_error", optimizer=opt)
    
    return model

def train_lstm_model():
    """Train LSTM model using your data"""
    print("📂 Loading data...")
    
    # Load your data files
    train_dataframe = pd.read_csv('data/processed/weather_train.csv', sep=";", header=None)
    dev_dataframe = pd.read_csv('data/processed/weather_dev.csv', sep=";", header=None)
    test_dataframe = pd.read_csv('data/processed/weather_test.csv', sep=";", header=None)
    
    train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor, min_val, max_val = import_data(
        train_dataframe, dev_dataframe, test_dataframe)
    
    time_steps = 1
    X_train = np.reshape(train_data, (train_data.shape[0] // time_steps, time_steps, train_data.shape[1]))
    X_dev = np.reshape(dev_data, (dev_data.shape[0] // time_steps, time_steps, dev_data.shape[1]))
    X_test = np.reshape(test_data, (test_data.shape[0] // time_steps, time_steps, test_data.shape[1]))
    Y_train = np.reshape(train_labels, (train_labels.shape[0] // time_steps, time_steps, 1))
    Y_dev = np.reshape(dev_labels, (dev_labels.shape[0] // time_steps, time_steps, 1))
    Y_test = np.reshape(test_labels, (test_labels.shape[0] // time_steps, time_steps, 1))
    
    print(f"✅ Data loaded: Train={X_train.shape[0]}, Dev={X_dev.shape[0]}, Test={X_test.shape[0]}")
    
    print("🧠 Building LSTM model...")
    model = build_model('glorot_uniform', 'adam')
    model.summary()
    
    print("⏳ Training LSTM model...")
    history = model.fit(
        X_train, Y_train,
        batch_size=16,
        epochs=30,  # Reduced for faster testing
        validation_data=(X_dev, Y_dev),
        verbose=1
    )
    
    # Save model and preprocessing parameters
    os.makedirs('models/saved_models', exist_ok=True)
    model.save('models/saved_models/lstm_model.h5')
    
    # Save preprocessing parameters
    joblib.dump({
        'min_val': min_val,
        'max_val': max_val,
        'scale_factor': scale_factor,
        'feature_count': 12
    }, 'models/saved_models/lstm_preprocess.pkl')
    
    print("✅ LSTM model saved successfully!")
    
    # Evaluate
    test_pred = model.predict(X_test)
    mse = np.mean((test_pred - Y_test) ** 2) * scale_factor * scale_factor
    print(f"📊 Test MSE: {mse:.6f}")
    
    return model

if __name__ == "__main__":
    train_lstm_model()