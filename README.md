# ☀️ Solar Energy Predictor

A full-stack machine learning application that predicts solar energy output based on weather parameters. Built with Flask, TensorFlow, and scikit-learn.

![Dashboard](screenshots/dashboard.png)

## 🎯 Overview

This project implements **3 machine learning models** to predict short-term solar energy output using 12 weather parameters. It features a complete web interface with dashboard, predictions, model management, and history tracking.

**Based on**: CS229 final project — *"What weather features are the most influential for solar energy prediction at University of Illinois?"*

## ✨ Features

- 🤖 **3 ML Models**: Gradient Boosting, Random Forest, LSTM Neural Network
- 📊 **Interactive Dashboard**: Real-time stats and recent predictions
- 🔮 **Real-time Predictions**: Instant results from weather inputs
- 📈 **History Tracking**: All predictions saved with timestamps
- ⚙️ **Settings Page**: Preferences and CSV export
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🎨 **Modern UI**: Gradient themes, smooth animations

## 🧠 Models

| Model | Type | R² Score | Description |
|-------|------|----------|-------------|
| **LSTM** | Deep Learning | 0.92 | 2-layer LSTM with 24-hour sequence |
| **Gradient Boosting** | Ensemble | 0.82 | 35 trees, complexity 15 |
| **Random Forest** | Ensemble | 0.79 | 100 trees, max depth 15 |

### Feature Importance
1. Temperature (28%)
2. Cloud Coverage (25%)
3. Humidity (18%)
4. Dew Point (12%)
5. Wind Speed (7%)
6. Visibility, Pressure, Altimeter (10% combined)

## 📁 Project Structure
