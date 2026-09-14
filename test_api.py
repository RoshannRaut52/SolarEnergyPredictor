# test_api.py
import requests
import json

# Test health
print("Testing health...")
response = requests.get("http://localhost:5000/health")
print(response.json())

print("\n" + "="*50)

# Test models list
print("Getting models...")
response = requests.get("http://localhost:5000/models")
print(response.json())

print("\n" + "="*50)

# Test prediction
print("Making prediction...")
data = {
    "features": {
        "Hour": 12,
        "Day": 15,
        "Month": 6,
        "Year": 2016,
        "Cloud.coverage": 30,
        "Visibility": 10,
        "Temperature": 25,
        "Dew.point": 15,
        "Relative.humidity": 60,
        "Wind.speed": 8,
        "Station.pressure": 1013,
        "Altimeter": 29.92
    },
    "model": "gradient_boosting"
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())