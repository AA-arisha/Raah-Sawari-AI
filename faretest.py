import requests

response = requests.post(
    "http://127.0.0.1:5001/predict-fare",
    json={
        "pickup":       "DHA Phase 2 Karachi",
        "destination":  "Gulshan-e-Iqbal Karachi",
        "vehicle_type": "car",
        "duration_min": 25.0
    }
)

print(response.json())