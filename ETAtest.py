import requests

response = requests.post(
    "http://127.0.0.1:5000/predict-eta",
    json={
        "pickup":      "DHA Phase 2 Karachi",
        "destination": "Gulshan-e-Iqbal Karachi"
    }
)

print(response.json())