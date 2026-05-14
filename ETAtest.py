import requests

response = requests.post(
    "http://localhost:3000/predict-eta",
    json={
        "pickup":      "DHA Phase 2 Karachi",
        "destination": "Gulshan-e-Iqbal Karachi"
    }
)

print(response.json())