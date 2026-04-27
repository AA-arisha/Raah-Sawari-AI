import requests

API_URL = "http://localhost:5000/predict-risk"

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# ════════════════════════════════════════════════════════════════
# Driver data — comes from database when driver requests a ride
# Add as many drivers as needed
# ════════════════════════════════════════════════════════════════
drivers = [
    {
        "driver_id"        : "D0001",
        "trips_completed"  : 1200,
        "driver_rating"    : 4.7,
        "cancellation_rate": 0.05,
        "experience_years" : 5,
        "night_trips_ratio": 0.2,
        "complaints_count" : 1
    },
    {
        "driver_id"        : "D0002",
        "trips_completed"  : 20,
        "driver_rating"    : 2.3,
        "cancellation_rate": 0.50,
        "experience_years" : 0,
        "night_trips_ratio": 0.75,
        "complaints_count" : 11
    },
    {
        "driver_id"        : "D0003",
        "trips_completed"  : 300,
        "driver_rating"    : 3.8,
        "cancellation_rate": 0.20,
        "experience_years" : 2,
        "night_trips_ratio": 0.45,
        "complaints_count" : 4
    }
]

# ════════════════════════════════════════════════════════════════
# Send each driver to API and print only the risk label
# ════════════════════════════════════════════════════════════════
for driver in drivers:
    try:
        response = requests.post(API_URL, json=driver, timeout=5)
        result   = response.json()

        risk_level = result.get("risk_level")
        print(f"{driver['driver_id']} : {RISK_LABELS.get(risk_level, 'Unknown')}")

    except requests.exceptions.ConnectionError:
        print(f"{driver['driver_id']} : Could not connect to API")
    except Exception as e:
        print(f"{driver['driver_id']} : {str(e)}")