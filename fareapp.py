from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import requests

app = Flask(__name__)
CORS(app)

# ── Load Fare Model ───────────────────────────────────────────────────────────
fare_model = joblib.load("fare_model.pkl")
le_vehicle = joblib.load("fare_le_vehicle.pkl")
le_traffic = joblib.load("fare_le_traffic.pkl")
le_time    = joblib.load("fare_le_time.pkl")
le_demand  = joblib.load("fare_le_demand.pkl")
le_supply  = joblib.load("fare_le_supply.pkl")

# ── Reuse ETA Traffic Model ───────────────────────────────────────────────────
traffic_model  = joblib.load("traffic_model.pkl")
eta_le_vehicle = joblib.load("le_vehicle.pkl")
eta_le_traffic = joblib.load("le_traffic.pkl")

# ── Helpers ───────────────────────────────────────────────────────────────────
def geocode(address):
    url     = "https://nominatim.openstreetmap.org/search"
    params  = {"q": address + ", Karachi, Pakistan", "format": "json", "limit": 1}
    headers = {"User-Agent": "raahsawari-app"}
    try:
        res  = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    return None, None

def haversine_km(lat1, lng1, lat2, lng2):
    R    = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
    return round(R * 2 * np.arcsin(np.sqrt(a)) * 1.4, 2)

def get_time_of_day(hour):
    if 6  <= hour <= 11: return "morning"
    if 12 <= hour <= 16: return "afternoon"
    if 17 <= hour <= 21: return "evening"
    return "night"

def get_demand_supply(hour):
    if hour in [8, 9, 18, 19, 20]:    # morning + evening rush
        return "high", "low"           # surge pricing
    elif hour in [7, 10, 17, 21]:      # semi peak
        return "medium", "medium"
    elif hour in [12, 13, 14]:         # afternoon
        return "low", "high"           # cheap
    elif hour in [0, 1, 2, 3, 4]:      # late night
        return "low", "low"
    else:                               # normal hours
        return "low", "high"

# ── Fare Endpoint ─────────────────────────────────────────────────────────────
@app.route("/predict-fare", methods=["POST"])
def predict_fare():
    try:
        data         = request.get_json()
        pickup_text  = data["pickup"]
        dest_text    = data["destination"]
        vehicle_type = data["vehicle_type"]   # bike / car / rickshaw
        duration_min = data["duration_min"]   # from ETA API

        # Step 1: Geocode
        origin_lat, origin_lng           = geocode(pickup_text)
        destination_lat, destination_lng = geocode(dest_text)

        if not all([origin_lat, origin_lng, destination_lat, destination_lng]):
            return jsonify({"status": "error", "message": "Location not found"}), 400

        # Step 2: Auto calculate
        distance_km  = haversine_km(origin_lat, origin_lng, destination_lat, destination_lng)
        hour         = datetime.now().hour
        time_of_day  = get_time_of_day(hour)
        petrol_price = 380.0

        # Simulate demand and supply from hour
        demand_level, supply_level = get_demand_supply(hour)

        # Step 3: Traffic from ETA traffic model
        v_enc         = eta_le_vehicle.transform([vehicle_type])[0]
        traffic_enc   = traffic_model.predict(pd.DataFrame([[
            distance_km, hour, v_enc
        ]], columns=["distance_km", "hour_of_day", "vehicle_type_enc"]))[0]
        traffic_level = eta_le_traffic.inverse_transform([traffic_enc])[0]

        # Step 4: Fuel cost
        fuel_km   = {"bike": 35, "car": 12, "rickshaw": 25}
        fuel_cost = round((distance_km / fuel_km[vehicle_type]) * petrol_price, 2)

        # Step 5: Encode for fare model
        vehicle_enc  = le_vehicle.transform([vehicle_type])[0]
        traffic_enc2 = le_traffic.transform([traffic_level])[0]
        time_enc     = le_time.transform([time_of_day])[0]
        demand_enc   = le_demand.transform([demand_level])[0]
        supply_enc   = le_supply.transform([supply_level])[0]

        # Step 6: Predict fare
        features = pd.DataFrame([[
            vehicle_enc,
            distance_km,
            duration_min,
            time_enc,
            traffic_enc2,
            demand_enc,
            supply_enc,
            petrol_price,
            fuel_cost
        ]], columns=[
            "vehicle_type_enc", "distance_km", "duration_minutes",
            "time_of_day_enc",  "traffic_level_enc",
            "demand_level_enc", "supply_level_enc",
            "petrol_price_per_liter", "fuel_cost"
        ])

        recommended_fare = round(float(fare_model.predict(features)[0]), 2)
        min_fare         = round(recommended_fare * 0.80, 2)
        max_fare         = round(recommended_fare * 1.25, 2)

        return jsonify({
            "min_fare":         min_fare,
            "recommended_fare": recommended_fare,
            "max_fare":         max_fare
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ── Home Route ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "message": "Raah Sawari Fare API is live!",
        "usage":   "POST /predict-fare with pickup, destination, vehicle_type, duration_min"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)