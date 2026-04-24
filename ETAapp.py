from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import requests

app = Flask(__name__)



# ── Load Models ───────────────────────────────────────────────────────────────
eta_model      = joblib.load("eta_model.pkl")
traffic_model  = joblib.load("traffic_model.pkl")
le_vehicle     = joblib.load("le_vehicle.pkl")
le_traffic     = joblib.load("le_traffic.pkl")

# ── Geocoding: text address → lat/lng ─────────────────────────────────────────
def geocode(address):
    """Convert text address to lat/lng using Nominatim (free, no key needed)"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":      address + ", Karachi, Pakistan",
        "format": "json",
        "limit":  1
    }
    headers = {"User-Agent": "raahsawari-app"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    return None, None

# ── Distance Calculator ───────────────────────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
    return round(R * 2 * np.arcsin(np.sqrt(a)) * 1.4, 2)

# ── Driver Arrival Estimate ───────────────────────────────────────────────────
def get_driver_arrival(traffic_level):
    """Estimate how long until driver reaches user"""
    if traffic_level == "high":
        return round(np.random.uniform(8, 15), 1)    # 8–15 mins in heavy traffic
    elif traffic_level == "medium":
        return round(np.random.uniform(4, 8), 1)     # 4–8 mins
    else:
        return round(np.random.uniform(2, 5), 1)     # 2–5 mins low traffic

# ── Main ETA Endpoint ─────────────────────────────────────────────────────────
@app.route("/predict-eta", methods=["POST"])
def predict_eta():
    try:
        data = request.get_json()

        pickup_text      = data["pickup"]       # "DHA Phase 2 Karachi"
        destination_text = data["destination"]  # "Gulshan-e-Iqbal Karachi"

        # Step 1: Geocode addresses
        origin_lat, origin_lng           = geocode(pickup_text)
        destination_lat, destination_lng = geocode(destination_text)

        if not all([origin_lat, origin_lng, destination_lat, destination_lng]):
            return jsonify({"status": "error", "message": "Could not find one or both locations"}), 400

        # Step 2: Calculate distance and hour automatically
        distance_km = haversine_km(origin_lat, origin_lng, destination_lat, destination_lng)
        hour_of_day = datetime.now().hour

        # Step 3: Predict ETA and traffic for all 3 vehicles
        vehicles = ["bike", "car", "rickshaw"]
        results  = []

        for vehicle in vehicles:
            vehicle_enc = le_vehicle.transform([vehicle])[0]

            # Predict traffic from your trained CSV model
            traffic_enc = traffic_model.predict(pd.DataFrame([[
                distance_km, hour_of_day, vehicle_enc
            ]], columns=["distance_km", "hour_of_day", "vehicle_type_enc"]))[0]

            traffic_label = le_traffic.inverse_transform([traffic_enc])[0]

            # Predict ETA
            eta_min = eta_model.predict(pd.DataFrame([[
                origin_lat, origin_lng,
                destination_lat, destination_lng,
                distance_km,
                vehicle_enc,
                hour_of_day,
                traffic_enc
            ]], columns=[
                "origin_lat", "origin_lng",
                "destination_lat", "destination_lng",
                "distance_km", "vehicle_type_enc",
                "hour_of_day", "traffic_level_enc"
            ]))[0]

            # Driver arrival time on top of ETA
            driver_arrival = get_driver_arrival(traffic_label)
            total_time     = round(eta_min + driver_arrival, 1)

            results.append({
                "vehicle":             vehicle,
                "trip_eta_min":        round(eta_min, 1),
                "driver_arrival_min":  driver_arrival,
                "total_time_min":      total_time,
                "traffic_level":       traffic_label,
            })

        return jsonify({
            "status":      "success",
            "pickup":      pickup_text,
            "destination": destination_text,
            "origin_lat":  origin_lat,
            "origin_lng":  origin_lng,
            "dest_lat":    destination_lat,
            "dest_lng":    destination_lng,
            "distance_km": distance_km,
            "hour":        hour_of_day,
            "rides":       results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)