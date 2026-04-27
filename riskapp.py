from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# ════════════════════════════════════════════════════════════════
# Load model and scaler once when API starts
# ════════════════════════════════════════════════════════════════
with open("knn_driver_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("driver_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

RISK_LABELS = {0: "Low Risk",    1: "Medium Risk",  2: "High Risk"}
RISK_EMOJI  = {0: "✅",          1: "⚠️",           2: "🚨"}

# ════════════════════════════════════════════════════════════════
# ROUTE 1 — Home
# ════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message" : "Driver Risk Analysis API is running!",
        "status"  : "success",
        "routes"  : {
            "GET  /"             : "Check if API is running",
            "POST /predict-risk" : "Predict driver risk level",
            "GET  /health"       : "Health check"
        }
    })

# ════════════════════════════════════════════════════════════════
# ROUTE 2 — Predict Risk (Main Route)
# SDA team calls this when passenger books a ride
# ════════════════════════════════════════════════════════════════
@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    try:
        data = request.get_json()

        # Check all required fields exist
        required_fields = [
            "driver_id", "trips_completed", "driver_rating",
            "cancellation_rate", "experience_years",
            "night_trips_ratio", "complaints_count"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error"  : f"Missing field: {field}",
                    "status" : "failed"
                }), 400

        # Build DataFrame with same column names as training
        input_df = pd.DataFrame([{
            "trips_completed"   : data["trips_completed"],
            "driver_rating"     : data["driver_rating"],
            "cancellation_rate" : data["cancellation_rate"],
            "experience_years"  : data["experience_years"],
            "night_trips_ratio" : data["night_trips_ratio"],
            "complaints_count"  : data["complaints_count"]
        }])

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        risk_level   = int(model.predict(input_scaled)[0])

        return jsonify({
            "status"     : "success",
            "driver_id"  : data["driver_id"],
            "risk_level" : risk_level,
            "risk_label" : RISK_LABELS[risk_level],
            "message"    : f"Driver is {RISK_LABELS[risk_level]} {RISK_EMOJI[risk_level]}"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

# ════════════════════════════════════════════════════════════════
# ROUTE 3 — Health Check
# ════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "healthy",
        "model"   : "KNN K=5",
        "version" : "1.0"
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  Driver Risk Analysis API Starting...")
    print("  URL : http://localhost:5000")
    print("  POST: http://localhost:5000/predict-risk")
    print("=" * 50)
    app.run(debug=True, port=5000)