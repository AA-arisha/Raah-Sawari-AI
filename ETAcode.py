import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("ETA2_dataset.csv")
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Encode Categorical Columns ────────────────────────────────────────────────
le_vehicle = LabelEncoder()
le_traffic  = LabelEncoder()

df["vehicle_type_enc"]  = le_vehicle.fit_transform(df["vehicle_type"])
df["traffic_level_enc"] = le_traffic.fit_transform(df["traffic_level"])

print(f"\n🚗 Vehicle encoding : {dict(zip(le_vehicle.classes_, le_vehicle.transform(le_vehicle.classes_)))}")
print(f"🚦 Traffic encoding : {dict(zip(le_traffic.classes_, le_traffic.transform(le_traffic.classes_)))}")

# ── Features & Target ─────────────────────────────────────────────────────────
FEATURES = [
    "origin_lat", "origin_lng",
    "destination_lat", "destination_lng",
    "distance_km",
    "vehicle_type_enc",
    "hour_of_day",
    "traffic_level_enc"
    # avg_speed_kmh REMOVED so other features become important
]

TARGET = "trip_duration_min"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📦 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Train Model ───────────────────────────────────────────────────────────────
print("\n🌲 Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"   MAE  (Mean Absolute Error)  : {mae:.2f} minutes")
print(f"   MSE  (Mean Squared Error)   : {mse:.2f}")
print(f"   RMSE (Root MSE)             : {rmse:.2f} minutes")
print(f"   R²   (Accuracy Score)       : {r2:.4f} ({r2*100:.2f}%)")

# ── Feature Importance ────────────────────────────────────────────────────────
print("\n🔍 Feature Importance:")
importance_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<25} {bar} {row['Importance']:.4f}")

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(model,      "eta_model.pkl")
joblib.dump(le_vehicle, "le_vehicle.pkl")
joblib.dump(le_traffic, "le_traffic.pkl")
print("\n💾 Model saved as eta_model.pkl")

# ── New Prediction ────────────────────────────────────────────────────────────
print("\n🔮 New Prediction Example:")

new_trip = {
    "origin_lat":      24.8920,
    "origin_lng":      67.0580,
    "destination_lat": 24.9300,
    "destination_lng": 67.0900,
    "distance_km":     8.5,
    "vehicle_type":    "car",       # car / bike / rickshaw
    "hour_of_day":     8,           # 0–23
    "traffic_level":   "high",      # low / medium / high
}

new_trip["vehicle_type_enc"]  = le_vehicle.transform([new_trip["vehicle_type"]])[0]
new_trip["traffic_level_enc"] = le_traffic.transform([new_trip["traffic_level"]])[0]

new_df = pd.DataFrame([[
    new_trip["origin_lat"],
    new_trip["origin_lng"],
    new_trip["destination_lat"],
    new_trip["destination_lng"],
    new_trip["distance_km"],
    new_trip["vehicle_type_enc"],
    new_trip["hour_of_day"],
    new_trip["traffic_level_enc"],
]], columns=FEATURES)

predicted_duration = model.predict(new_df)[0]

print(f"   Vehicle    : {new_trip['vehicle_type']}")
print(f"   Distance   : {new_trip['distance_km']} km")
print(f"   Hour       : {new_trip['hour_of_day']}:00")
print(f"   Traffic    : {new_trip['traffic_level']}")
print(f"\n   ⏱️  Predicted ETA : {predicted_duration:.2f} minutes")