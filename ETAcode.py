import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 1 — ETA Regressor (predicts trip_duration_min)
# ════════════════════════════════════════════════════════════════════════════════
ETA_FEATURES = [
    "origin_lat", "origin_lng",
    "destination_lat", "destination_lng",
    "distance_km",
    "vehicle_type_enc",
    "hour_of_day",
    "traffic_level_enc"
]

X_eta = df[ETA_FEATURES]
y_eta = df["trip_duration_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X_eta, y_eta, test_size=0.2, random_state=42
)
print(f"\n📦 ETA Train size: {len(X_train)} | Test size: {len(X_test)}")

print("\n🌲 Training ETA Random Forest...")
eta_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
eta_model.fit(X_train, y_train)
print("✅ ETA Model trained!")

# ── Evaluate ETA Model ────────────────────────────────────────────────────────
y_pred = eta_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)
r2     = r2_score(y_test, y_pred)

print("\n📊 ETA Model Performance:")
print(f"   MAE  : {mae:.2f} minutes")
print(f"   MSE  : {mse:.2f}")
print(f"   RMSE : {rmse:.2f} minutes")
print(f"   R²   : {r2:.4f} ({r2*100:.2f}%)")

print("\n🔍 ETA Feature Importance:")
importance_df = pd.DataFrame({
    "Feature":    ETA_FEATURES,
    "Importance": eta_model.feature_importances_
}).sort_values("Importance", ascending=False)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<25} {bar} {row['Importance']:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Traffic Classifier (predicts traffic_level from CSV patterns)
# ════════════════════════════════════════════════════════════════════════════════
TRAFFIC_FEATURES = [
    "distance_km",
    "hour_of_day",
    "vehicle_type_enc"
]

X_traffic = df[TRAFFIC_FEATURES]
y_traffic = df["traffic_level_enc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_traffic, y_traffic, test_size=0.2, random_state=42
)
print(f"\n📦 Traffic Train size: {len(X_tr)} | Test size: {len(X_te)}")

print("\n🌲 Training Traffic Random Forest...")
traffic_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
traffic_model.fit(X_tr, y_tr)
print("✅ Traffic Model trained!")

# ── Evaluate Traffic Model ────────────────────────────────────────────────────
traffic_acc = traffic_model.score(X_te, y_te)
print(f"\n📊 Traffic Classifier Accuracy: {traffic_acc*100:.2f}%")

print("\n🔍 Traffic Feature Importance:")
t_importance_df = pd.DataFrame({
    "Feature":    TRAFFIC_FEATURES,
    "Importance": traffic_model.feature_importances_
}).sort_values("Importance", ascending=False)
for _, row in t_importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<25} {bar} {row['Importance']:.4f}")

# ── Save All Models ───────────────────────────────────────────────────────────
joblib.dump(eta_model,     "eta_model.pkl")
joblib.dump(traffic_model, "traffic_model.pkl")
joblib.dump(le_vehicle,    "le_vehicle.pkl")
joblib.dump(le_traffic,    "le_traffic.pkl")

print("\n💾 All models saved:")
print("   ✅ eta_model.pkl")
print("   ✅ traffic_model.pkl")
print("   ✅ le_vehicle.pkl")
print("   ✅ le_traffic.pkl")

# ── Test Prediction ───────────────────────────────────────────────────────────
print("\n🔮 Test Prediction:")

origin_lat, origin_lng         = 24.8920, 67.0580
destination_lat, destination_lng = 24.9300, 67.0900
distance_km = 8.5
hour_of_day = datetime.now().hour if 'datetime' in dir() else 8

from datetime import datetime
hour_of_day = datetime.now().hour

for vehicle in ["bike", "car", "rickshaw"]:
    vehicle_enc = le_vehicle.transform([vehicle])[0]

    # Traffic predicted by YOUR CSV trained model
    traffic_enc   = traffic_model.predict(pd.DataFrame([[
        distance_km, hour_of_day, vehicle_enc
    ]], columns=TRAFFIC_FEATURES))[0]
    traffic_label = le_traffic.inverse_transform([traffic_enc])[0]

    # ETA predicted by YOUR CSV trained model
    eta_min = eta_model.predict(pd.DataFrame([[
        origin_lat, origin_lng,
        destination_lat, destination_lng,
        distance_km, vehicle_enc,
        hour_of_day, traffic_enc
    ]], columns=ETA_FEATURES))[0]

    print(f"\n   🚗 {vehicle.upper()}")
    print(f"      Traffic  : {traffic_label}")
    print(f"      ETA      : {eta_min:.1f} minutes")