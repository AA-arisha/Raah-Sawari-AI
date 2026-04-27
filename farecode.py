import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("fare_dataset.csv")
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── Encode Categorical Columns ────────────────────────────────────────────────
le_vehicle = LabelEncoder()
le_traffic  = LabelEncoder()
le_time     = LabelEncoder()
le_demand   = LabelEncoder()
le_supply   = LabelEncoder()

df["vehicle_type_enc"]  = le_vehicle.fit_transform(df["vehicle_type"])
df["traffic_level_enc"] = le_traffic.fit_transform(df["traffic_level"])
df["time_of_day_enc"]   = le_time.fit_transform(df["time_of_day"])
df["demand_level_enc"]  = le_demand.fit_transform(df["demand_level"])
df["supply_level_enc"]  = le_supply.fit_transform(df["supply_level"])

print(f"\n🚗 Vehicle : {dict(zip(le_vehicle.classes_, le_vehicle.transform(le_vehicle.classes_)))}")
print(f"🚦 Traffic : {dict(zip(le_traffic.classes_, le_traffic.transform(le_traffic.classes_)))}")
print(f"🕐 Time    : {dict(zip(le_time.classes_,    le_time.transform(le_time.classes_)))}")
print(f"📈 Demand  : {dict(zip(le_demand.classes_,  le_demand.transform(le_demand.classes_)))}")
print(f"📉 Supply  : {dict(zip(le_supply.classes_,  le_supply.transform(le_supply.classes_)))}")

# ── Features & Target ─────────────────────────────────────────────────────────
FEATURES = [
    "vehicle_type_enc",
    "distance_km",
    "duration_minutes",
    "time_of_day_enc",
    "traffic_level_enc",
    "demand_level_enc",
    "supply_level_enc",
    "petrol_price_per_liter",
    "fuel_cost"
]

TARGET = "recommended_fare"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📦 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Train Gradient Boosting ───────────────────────────────────────────────────
print("\n🚀 Training Gradient Boosting...")
fare_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
fare_model.fit(X_train, y_train)
print("✅ Fare Model trained!")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = fare_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)
r2     = r2_score(y_test, y_pred)

print("\n📊 Fare Model Performance:")
print(f"   MAE  : {mae:.2f} PKR")
print(f"   MSE  : {mse:.2f}")
print(f"   RMSE : {rmse:.2f} PKR")
print(f"   R²   : {r2:.4f} ({r2*100:.2f}%)")

# ── Feature Importance ────────────────────────────────────────────────────────
print("\n🔍 Feature Importance:")
importance_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": fare_model.feature_importances_
}).sort_values("Importance", ascending=False)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<25} {bar} {row['Importance']:.4f}")

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(fare_model, "fare_model.pkl")
joblib.dump(le_vehicle, "fare_le_vehicle.pkl")
joblib.dump(le_traffic, "fare_le_traffic.pkl")
joblib.dump(le_time,    "fare_le_time.pkl")
joblib.dump(le_demand,  "fare_le_demand.pkl")
joblib.dump(le_supply,  "fare_le_supply.pkl")

print("\n💾 All models saved:")
print("   ✅ fare_model.pkl")
print("   ✅ fare_le_vehicle.pkl")
print("   ✅ fare_le_traffic.pkl")
print("   ✅ fare_le_time.pkl")
print("   ✅ fare_le_demand.pkl")
print("   ✅ fare_le_supply.pkl")
print("\n🎉 Training complete! Now run fareapp.py")