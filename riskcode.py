import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ════════════════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ════════════════════════════════════════════════════════════════
print("=" * 55)
print("  DRIVER RISK ANALYSIS — KNN TRAINING")
print("=" * 55)

df = pd.read_csv("driver_risk_dataset.csv")
print(f"\nDataset loaded: {len(df)} drivers")
print(f"Columns: {list(df.columns)}\n")

# ════════════════════════════════════════════════════════════════
# STEP 2 — Select Features and Target
# ════════════════════════════════════════════════════════════════
features = [
    "trips_completed",
    "driver_rating",
    "cancellation_rate",
    "experience_years",
    "night_trips_ratio",
    "complaints_count"
]

target = "risk_label"

X = df[features]
y = df[target]

print(f"Features used for training : {features}")
print(f"Target column              : {target}")
print(f"\nRisk label distribution:")
print(f"  Low Risk    (0) : {sum(y == 0)} drivers")
print(f"  Medium Risk (1) : {sum(y == 1)} drivers")
print(f"  High Risk   (2) : {sum(y == 2)} drivers")

# ════════════════════════════════════════════════════════════════
# STEP 3 — Split into Training and Testing
# ════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining rows : {len(X_train)} (80%)")
print(f"Testing rows  : {len(X_test)}  (20%)")

# ════════════════════════════════════════════════════════════════
# STEP 4 — Scale Features (VERY IMPORTANT FOR KNN)
# ════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nFeatures scaled successfully (StandardScaler)")

# ════════════════════════════════════════════════════════════════
# STEP 5 — Train KNN Model
# ════════════════════════════════════════════════════════════════
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

print(f"\nKNN model trained with K = {k}")

# ════════════════════════════════════════════════════════════════
# STEP 6 — Test the Model
# ════════════════════════════════════════════════════════════════
y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print(f"  MODEL ACCURACY : {accuracy * 100:.2f}%")
print("=" * 55)

print("\nDetailed Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Risk", "Medium Risk", "High Risk"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("(Rows = Actual, Columns = Predicted)")

# ════════════════════════════════════════════════════════════════
# STEP 7 — Save Model and Scaler
# ════════════════════════════════════════════════════════════════
with open("knn_driver_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("driver_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n" + "=" * 55)
print("  FILES SAVED")
print("=" * 55)
print("  knn_driver_model.pkl  → trained KNN model")
print("  driver_scaler.pkl     → scaler for new inputs")
print("\nTraining complete! Now run app.py to start the API.")