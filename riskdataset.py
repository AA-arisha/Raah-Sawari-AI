import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

NUM_DRIVERS = 5000

data = []

for i in range(1, NUM_DRIVERS + 1):
    driver_id = f"D{i:04d}"

    # ── Generate raw feature values ──────────────────────────────

    trips_completed     = random.randint(0, 4000)
    driver_rating       = round(random.uniform(1.0, 5.0), 1)
    cancellation_rate   = round(random.uniform(0.0, 0.65), 2)
    experience_years    = random.randint(0, 10)
    night_trips_ratio   = round(random.uniform(0.0, 1.0), 2)
    complaints_count    = random.randint(0, 20)

    # ── Calculate risk score ─────────────────────────────────────
    score = 0

    # trips_completed
    if trips_completed <= 50:
        score += 3
    elif trips_completed <= 200:
        score += 2
    elif trips_completed <= 500:
        score += 1

    # driver_rating
    if driver_rating < 2.5:
        score += 3
    elif driver_rating < 3.5:
        score += 2
    elif driver_rating < 4.2:
        score += 1

    # cancellation_rate
    if cancellation_rate > 0.5:
        score += 3
    elif cancellation_rate > 0.3:
        score += 2
    elif cancellation_rate > 0.15:
        score += 1

    # experience_years
    if experience_years == 0:
        score += 2
    elif experience_years == 1:
        score += 1

    # night_trips_ratio
    if night_trips_ratio > 0.7:
        score += 2
    elif night_trips_ratio > 0.4:
        score += 1

    # complaints_count
    if complaints_count > 10:
        score += 3
    elif complaints_count > 5:
        score += 2
    elif complaints_count > 2:
        score += 1

    # ── Assign risk label ────────────────────────────────────────
    if score >= 8:
        risk_label = 2      # High Risk
    elif score >= 4:
        risk_label = 1      # Medium Risk
    else:
        risk_label = 0      # Low Risk

    data.append([
        driver_id,
        trips_completed,
        driver_rating,
        cancellation_rate,
        experience_years,
        night_trips_ratio,
        complaints_count,
        score,
        risk_label
    ])

# ── Save to CSV ──────────────────────────────────────────────────
df = pd.DataFrame(data, columns=[
    "driver_id",
    "trips_completed",
    "driver_rating",
    "cancellation_rate",
    "experience_years",
    "night_trips_ratio",
    "complaints_count",
    "risk_score",        # keeping this so you can see how score was calculated
    "risk_label"        # 0=Low  1=Medium  2=High
])

df.to_csv("driver_risk_dataset.csv", index=False)

# ── Print summary ────────────────────────────────────────────────
print("=" * 50)
print(f"Total drivers generated : {len(df)}")
print("=" * 50)

counts = df["risk_label"].value_counts().sort_index()
print(f"Low Risk    (0) : {counts.get(0, 0)} drivers")
print(f"Medium Risk (1) : {counts.get(1, 0)} drivers")
print(f"High Risk   (2) : {counts.get(2, 0)} drivers")

print("\nSample rows:")
print(df.head(10).to_string(index=False))