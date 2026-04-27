import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

NUM_TRIPS = 5000

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
VEHICLE_FUEL_CONSUMPTION = {
    "bike"    : 35,   # km per liter
    "rickshaw": 25,
    "car"     : 12
}

VEHICLE_BASE_RATE = {
    "bike"    : 10,   # base fare per km (PKR)
    "rickshaw": 15,
    "car"     : 25
}

TRAFFIC_MULTIPLIER = {
    "low"   : 1.0,
    "medium": 1.15,
    "high"  : 1.30
}

DEMAND_SUPPLY_MULTIPLIER = {
    ("high",   "low")   : 1.35,   # high demand, low supply  → surge
    ("high",   "medium"): 1.20,
    ("high",   "high")  : 1.05,
    ("medium", "low")   : 1.15,
    ("medium", "medium"): 1.00,
    ("medium", "high")  : 0.95,
    ("low",    "low")   : 1.00,
    ("low",    "medium"): 0.90,
    ("low",    "high")  : 0.85
}

TIME_MULTIPLIER = {
    "morning"  : 1.10,   # rush hour
    "afternoon": 1.00,
    "evening"  : 1.15,   # rush hour
    "night"    : 1.20    # late night premium
}

# ════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ════════════════════════════════════════════════════════════════
print("=" * 55)
print("  FARE RECOMMENDATION — DATASET GENERATION")
print("=" * 55)

data = []

for i in range(1, NUM_TRIPS + 1):
    trip_id = f"T{i:04d}"

    # ── Random feature values ─────────────────────────────────
    vehicle_type       = random.choice(["bike", "rickshaw", "car"])
    distance_km        = round(random.uniform(1.0, 40.0), 2)
    duration_minutes   = round(distance_km * random.uniform(2.5, 5.0), 1)
    time_of_day        = random.choice(["morning", "afternoon", "evening", "night"])
    traffic_level      = random.choice(["low", "medium", "high"])
    demand_level       = random.choice(["low", "medium", "high"])
    supply_level       = random.choice(["low", "medium", "high"])
    petrol_price       = round(random.uniform(250.0, 320.0), 1)   # PKR per liter

    # ── Fuel cost for this trip ───────────────────────────────
    liters_used        = distance_km / VEHICLE_FUEL_CONSUMPTION[vehicle_type]
    fuel_cost          = round(liters_used * petrol_price, 2)

    # ── Base fare ─────────────────────────────────────────────
    base_fare          = distance_km * VEHICLE_BASE_RATE[vehicle_type]

    # ── Apply multipliers ─────────────────────────────────────
    traffic_mult       = TRAFFIC_MULTIPLIER[traffic_level]
    demand_supply_mult = DEMAND_SUPPLY_MULTIPLIER[(demand_level, supply_level)]
    time_mult          = TIME_MULTIPLIER[time_of_day]

    # ── Recommended fare ──────────────────────────────────────
    recommended_fare   = base_fare * traffic_mult * demand_supply_mult * time_mult
    recommended_fare   = round(recommended_fare + fuel_cost, 2)

    # ── Min and Max fare ──────────────────────────────────────
    min_fare           = round(recommended_fare * 0.80, 2)
    max_fare           = round(recommended_fare * 1.25, 2)

    data.append([
        trip_id,
        vehicle_type,
        distance_km,
        duration_minutes,
        time_of_day,
        traffic_level,
        demand_level,
        supply_level,
        petrol_price,
        fuel_cost,
        min_fare,
        recommended_fare,
        max_fare
    ])

# ════════════════════════════════════════════════════════════════
# SAVE TO CSV
# ════════════════════════════════════════════════════════════════
df = pd.DataFrame(data, columns=[
    "trip_id",
    "vehicle_type",
    "distance_km",
    "duration_minutes",
    "time_of_day",
    "traffic_level",
    "demand_level",
    "supply_level",
    "petrol_price_per_liter",
    "fuel_cost",
    "min_fare",
    "recommended_fare",
    "max_fare"
])

df.to_csv("fare_dataset.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"\nTotal trips generated : {len(df)}")
print(f"\nVehicle distribution:")
print(f"  Bike     : {len(df[df['vehicle_type'] == 'bike'])}")
print(f"  Rickshaw : {len(df[df['vehicle_type'] == 'rickshaw'])}")
print(f"  Car      : {len(df[df['vehicle_type'] == 'car'])}")

print(f"\nFare ranges:")
print(f"  Min fare         : {df['min_fare'].min()} – {df['min_fare'].max()} PKR")
print(f"  Recommended fare : {df['recommended_fare'].min()} – {df['recommended_fare'].max()} PKR")
print(f"  Max fare         : {df['max_fare'].min()} – {df['max_fare'].max()} PKR")

print(f"\nSample rows:")
print(df.head(5).to_string(index=False))

print("\nDataset saved to fare_dataset.csv")