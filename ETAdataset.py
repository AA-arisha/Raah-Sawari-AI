import pandas as pd
import numpy as np
import random
import os

file_name = "ETA2_dataset.csv"
NUM_ROWS = 1000  # change as needed

# Karachi zones with realistic coordinates
ZONES = {
    "DHA":             (24.795, 67.065),
    "Clifton":         (24.810, 67.030),
    "Saddar":          (24.860, 67.010),
    "Gulshan":         (24.930, 67.090),
    "North_Nazimabad": (24.945, 67.060),
    "Korangi":         (24.830, 67.130),
    "Malir":           (24.895, 67.205),
    "Orangi":          (24.955, 66.990),
    "Landhi":          (24.855, 67.195),
    "FB_Area":         (24.940, 67.075),
    "PECHS":           (24.870, 67.055),
    "Nazimabad":       (24.920, 67.045),
    "Liaquatabad":     (24.905, 67.035),
    "Surjani":         (24.990, 67.025),
    "Baldia":          (24.900, 66.990),
}

ZONE_NAMES = list(ZONES.keys())

def get_location_near_zone(zone_name, spread=0.015):
    base_lat, base_lng = ZONES[zone_name]
    lat = base_lat + random.uniform(-spread, spread)
    lng = base_lng + random.uniform(-spread, spread)
    return round(lat, 6), round(lng, 6)

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def get_hour():
    """Weighted random hour — realistic Karachi activity pattern"""
    hours   = list(range(24))
    weights = [
        1, 1, 1, 1, 1, 2,   # 0–5  late night / very early
        3, 5, 8, 7, 5, 5,   # 6–11 morning + rush
        6, 6, 5, 5, 5, 7,   # 12–17 afternoon
        9, 8, 6, 4, 3, 2    # 18–23 evening peak + wind down
    ]
    return random.choices(hours, weights=weights)[0]

def get_traffic(hour):
    """Realistic Karachi traffic distribution by hour"""
    if hour in [8, 9, 18, 19, 20]:         # peak rush hours
        weights = [0.10, 0.25, 0.65]
    elif hour in [7, 10, 17, 21]:           # semi-peak
        weights = [0.25, 0.45, 0.30]
    elif hour in [12, 13, 14]:              # afternoon moderate
        weights = [0.40, 0.40, 0.20]
    elif hour in [0, 1, 2, 3, 4, 5]:       # late night
        weights = [0.95, 0.04, 0.01]
    else:                                    # normal hours
        weights = [0.60, 0.30, 0.10]
    return random.choices(["low", "medium", "high"], weights=weights)[0]

def get_speed(vehicle, traffic):
    """Realistic Karachi speeds km/h per vehicle and traffic"""
    speeds = {
        "car":      {"low": (45, 65), "medium": (25, 44), "high": (10, 24)},
        "bike":     {"low": (40, 60), "medium": (25, 39), "high": (12, 24)},
        "rickshaw": {"low": (25, 35), "medium": (15, 24), "high": (8,  14)},
    }
    lo, hi = speeds[vehicle][traffic]
    speed = random.uniform(lo, hi) + random.uniform(-2, 2)  # small noise
    return round(max(5, speed), 2)

# ── Generate ──────────────────────────────────────────────────────────────────
data = []
attempts = 0

while len(data) < NUM_ROWS:
    attempts += 1

    origin_zone, dest_zone = random.sample(ZONE_NAMES, 2)
    origin      = get_location_near_zone(origin_zone)
    destination = get_location_near_zone(dest_zone)

    straight_km = haversine_km(origin[0], origin[1], destination[0], destination[1])
    road_km     = round(straight_km * random.uniform(1.3, 1.6), 2)

    if road_km < 2.0:
        continue  # skip unrealistically short trips

    vehicle  = random.choices(["car", "bike", "rickshaw"], weights=[0.50, 0.30, 0.20])[0]
    hour     = get_hour()
    traffic  = get_traffic(hour)
    speed    = get_speed(vehicle, traffic)
    duration = round((road_km / speed) * 60, 2)

    data.append([
        origin[0],      origin[1],
        destination[0], destination[1],
        road_km,
        vehicle,
        hour,
        traffic,
        duration,
        speed
    ])

    if len(data) % 100 == 0:
        print(f"✔ {len(data)}/{NUM_ROWS} rows generated...")

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(data, columns=[
    "origin_lat", "origin_lng",
    "destination_lat", "destination_lng",
    "distance_km", "vehicle_type",
    "hour_of_day", "traffic_level",
    "trip_duration_min", "avg_speed_kmh"
])

if not os.path.exists(file_name):
    df.to_csv(file_name, index=False)
    print(f"\n✅ File created: {len(df)} rows")
else:
    df.to_csv(file_name, mode='a', header=False, index=False)
    print(f"\n✅ {len(df)} rows appended to existing file")

print(f"📁 Saved at: {os.path.abspath(file_name)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n📊 Traffic Distribution:")
print(df["traffic_level"].value_counts())
print("\n🚗 Vehicle Distribution:")
print(df["vehicle_type"].value_counts())
print("\n🕐 Hour Distribution:")
print(df["hour_of_day"].value_counts().sort_index())
print("\n📈 Preview:")
print(df.head(10).to_string(index=False))