
import requests
import pandas as pd
import random
import time
import os
from datetime import datetime, timezone

# 🔑 Your TomTom API key
API_KEY = "7P4psmZgGyoSQc1bwplcm6LrENOkwkqu"

file_name = "ETA_dataset.csv"

def get_valid_location():
    lat = random.uniform(24.80, 25.00)
    lng = random.uniform(67.00, 67.30)
    return lat, lng

data = []

for i in range(100):

    origin = get_valid_location()
    destination = get_valid_location()

    vehicle = random.choice(["car", "bike", "rickshaw"])

    if vehicle == "bike":
        travel_mode = "motorcycle"
    else:
        travel_mode = "car"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}/json"
        f"?travelMode={travel_mode}"
        f"&traffic=true"
        f"&departAt={now}"
        f"&key={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        result = response.json()

        print(f"🔍 Row {i+1} response status: {result.get('formatVersion', 'check error')}")

        if "routes" not in result or len(result["routes"]) == 0:
            print(f"⚠️ No routes found: {result}")
            continue

        summary = result["routes"][0]["summary"]

        distance_km = summary["lengthInMeters"] / 1000
        duration_min = summary["travelTimeInSeconds"] / 60
        traffic_delay_min = summary.get("trafficDelayInSeconds", 0) / 60

        if distance_km < 1:
            print(f"⚠️ Too short ({distance_km:.2f} km), skipping...")
            continue

        # 🚦 Traffic level
        if traffic_delay_min > 1:
            traffic = "high"
        elif traffic_delay_min > 0.2:
            traffic = "medium"
        else:
            traffic = "low"

        # 🛺 Rickshaw adjustment
        if vehicle == "rickshaw":
            duration_min *= 1.25

        # 📏 Average speed
        avg_speed = (distance_km / duration_min) * 60

        hour = datetime.now().hour

        data.append([
            round(origin[0], 6),      # origin lat
            round(origin[1], 6),      # origin lng
            round(destination[0], 6), # destination lat
            round(destination[1], 6), # destination lng
            round(distance_km, 2),
            vehicle,
            hour,
            traffic,
            round(duration_min, 2),
            round(avg_speed, 2)
        ])

        print(f"✔ Row {i+1}: {vehicle} | from ({origin[0]:.4f},{origin[1]:.4f}) to ({destination[0]:.4f},{destination[1]:.4f}) | {distance_km:.2f} km | {duration_min:.2f} min | speed: {avg_speed:.1f} km/h | traffic: {traffic}")

        time.sleep(1)

    except Exception as e:
        print(f"❌ Error on row {i+1}: {e}")
        continue


# 💾 Save dataset
if len(data) == 0:
    print("❌ No data collected. Check your API key!")
else:
    df = pd.DataFrame(data, columns=[
        "origin_lat",
        "origin_lng",
        "destination_lat",
        "destination_lng",
        "distance_km",
        "vehicle_type",
        "hour_of_day",
        "traffic_level",
        "trip_duration_min",
        "avg_speed_kmh"
    ])

    if not os.path.exists(file_name):
        df.to_csv(file_name, index=False)
        print(f"✅ File created with {len(df)} rows")
    else:
        df.to_csv(file_name, mode='a', header=False, index=False)
        print(f"✅ {len(df)} rows appended to existing file")

    print(f"📁 Saved at: {os.path.abspath(file_name)}")
    print("\n📊 Preview:")
    print(df.head())