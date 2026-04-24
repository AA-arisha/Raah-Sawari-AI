# 🚖 Raah Sawari — AI-Powered ETA Prediction Backend

This is the AI/ML backend for Raah Sawari, a ride-hailing app for Karachi.
It uses a Random Forest model trained on Karachi traffic data to predict
ETA (Estimated Time of Arrival) for bike, car, and rickshaw rides.

---

## 📁 Dataset

The training dataset is not included in this repository due to file size.

📥 **Download CSV file from Google Drive:**
👉((https://drive.google.com/file/d/1ciTqix5m_mvFdMFopmAUS4cW7cppjxry/view?usp=sharing)E))

After downloading, place it in the same folder as the Python files:
raahsawari/
├── ETA2_dataset.csv   ← place here
├── ETAcode.py
├── ETAapp.py
└── ETAtest.py
---

## ⚙️ Installation

Install required libraries:
```bash
pip install pandas numpy scikit-learn joblib flask requests
```

---

## 🚀 How to Run

### Step 1 — Train the Model
Run this first to train the Random Forest on the CSV data.
This will generate the `.pkl` model files.

```bash
python ETAcode.py
```

You should see:
✅ Dataset loaded: 12000 rows
🌲 Training ETA Random Forest...
✅ ETA Model trained!
🌲 Training Traffic Random Forest...
✅ Traffic Model trained!
💾 All models saved:
✅ eta_model.pkl
✅ traffic_model.pkl
✅ le_vehicle.pkl
✅ le_traffic.pkl
---

### Step 2 — Start the Flask API
Open a terminal and run:

```bash
python ETAapp.py
```

You should see:
Running on http://127.0.0.1:5000
Debug mode: on
**Keep this terminal open and running.**

---

### Step 3 — Test the API
Open a **second terminal** and run:

```bash
python ETAtest.py
```

You should see a response like:
```json
{
    "status": "success",
    "distance_km": 7.29,
    "rides": [
        {
            "vehicle": "bike",
            "trip_eta_min": 8.3,
            "driver_arrival_min": 2.5,
            "total_time_min": 10.8,
            "traffic_level": "low"
        },
        {
            "vehicle": "car",
            "trip_eta_min": 7.6,
            "driver_arrival_min": 3.3,
            "total_time_min": 10.9,
            "traffic_level": "low"
        },
        {
            "vehicle": "rickshaw",
            "trip_eta_min": 15.0,
            "driver_arrival_min": 3.8,
            "total_time_min": 18.8,
            "traffic_level": "low"
        }
    ]
}
```

---

## 📡 API Usage

**Endpoint:** `POST /predict-eta`

**Request Body:**
```json
{
    "pickup":      "DHA Phase 2 Karachi",
    "destination": "Gulshan-e-Iqbal Karachi"
}
```

**Response:**
```json
{
    "status":      "success",
    "distance_km": 7.29,
    "hour":        17,
    "rides": [
        {
            "vehicle":            "bike",
            "trip_eta_min":       8.3,
            "driver_arrival_min": 2.5,
            "total_time_min":     10.8,
            "traffic_level":      "low"
        }
    ]
}
```

---

## 🧠 How It Works
User types pickup + destination (text)
↓
Nominatim API converts text → lat/lng coordinates (free, no key needed)
↓
System auto calculates:
• distance_km  — from coordinates (Haversine formula)
• hour_of_day  — from system clock
↓
Random Forest Model 1 predicts:
• traffic_level — learned from 12,000 Karachi trip records
↓
Random Forest Model 2 predicts:
• trip_duration_min — ETA for bike, car, rickshaw
↓
Driver arrival time added based on traffic level
↓
Returns full response to frontend
---

## 🛠️ Tech Stack

- **Python 3.13**
- **scikit-learn** — Random Forest
- **Flask** — REST API
- **Nominatim** — Free geocoding
- **pandas / numpy** — Data processing

---

## 👩‍💻 Developed by
Raah Sawari AI Team
