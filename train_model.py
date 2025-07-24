import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from datetime import datetime
import pytz

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load trained model
model = tf.keras.models.load_model("yield_model.keras")

# Fetch new sensor data (without actual yield)
docs = db.collection("dataCollectionSensor").stream()

unlabeled_data = []
docs_to_update = []

for doc in docs:
    record = doc.to_dict()
    if all(k in record for k in ["temperature", "humidity", "localMoisture"]):
        unlabeled_data.append([
            int(record["temperature"]),
            int(record["humidity"]),
            int(record["localMoisture"])
        ])
        docs_to_update.append((doc.id, record))

if not unlabeled_data:
    print("No new sensor data to predict.")
    exit()

# Scale input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(unlabeled_data)

# Predict yield
predicted_yields = model.predict(X_scaled).flatten()

# Get current Philippine time
ph_tz = pytz.timezone("Asia/Manila")

for i, (doc_id, original) in enumerate(docs_to_update):
    now = datetime.now(ph_tz)

    formatted_time = now.strftime("%I:%M %p")  # 12-hour time string
    formatted_date = now.strftime("%m/%d/%Y")  # MM/DD/YYYY
    hour_only = now.strftime("%I")  # 12-hour format without AM/PM
    day_only = str(now.day)         # Day of month (1–31)

    predicted = float(predicted_yields[i])
    db.collection("predictedYield").add({
        "temperature": original["temperature"],
        "humidity": original["humidity"],
        "soilMoisture": original["localMoisture"],
        "timestamp": now.isoformat(),
        "date": formatted_date,
        "time": formatted_time,
        "day": day_only,
        "hour": hour_only,
        "predicted_yield": round(predicted, 2),
        "source": "predicted"
    })

print("✅ Yield prediction saved with date, time, day, and hour.")
