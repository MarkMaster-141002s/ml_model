import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import pytz
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define Manila timezone
manila_tz = pytz.timezone("Asia/Manila")

# Step 1: Fetch sensor data from Firestore
docs = db.collection("dataCollectionSensor").stream()

data = []
for doc in docs:
    d = doc.to_dict()
    try:
        raw_ts = d.get('timestamp')

        # Convert Firestore timestamp to Manila time
        if hasattr(raw_ts, 'astimezone'):
            manila_time = raw_ts.astimezone(manila_tz)
        else:
            raw_ts = pd.to_datetime(raw_ts)
            manila_time = raw_ts.tz_localize('UTC').astimezone(manila_tz)

        data.append({
            'temperature': float(d.get('temperature')),
            'humidity': float(d.get('humidity')),
            'soil_moisture': float(d.get('localMoisture')),
            'timestamp': manila_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print("Skipped document due to error:", e)
        continue

df = pd.DataFrame(data)
df.dropna(inplace=True)

# Step 2: Simulate realistic yield based on Year 1–3 logic
def estimate_yield(temp, hum, moist):
    if moist < 30:
        return 40 + 0.1 * temp + 0.05 * hum
    elif moist < 50:
        return 80 + 0.1 * temp + 0.05 * hum
    elif moist < 70:
        return 120 + 0.1 * temp + 0.05 * hum
    else:
        return 280 + 0.05 * temp + 0.02 * hum

df['yield'] = df.apply(lambda row: estimate_yield(
    row['temperature'],
    row['humidity'],
    row['soil_moisture']
), axis=1)

# Step 3: Train regression model
X = df[['temperature', 'humidity', 'soil_moisture']]
y = df['yield']

model = LinearRegression()
model.fit(X, y)

# Save trained model
joblib.dump(model, 'yield_model.pkl')

# Step 4: Predict
df['predicted_yield'] = model.predict(X)

# Step 5: Save predictions to Firestore
for _, row in df.iterrows():
    db.collection('yieldPredictions').add({
        'temperature': row['temperature'],
        'humidity': row['humidity'],
        'soil_moisture': row['soil_moisture'],
        'predicted_yield': row['predicted_yield'],
        'timestamp': row['timestamp']  # in Manila time (formatted)
    })

print("✅ Model trained and predictions saved using Manila time.")
