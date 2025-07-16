import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ✅ Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ✅ Load Data from CSV
print("📄 Loading data from yield_data.csv...")
try:
    df = pd.read_csv("yield_data.csv")
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    exit()

# ✅ Normalize column names
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={
    "soilmoisture": "soil_moisture",
    "localmoisture": "soil_moisture"  # Optional alias
}, inplace=True)

# ✅ Drop rows with missing values
df = df.dropna()

# ✅ Validate column presence
required_columns = ["temperature", "humidity", "soil_moisture", "yield"]
if not all(col in df.columns for col in required_columns):
    print(f"❌ CSV is missing one of the required columns: {required_columns}")
    print(f"📌 Found columns: {list(df.columns)}")
    exit()

print(f"✅ Loaded {len(df)} valid records.\n")
print(df.head())

# ✅ Check for minimum data
if len(df) < 5:
    print("❌ Not enough data to train the model. Add more rows to 'yield_data.csv'.")
    exit()

# ✅ Split Features and Labels
X_unscaled = df[["temperature", "humidity", "soil_moisture"]]
y = df["yield"]

# ✅ Normalize Inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ✅ Define TensorFlow Model
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ Train Model
print("🏋️ Training model...")
history = tf_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# ✅ Save TensorFlow Model
tf_model.save("yield_model.keras")
print("✅ Model saved as 'yield_model.keras'")

# ✅ Predict on Test Set
print("🔮 Predicting yields...")
y_pred = tf_model.predict(X_test).flatten()

# ✅ Save Predictions to Firestore
pred_collection = db.collection("predictedYield")

print("📤 Saving predictions to Firestore...")
# Use unscaled original values
X_test_original = X_unscaled.iloc[y_test.index]
for i, idx in enumerate(y_test.index):
    prediction_data = {
        "timestamp": datetime.utcnow(),
        "temperature": float(X_test_original.iloc[i]["temperature"]),
        "humidity": float(X_test_original.iloc[i]["humidity"]),
        "soil_moisture": float(X_test_original.iloc[i]["soil_moisture"]),
        "actual_yield": float(y_test.iloc[i]),
        "predicted_yield": float(y_pred[i])
    }
    pred_collection.add(prediction_data)

print("✅ Predictions saved to Firestore collection 'predictedYield'")
