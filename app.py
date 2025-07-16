from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# ✅ Load the trained TensorFlow model
model = tf.keras.models.load_model("yield_model.keras")

@app.route('/')
def home():
    return "✅ Yield Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ Read input
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        soil_moisture = data['soil_moisture']

        # If soil_moisture is a list (multiple sensors), average it
        if isinstance(soil_moisture, list):
            soil_moisture = sum(soil_moisture) / len(soil_moisture)

        # ✅ Prepare input for prediction
        input_data = np.array([[temperature, humidity, soil_moisture]])

        # 🔮 Predict
        predicted_yield = model.predict(input_data)[0][0]

        return jsonify({
            "predicted_yield": round(float(predicted_yield), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
