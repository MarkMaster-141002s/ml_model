from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('yield_model.pkl')

@app.route('/')
def home():
    return "✅ Yield Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input features
        temp = float(data['temperature'])
        hum = float(data['humidity'])
        moist = float(data['soil_moisture'])

        # Predict yield
        input_array = np.array([[temp, hum, moist]])
        predicted_yield = model.predict(input_array)[0]

        return jsonify({
            'predicted_yield': round(predicted_yield, 2),
            'input': {
                'temperature': temp,
                'humidity': hum,
                'soil_moisture': moist
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
