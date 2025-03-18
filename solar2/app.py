from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("final_trained_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Solar Power Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Convert to numpy array
        
        # Preprocess input using the saved scaler
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Return response
        return jsonify({"predicted_power_kw": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
