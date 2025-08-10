from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("model.save")
scaler = joblib.load("transform.save")
feature_names = joblib.load("features.save")

# ------------------- ROUTES -------------------

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Manual input form
@app.route('/manual')
def manual():
    return render_template('Manual_predict.html', feature_names=feature_names)

# Sensor input form
@app.route('/sensor')
def sensor():
    return render_template('Sensor_predict.html', feature_names=feature_names)

# Manual prediction processing
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        features = [float(request.form.get(name)) for name in feature_names]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        return render_template('Manual_predict.html', feature_names=feature_names,
                               prediction_text=f"Permanent Magnet surface temperature: {prediction:.4f}")
    except Exception as e:
        return render_template('Manual_predict.html', feature_names=feature_names,
                               prediction_text=f"Error: {e}")

# Sensor prediction processing
@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    try:
        features = [float(request.form.get(name)) for name in feature_names]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        return render_template('Sensor_predict.html', feature_names=feature_names,
                               prediction_text=f"Sensor Prediction: {prediction:.4f}")
    except Exception as e:
        return render_template('Sensor_predict.html', feature_names=feature_names,
                               prediction_text=f"Error: {e}")

# ------------------- RUN APP -------------------
if __name__ == '__main__':
    app.run(debug=True)
