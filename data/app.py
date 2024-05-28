from flask import Flask, send_from_directory, request, jsonify
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__, static_folder='../static', template_folder='../')

# Load the model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return send_from_directory('../', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form, excluding 'weight' and 'height'
        data = [float(request.form.get(key)) for key in request.form.keys() if key not in ['weight', 'height']]
        print(f"Received data: {data}")  # Debug information

        # Ensure the length of data matches the expected number of features
        expected_num_features = 18  # Adjust this based on the actual number of features
        if len(data) != expected_num_features:
            raise ValueError(f"Expected {expected_num_features} features, but got {len(data)}")

        data = np.array(data).reshape(1, -1)

        # Scale the data
        data = scaler.transform(data)
        print(f"Scaled data: {data}")  # Debug information

        # Make prediction
        probability = model.predict_proba(data)[0][1]  # Get the probability of the positive class
        prediction_percentage = probability * 100
        formatted_prediction = f"{prediction_percentage:.4f}%"
        print(f"Prediction Probability: {formatted_prediction}")  # Debug information

        return jsonify({'prediction': formatted_prediction})
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())  # Print the traceback for detailed error information
        return jsonify({'prediction': f"Error occurred during prediction: {e}"})

if __name__ == '__main__':
    app.run(debug=True)
