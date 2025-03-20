from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Load saved models and encoders
clf = joblib.load(open('./model.pkl', 'rb'))
ohe = joblib.load(open('./ohe.pkl', 'rb'))
scaler = joblib.load(open('./scaler.pkl', 'rb'))
label = joblib.load(open('./label.pkl', 'rb'))

# Fix OneHotEncoder handle_unknown issue
ohe.handle_unknown = 'ignore'

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to Job Role Prediction API!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        required_fields = [
            'Acedamic percentage in Operating Systems',
            'percentage in Algorithms',
            'Percentage in Programming Concepts',
            'Percentage in Software Engineering',
            'Percentage in Computer Networks',
            'Percentage in Electronics Subjects',
            'Percentage in Computer Architecture',
            'Percentage in Mathematics',
            'Percentage in Communication skills',
            'Logical quotient rating',
            'hackathons',
            'coding skills rating',
            'public speaking points',
            'self-learning capability?',
            'Extra-courses did',
            'certifications',
            'Interested subjects',
            'interested career area ',
            'Job/Higher Studies?',
            'Type of company want to settle in?',
            'worked in teams ever?'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Convert input data into DataFrame
        user_input_df = pd.DataFrame([data])

        # One-Hot Encode & Scale the input
        user_encoded = ohe.transform(user_input_df)
        user_scaled = scaler.transform(user_encoded)

        # Predict job role
        prediction = clf.predict(user_scaled)
        predicted_role = label.inverse_transform(prediction)

        return jsonify({'success': True, 'Suggested Job Role': predicted_role[0]})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
