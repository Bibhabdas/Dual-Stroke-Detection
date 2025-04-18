from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import joblib  # For loading the trained model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'nootbook/uploads/'  # Folder to store uploaded images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/data',methods=['GET'])
def get_data():
    return jsonify({"message":"hello from python"})

# Load the trained model
ridge_model = joblib.load("models/heart_stroke_model.pkl")  # Ensure the trained model is saved
standard_scaler = joblib.load("models/scaler.pkl")  # Load the scaler if used

# Load Brain Stroke Prediction Model
brain_model = load_model("models/model.h5")

# Function to preprocess brain stroke images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (240,240))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions
    return image

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/success')
def success_page():
    return render_template('success.html')


@app.route('/phone')
def phone_page():
    return render_template('phone.html')


@app.route('/email')
def email_page():
    return render_template('email.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/heart_predict')
def heart_predict_page():
    return render_template('heart_predict.html')

@app.route('/cardio')
def doctor__cardio_page():
    return render_template('cardio.html')

@app.route('/dibya_cardio')
def dibya__cardio_page():
    return render_template('dibya_cardio.html')

@app.route('/pathak_cardio')
def pathak__cardio_page():
    return render_template('pathak_cardio.html')

@app.route('/ranga_cardio')
def ranga__cardio_page():
    return render_template('ranga_cardio.html')

@app.route('/sarita_cardio')
def sarita__cardio_page():
    return render_template('sarita_cardio.html')

@app.route('/vignesh_cardio')
def vignesh__cardio_page():
    return render_template('vignesh_cardio.html')

@app.route('/neuro')
def doctor_neuro_page():
    return render_template('neuro.html')

@app.route('/abidha_neuro')
def abidha_neuro_page():
    return render_template('abidha_neuro.html')

@app.route('/avinash_neuro')
def avinash_neuro_page():
    return render_template('avinash_neuro.html')

@app.route('/rashmi_neuro')
def rashmi_neuro_page():
    return render_template('rashmi_neuro.html')

@app.route('/siva_neuro')
def siva_neuro_page():
    return render_template('siva_neuro.html')

@app.route('/sushant_neuro')
def sushant_neuro_page():
    return render_template('sushant_neuro.html')


'''@app.route('/')
def home():
    return render_template('heart_predict.html')'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [
            request.form['gender'],
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            request.form['ever_married'],
            request.form['work_type'],
            request.form['residence_type'],
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            request.form['smoking_status']
        ]

        # Convert categorical values to numerical
        mapping = {
            "Male": 0, "Female": 1, "Other": 2,
            "Yes": 1, "No": 0,
            "Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4,
            "Urban": 0, "Rural": 1,
            "never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3
        }
        
        processed_data = [mapping.get(val, val) for val in data]
        processed_data = np.array(processed_data).reshape(1, -1)
        processed_data = standard_scaler.transform(processed_data)

        # Make prediction
        prediction = ridge_model.predict(processed_data)[0]
        result_text = "High" if prediction == 1 else "No"

        # Redirect to heart_result.html with prediction result
        return render_template('heart_result.html', result=result_text)
    
    except Exception as e:
        return render_template('heart_result.html', result=f"Error: {str(e)}")

@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/brain_predict', methods=['POST'])
def brain_predict():
    try:
        uploaded_file = request.files['document']
        if uploaded_file.filename == '':
            return "No file selected"

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        processed_image = preprocess_image(file_path)
        prediction = brain_model.predict(processed_image)[0]
        predicted_class = np.argmax(prediction)  # Get class index
        
        # Class names based on your dataset (Adjust if needed)
        class_labels = ["High", "No"]
        result_text = f"{class_labels[predicted_class]}"

        return render_template('brain_result.html', result=result_text)
    except Exception as e:
        return render_template('brain_result.html', result=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
