import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("../models/best_model.pkl")

st.title("💰 SmartPremium: Insurance Cost Predictor")

# Inputs
age = st.number_input("Age", 18, 100)
income = st.number_input("Annual Income", 10000, 10000000)
health_score = st.slider("Health Score", 0, 100, 50)
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
smoking = st.selectbox("Smoking Status", ["Yes", "No"])

input_data = pd.DataFrame({
    "Age": [age],
    "Annual Income": [income],
    "Health Score": [health_score],
    "Policy Type": [policy_type],
    "Location": [location],
    "Smoking Status": [smoking]
})

if st.button("Predict Premium"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Premium: ₹ {prediction[0]:,.2f}")
