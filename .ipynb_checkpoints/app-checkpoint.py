
import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("best_churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("📊 Customer Churn Predictor")

user_input = {}

st.sidebar.header("Customer Info")

# Example of binary inputs
user_input['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1])
user_input['tenure'] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
user_input['MonthlyCharges'] = st.sidebar.slider("Monthly Charges", 0, 150, 70)
user_input['TotalCharges'] = st.sidebar.slider("Total Charges", 0, 10000, 2000)

# Example of one-hot multicategorical
user_input['Contract_Two year'] = st.sidebar.selectbox("Contract: Two Year", [0, 1])
user_input['InternetService_Fiber optic'] = st.sidebar.selectbox("Fiber Optic Internet", [0, 1])
user_input['PaymentMethod_Electronic check'] = st.sidebar.selectbox("Electronic Check", [0, 1])

# Fill null columns 
input_df = pd.DataFrame([user_input])
for col in model_columns:
    if col not in input_df:
        input_df[col] = 0

# Reorder columns
input_df = input_df[model_columns]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("🔍 Prediction Result")
    st.write("Churn:" if prediction == 1 else "Not Churn")
    st.write(f"Probability of churn: **{probability:.2f}**")
