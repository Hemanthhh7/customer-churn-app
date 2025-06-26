import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("📊 Customer Churn Prediction App")

# First ask for user input
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Only after inputs, build the data
data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([data])

# Apply scaler on correct columns
input_scaled = scaler.transform(input_df)

# Predict on button click
if st.button("Predict Churn"):
    result = model.predict(input_scaled)
    if result[0] == 1:
        st.error("⚠️ Customer likely to churn.")
    else:
        st.success("✅ Customer likely to stay.")

