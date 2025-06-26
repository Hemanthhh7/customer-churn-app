import streamlit as st
import pandas as pd
import pickle

# Load model + scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("📊 Customer Churn Prediction App")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Prepare input
data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

df = pd.DataFrame([data])
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
    df[["tenure", "MonthlyCharges", "TotalCharges"]])

# Predict
if st.button("Predict Churn"):
    result = model.predict(df)
    if result[0] == 1:
        st.error("⚠️ Customer likely to churn.")
    else:
        st.success("✅ Customer likely to stay.")
