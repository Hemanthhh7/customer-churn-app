import streamlit as st
import pandas as pd
import pickle

# Load pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title("📊 Customer Churn Prediction App")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0)

phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Build input dataframe matching pipeline feature names
input_df = pd.DataFrame([{
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == 'Yes' else 0,
    'MultipleLines': 1 if multiple_lines == 'Yes' else (0 if multiple_lines == 'No' else -1),
    'InternetService': 0 if internet_service == 'No' else (1 if internet_service == 'DSL' else 2),
    'OnlineSecurity': 1 if online_security == 'Yes' else (0 if online_security == 'No' else -1),
    'OnlineBackup': 1 if online_backup == 'Yes' else (0 if online_backup == 'No' else -1),
    'DeviceProtection': 1 if device_protection == 'Yes' else (0 if device_protection == 'No' else -1),
    'TechSupport': 1 if tech_support == 'Yes' else (0 if tech_support == 'No' else -1),
    'StreamingTV': 1 if streaming_tv == 'Yes' else (0 if streaming_tv == 'No' else -1),
    'StreamingMovies': 1 if streaming_movies == 'Yes' else (0 if streaming_movies == 'No' else -1),
    'Contract': 0 if contract == 'Month-to-month' else (1 if contract == 'One year' else 2),
    'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
    'PaymentMethod': 0 if payment_method == 'Electronic check' else
                     (1 if payment_method == 'Mailed check' else
                     (2 if payment_method == 'Bank transfer (automatic)' else 3)),
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}])

# Ensure order matches what scaler expects
input_df = input_df[list(pipe.named_steps['scaler'].feature_names_in_)]

# Predict
if st.button("Predict Churn"):
    result = pipe.predict(input_df)
    if result[0] == 1:
        st.error("⚠️ Customer likely to churn.")
    else:
        st.success("✅ Customer likely to stay.")


