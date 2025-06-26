# Prepare input dict matching all features used in training
data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Create DataFrame with correct columns
input_df = pd.DataFrame([data])

# Scale entire input_df if that's what you did during training
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Churn"):
    result = model.predict(input_scaled)
    if result[0] == 1:
        st.error("⚠️ Customer likely to churn.")
    else:
        st.success("✅ Customer likely to stay.")
