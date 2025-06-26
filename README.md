# 📊 Customer Churn Prediction App

This project predicts customer churn based on Telco customer data using a Random Forest model in a pipeline (with StandardScaler).  

## 🚀 Project Components
- **Model training:** Colab notebook (RandomForestClassifier + StandardScaler in Pipeline)
- **Serving app:** Streamlit app (app.py)
- **Deployment:** Streamlit Cloud

## 🗂 Files
- `app.py` — Streamlit frontend for predictions
- `pipe.pkl` — Trained model (Colab generated)
- `requirements.txt` — Dependencies list for Streamlit Cloud
- `demo.mp4` — Demo video of the app (optional)

## 🛠 How to Run
### Local
```bash
streamlit run app.py
