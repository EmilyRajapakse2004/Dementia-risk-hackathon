import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../data/processed/dementia_risk_model.pkl")

st.set_page_config(page_title="Dementia Risk Predictor", layout="centered")

st.title("🧠 Dementia Risk Prediction")
st.write("Enter patient details to assess dementia risk.")

st.divider()

# ---- User Inputs ----
age = st.number_input("Age", min_value=40, max_value=120, value=65)

education_years = st.number_input(
    "Years of Education", min_value=0, max_value=25, value=12
)

gender = st.selectbox("Gender", ["Male", "Female"])

marital_status = st.selectbox(
    "Marital Status", ["Married", "Single", "Divorced", "Widowed"]
)

# ---- Encoding (MUST match training) ----
gender_encoded = 1 if gender == "Male" else 0

marital_map = {
    "Married": 0,
    "Single": 1,
    "Divorced": 2,
    "Widowed": 3
}

marital_encoded = marital_map[marital_status]

# ---- Create Input DataFrame ----
input_data = pd.DataFrame([{
    "age": age,
    "education_years": education_years,
    "gender": gender_encoded,
    "marital_status": marital_encoded
}])

st.divider()

# ---- Prediction ----
if st.button("🔍 Predict Dementia Risk"):
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    st.subheader("Prediction Result")

    if risk_percent >= 50:
        st.error(f"⚠️ High Dementia Risk: {risk_percent:.2f}%")
    else:
        st.success(f"✅ Low Dementia Risk: {risk_percent:.2f}%")

    st.caption("This is a probabilistic prediction, not a medical diagnosis.")
