import streamlit as st
from modules.data_cleaning import load_and_clean_data
from modules.preprocessing import preprocess_features
from modules.model import train_model, evaluate_model, save_model
from modules.new_prediction import predict_new_patient

# Load dataset
df = load_and_clean_data("data/dementia_data.csv")
target_col = "dementia"

# Preprocessing
X_train_scaled, X_test_scaled, y_train, y_test, scaler, columns = preprocess_features(df, target_col)

# Train model
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
save_model(model, scaler, columns)

st.title("Dementia Risk Prediction Hackathon App")

st.subheader("Enter patient details")

patient = {
    "age": st.number_input("Age", 50, 100, 70),
    "education_years": st.number_input("Years of Education", 0, 20, 12),
    "sex": st.selectbox("Sex", ["M", "F"]),
    "marital_status": st.selectbox("Marital Status", ["Married", "Single", "Widowed", "Divorced"]),
    "social_activity": st.number_input("Social Activity (1-6)", 1, 6, 3),
    "exercise_hours": st.number_input("Weekly Exercise Hours", 0, 10, 3),
    "smoking_status": st.selectbox("Smoking (0=No,1=Yes)", [0,1]),
    "alcohol_units": st.number_input("Weekly Alcohol Units", 0, 20, 1)
}

if st.button("Predict Risk"):
    result = predict_new_patient(patient)
    if result == 1:
        st.error("Patient is at risk of dementia")
    else:
        st.success("Patient is NOT at risk")