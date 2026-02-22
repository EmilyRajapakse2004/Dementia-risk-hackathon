import pandas as pd
import joblib


def predict_new_patient(patient_data, model_path="model.pkl", scaler_path="scaler.pkl",
                        columns_path="model_columns.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_columns = joblib.load(columns_path)

    # Convert input dict to dataframe
    df = pd.DataFrame([patient_data])

    # One-hot encode categorical features
    df = pd.get_dummies(df)

    # Align with training columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)
    return prediction[0]