from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def save_model(model, scaler, columns, model_path="model.pkl", scaler_path="scaler.pkl", columns_path="model_columns.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(columns, columns_path)