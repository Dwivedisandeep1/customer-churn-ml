from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.feature_engineering import create_features


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.joblib"


# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Telecom customer churn prediction service",
    version="1.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_customer(customer_data: dict):

    # Convert input to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Apply production feature engineering
    customer_features = create_features(customer_df)

    # customerID is not required by the model
    if "customerID" in customer_features.columns:
        customer_features = customer_features.drop(
            columns=["customerID"]
        )

    # Generate churn probability
    probability = model.predict_proba(
        customer_features
    )[0, 1]

    # Apply optimized threshold
    prediction = int(probability >= threshold)

    # Business risk categories
    if probability >= 0.60:
        risk_level = "HIGH"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": float(threshold),
        "model": "Tuned XGBoost",
        "model_version": "1.0"
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Tuned XGBoost",
        "model_version": "1.0"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    return predict_customer(
        customer.model_dump()
    )