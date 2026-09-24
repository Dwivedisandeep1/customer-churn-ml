from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Tuned XGBoost"
    assert data["model_version"] == "1.0"


def test_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["churn_probability"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["threshold"] == 0.6
    assert data["model"] == "Tuned XGBoost"
    assert data["model_version"] == "1.0"


def test_invalid_customer_data():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
