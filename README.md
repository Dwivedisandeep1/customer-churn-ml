# Telecom Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in the telecommunications industry.

The project covers the complete ML lifecycle from exploratory data analysis and feature engineering to model training, explainability, model persistence, automated testing, and deployment through a FastAPI REST API.

---

## Project Overview

Customer churn is a major business problem for telecom companies. Identifying customers who are likely to leave allows businesses to prioritize retention strategies.

This project builds a binary classification model that predicts whether a customer is likely to churn.

### Prediction

- `0` → Customer is predicted not to churn
- `1` → Customer is predicted to churn

The production API also assigns a business risk level:

- `LOW` → probability < 0.30
- `MEDIUM` → probability 0.30–0.59
- `HIGH` → probability ≥ 0.60

The risk bands are application-level business rules and are separate from the model's probability output.

---

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer demographic information, account information, subscribed services, billing information, and churn status.

Key variables include:

- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Internet service
- Online security
- Technical support
- Payment method
- Paperless billing
- Churn

---

## Machine Learning Workflow

```text
Raw Data
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ↓
Model Comparison
    ↓
XGBoost Hyperparameter Tuning
    ↓
Threshold Optimization
    ↓
Final Test Evaluation
    ↓
SHAP Explainability
    ↓
Model Serialization
    ↓
FastAPI Inference API
    ↓
Automated API Tests