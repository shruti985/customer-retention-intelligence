# ============================================================
# predictor.py  (v2 — production upgrade)
# Changes from v1:
#   • Loads tuned threshold from model/threshold.pkl
#   • Returns df_row_scaled for direct SHAP consumption
#   • Returns raw values dict for explainer + segmentation
# ============================================================

import joblib
import numpy as np
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "model")

model            = joblib.load(os.path.join(BASE, "best_model.pkl"))
scaler           = joblib.load(os.path.join(BASE, "scaler.pkl"))
feature_columns  = joblib.load(os.path.join(BASE, "feature_columns.pkl"))
scale_cols       = joblib.load(os.path.join(BASE, "scale_cols.pkl"))

_threshold_path = os.path.join(BASE, "threshold.pkl")
THRESHOLD = joblib.load(_threshold_path) if os.path.exists(_threshold_path) else 0.5
print(f"[predictor] Using decision threshold: {THRESHOLD:.2f}")


def get_risk_tier(prob: float) -> tuple:
    if prob >= 0.65:
        return "HIGH",   "#F85149"
    elif prob >= 0.35:
        return "MEDIUM", "#D29922"
    else:
        return "LOW",    "#3FB950"


def build_row(form_data: dict) -> tuple:
    fd = form_data

    tenure          = float(fd.get("tenure", 0))
    monthly_charges = float(fd.get("MonthlyCharges", 0))
    total_charges   = float(fd.get("TotalCharges", 0))

    def yn(key):
        return 1 if fd.get(key, "No") == "Yes" else 0

    binary = {
        "SeniorCitizen":    int(fd.get("SeniorCitizen", 0)),
        "Partner":          yn("Partner"),
        "Dependents":       yn("Dependents"),
        "PhoneService":     yn("PhoneService"),
        "MultipleLines":    yn("MultipleLines"),
        "OnlineSecurity":   yn("OnlineSecurity"),
        "OnlineBackup":     yn("OnlineBackup"),
        "DeviceProtection": yn("DeviceProtection"),
        "TechSupport":      yn("TechSupport"),
        "StreamingTV":      yn("StreamingTV"),
        "StreamingMovies":  yn("StreamingMovies"),
        "PaperlessBilling": yn("PaperlessBilling"),
    }

    contract = fd.get("Contract", "Month-to-month")
    payment  = fd.get("PaymentMethod", "Electronic check")
    internet = fd.get("InternetService", "Fiber optic")
    gender   = fd.get("gender", "Male")

    ohe = {
        "Contract_Month-to-month":                1 if contract == "Month-to-month" else 0,
        "Contract_One year":                      1 if contract == "One year"        else 0,
        "Contract_Two year":                      1 if contract == "Two year"        else 0,
        "PaymentMethod_Bank transfer (automatic)":1 if payment == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":  1 if payment == "Credit card (automatic)"   else 0,
        "PaymentMethod_Electronic check":         1 if payment == "Electronic check"          else 0,
        "PaymentMethod_Mailed check":             1 if payment == "Mailed check"              else 0,
        "InternetService_DSL":         1 if internet == "DSL"         else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No":          1 if internet == "No"          else 0,
        "gender_Female": 1 if gender == "Female" else 0,
        "gender_Male":   1 if gender == "Male"   else 0,
    }

    service_vals = [
        binary["PhoneService"], binary["MultipleLines"],
        binary["OnlineSecurity"], binary["OnlineBackup"],
        binary["DeviceProtection"], binary["TechSupport"],
        binary["StreamingTV"], binary["StreamingMovies"]
    ]

    engineered = {
        "charges_per_tenure":    monthly_charges / (tenure + 1),
        "num_services":          sum(service_vals),
        "is_high_value_at_risk": 1 if (monthly_charges > 70 and tenure < 12) else 0,
    }

    row = {
        "tenure":         tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges":   total_charges,
        **binary, **ohe, **engineered,
    }

    df_row = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_columns]

    df_row_scaled = df_row.copy()
    df_row_scaled[scale_cols] = scaler.transform(df_row_scaled[scale_cols])

    raw = {
        "tenure":              tenure,
        "MonthlyCharges":      monthly_charges,
        "Contract":            contract,
        "PaymentMethod":       payment,
        "InternetService":     internet,
        "TechSupport":         binary["TechSupport"],
        "OnlineSecurity":      binary["OnlineSecurity"],
        "num_services":        engineered["num_services"],
        "SeniorCitizen":       binary["SeniorCitizen"],
        "PaperlessBilling":    binary["PaperlessBilling"],
        "Partner":             binary["Partner"],
        "Dependents":          binary["Dependents"],
        "is_high_value_at_risk": engineered["is_high_value_at_risk"],
    }

    return df_row_scaled, raw


def predict(form_data: dict) -> dict:
    df_row_scaled, raw = build_row(form_data)
    prob             = float(model.predict_proba(df_row_scaled)[0][1])
    risk, risk_color = get_risk_tier(prob)
    return {
        "probability":   round(prob * 100, 1),
        "risk":          risk,
        "risk_color":    risk_color,
        "df_row_scaled": df_row_scaled,
        "_raw":          raw,
    }