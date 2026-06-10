import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_explainer = joblib.load(
    os.path.join(BASE_DIR, "..", "model", "shap_explainer.pkl")
)

_feature_columns = joblib.load(
    os.path.join(BASE_DIR, "..", "model", "feature_columns.pkl")
)

FEATURE_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "charges_per_tenure": "Charges per tenure month",
    "num_services": "Number of services",
    "is_high_value_at_risk": "High-value at-risk flag",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "PaperlessBilling": "Paperless billing",
    "Contract_Month-to-month": "Contract: month-to-month",
    "Contract_One year": "Contract: one year",
    "Contract_Two year": "Contract: two year",
    "PaymentMethod_Electronic check": "Payment: electronic check",
    "PaymentMethod_Mailed check": "Payment: mailed check",
    "PaymentMethod_Bank transfer (automatic)": "Payment: bank transfer",
    "PaymentMethod_Credit card (automatic)": "Payment: credit card",
    "InternetService_Fiber optic": "Internet: fiber optic",
    "InternetService_DSL": "Internet: DSL",
    "InternetService_No": "No internet service",
    "TechSupport": "Tech support",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "gender_Female": "Gender: female",
    "gender_Male": "Gender: male",
}

# SAFE SHAP EXTRACTION

def _get_sv(shap_vals):
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = np.array(shap_vals)

    if shap_vals.ndim == 3:
        shap_vals = shap_vals[0]

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    return shap_vals


def _safe_float(x):
    return float(np.array(x).reshape(-1)[0])
#  STORY ENGINE (NEW - THIS FIXES YOUR UI QUALITY)

def explain_feature(feature, value, direction):

    stories = {
        "MonthlyCharges": "High monthly charges often reduce perceived value, increasing churn risk.",
        "tenure": "Short tenure means the customer is still early in their journey and more likely to leave.",
        "Contract_Month-to-month": "Month-to-month contracts allow easy switching with no long-term lock-in.",
        "PaymentMethod_Electronic check": "Electronic check users historically show higher churn behavior.",
        "TechSupport": "Lack of tech support reduces customer satisfaction and increases churn risk.",
        "SeniorCitizen": "Senior customers may be more sensitive to pricing and service experience.",
        "Partner": "Customers with partners tend to be more stable and less likely to churn.",
        "InternetService_Fiber optic": "Fiber users often churn due to higher pricing expectations."
    }

    base = stories.get(feature, "This factor influences churn based on historical customer behavior.")

    return ("📈 " if direction == "increases" else "📉 ") + base

# MAIN SHAP FUNCTION

def get_shap_explanation(df_row_scaled: pd.DataFrame, top_n: int = 5):

    shap_vals = _explainer.shap_values(df_row_scaled)
    sv = _get_sv(shap_vals)

    cols = _feature_columns
    abs_sv = np.abs(sv)

    top_idx = np.argsort(abs_sv)[::-1][:top_n]
    top_idx = [int(i) for i in np.ravel(top_idx)]

    result = []

    for i in top_idx:
        val = _safe_float(sv[i])
        fname = cols[i]
        direction = "increases" if val > 0 else "decreases"

        result.append({
            "feature_label": FEATURE_LABELS.get(fname, fname),
            "shap_value": round(val, 4),
            "direction": direction,
            "direction_color": "#F85149" if val > 0 else "#3FB950",
            "magnitude": round(abs(val), 4),
            "bar_pct": 0,
            "story": explain_feature(fname, val, direction)
        })

    max_mag = max(r["magnitude"] for r in result) if result else 1

    for r in result:
        r["bar_pct"] = round((r["magnitude"] / max_mag) * 100, 1)

    return result

# WATERFALL CHART (UNCHANGED BUT SAFE)
def make_waterfall_chart(df_row_scaled: pd.DataFrame) -> str:

    shap_vals = _explainer.shap_values(df_row_scaled)
    sv = _get_sv(shap_vals)

    cols = _feature_columns
    abs_sv = np.abs(sv)

    top_idx = np.argsort(abs_sv)[::-1][:8]
    top_idx = [int(i) for i in np.ravel(top_idx)]

    top_vals = [sv[i] for i in top_idx]
    top_labels = [FEATURE_LABELS.get(cols[i], cols[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#161B22")
    ax.set_facecolor("#0D1117")

    colors = ["#F85149" if v > 0 else "#3FB950" for v in top_vals]
    y_pos = range(len(top_vals))[::-1]

    ax.barh(
        list(y_pos),
        list(reversed(top_vals)),
        color=list(reversed(colors)),
        edgecolor="#0D1117",
        linewidth=0.8,
        height=0.65,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(reversed(top_labels)), fontsize=9, color="#8B949E")

    ax.set_xlabel("SHAP value (+ increases churn risk, − decreases)", color="#8B949E")
    ax.set_title("Model Explanation — Feature Contributions", color="#E6EDF3")

    ax.axvline(0, color="#2A3240", linewidth=1.2)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3240")

    ax.tick_params(colors="#8B949E")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")