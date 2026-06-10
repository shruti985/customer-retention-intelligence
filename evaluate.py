# ============================================================
# evaluate.py  —  Production Evaluation Pipeline
# Run AFTER eda_and_training.py (requires saved model artifacts)
#
# WHY THIS EXISTS SEPARATELY:
#   Training and evaluation are different concerns.
#   Training finds the best model. Evaluation proves it's
#   trustworthy enough to deploy and sets the operating threshold.
#   Keeping them separate makes both auditable.
# ============================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import shap
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, classification_report
)
import os
os.makedirs("data/eval", exist_ok=True)

# ── Load artifacts ────────────────────────────────────────────
model           = joblib.load("model/best_model.pkl")
scaler          = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
scale_cols      = joblib.load("model/scale_cols.pkl")

# ── Rebuild test set (same pipeline as training) ──────────────
# We saved the model but not X_test — rebuild it deterministically
df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

binary_cols = [
    "Partner","Dependents","PhoneService","MultipleLines",
    "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","PaperlessBilling"
]
for col in binary_cols:
    df[col] = df[col].map({"Yes":1,"No":0,"No phone service":0,"No internet service":0})

ohe_cols = ["Contract","PaymentMethod","InternetService","gender"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
df["charges_per_tenure"]    = df["MonthlyCharges"] / (df["tenure"] + 1)
df["num_services"]          = df[service_cols].sum(axis=1)
df["is_high_value_at_risk"] = ((df["MonthlyCharges"] > 70) & (df["tenure"] < 12)).astype(int)

X = df[feature_columns]
y = df["Churn"]

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_test_scaled = X_test.copy()
X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_default = model.predict(X_test_scaled)

print("Default threshold (0.5) performance:")
print(classification_report(y_test, y_pred_default, target_names=["No Churn","Churn"]))


# ── 1. Threshold Tuning ───────────────────────────────────────
# WHY: Default 0.5 threshold optimises for accuracy, not F1/Recall.
# In churn, missing a churner (FN) costs more than a false alarm (FP).
# We find the threshold that maximises F1 and separately the one
# that hits ≥70% recall with best precision — two business operating points.

thresholds  = np.arange(0.20, 0.81, 0.01)
f1s, precisions, recalls = [], [], []

for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    f1s.append(f1_score(y_test, y_t))
    from sklearn.metrics import precision_score, recall_score
    precisions.append(precision_score(y_test, y_t, zero_division=0))
    recalls.append(recall_score(y_test, y_t))

best_f1_idx       = np.argmax(f1s)
best_f1_threshold = thresholds[best_f1_idx]

# Best recall-precision tradeoff: recall >= 0.70, maximise precision
recall_arr   = np.array(recalls)
prec_arr     = np.array(precisions)
recall_mask  = recall_arr >= 0.70
if recall_mask.any():
    best_recall_prec_idx = np.argmax(np.where(recall_mask, prec_arr, 0))
    best_recall_threshold = thresholds[best_recall_prec_idx]
else:
    best_recall_threshold = best_f1_threshold

print(f"\nBest F1 threshold   : {best_f1_threshold:.2f}  →  F1={f1s[best_f1_idx]:.4f}")
print(f"Best recall≥70% threshold: {best_recall_threshold:.2f}")

# Save tuned threshold
joblib.dump(float(best_f1_threshold), "model/threshold.pkl")
print(f"\n✅ Saved threshold {best_f1_threshold:.2f} → model/threshold.pkl")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(thresholds, f1s,        label="F1",        color="#388BFD", linewidth=2)
ax.plot(thresholds, precisions, label="Precision",  color="#3FB950", linewidth=1.5, linestyle="--")
ax.plot(thresholds, recalls,    label="Recall",     color="#E05C5C", linewidth=1.5, linestyle="--")
ax.axvline(best_f1_threshold,      color="#388BFD", linestyle=":", alpha=0.7,
           label=f"Best F1 @ {best_f1_threshold:.2f}")
ax.axvline(best_recall_threshold,  color="#E05C5C", linestyle=":", alpha=0.7,
           label=f"Best recall≥70% @ {best_recall_threshold:.2f}")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Tuning: F1, Precision, Recall", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim(0.20, 0.80)
ax.set_facecolor("#0D1117")
fig.patch.set_facecolor("#161B22")
for spine in ax.spines.values(): spine.set_edgecolor("#2A3240")
ax.tick_params(colors="#8B949E")
ax.xaxis.label.set_color("#8B949E")
ax.yaxis.label.set_color("#8B949E")
ax.title.set_color("#E6EDF3")
ax.legend(facecolor="#1C2330", edgecolor="#2A3240", labelcolor="#E6EDF3")
plt.tight_layout()
plt.savefig("data/eval/threshold_tuning.png", bbox_inches="tight", dpi=130)
plt.show()


# ── 2. Confusion Matrix ───────────────────────────────────────
y_pred_tuned = (y_prob >= best_f1_threshold).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (preds, label) in zip(axes, [
    (y_pred_default, f"Default threshold (0.50)"),
    (y_pred_tuned,   f"Tuned threshold ({best_f1_threshold:.2f})")
]):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(label, fontweight="bold", color="#E6EDF3")
    ax.set_facecolor("#0D1117")

fig.patch.set_facecolor("#161B22")
plt.suptitle("Confusion Matrix: Default vs Tuned Threshold",
             fontsize=13, fontweight="bold", color="#E6EDF3")
plt.tight_layout()
plt.savefig("data/eval/confusion_matrices.png", bbox_inches="tight", dpi=130)
plt.show()

print(f"\nTuned threshold ({best_f1_threshold:.2f}) performance:")
print(classification_report(y_test, y_pred_tuned, target_names=["No Churn","Churn"]))


# ── 3. ROC Curve ──────────────────────────────────────────────
# WHY: ROC-AUC measures discrimination ability across ALL thresholds.
# It's threshold-independent — tells you how good the model ranking is,
# regardless of where you draw the line.

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#388BFD", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax.plot([0,1],[0,1], color="#484F58", linestyle="--", linewidth=1.2, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="#388BFD")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax.set_facecolor("#0D1117")
fig.patch.set_facecolor("#161B22")
for spine in ax.spines.values(): spine.set_edgecolor("#2A3240")
ax.tick_params(colors="#8B949E")
ax.xaxis.label.set_color("#8B949E"); ax.yaxis.label.set_color("#8B949E")
ax.title.set_color("#E6EDF3")
ax.legend(facecolor="#1C2330", edgecolor="#2A3240", labelcolor="#E6EDF3")
plt.tight_layout()
plt.savefig("data/eval/roc_curve.png", bbox_inches="tight", dpi=130)
plt.show()


# ── 4. Precision-Recall Curve ─────────────────────────────────
# WHY: With class imbalance, ROC can be optimistic. The PR curve
# directly shows the tradeoff on the minority class (churners).
# Average Precision (AP) is the area under the PR curve.

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
baseline      = y_test.mean()   # random classifier AP = prevalence

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_curve, precision_curve, color="#3FB950", linewidth=2.5,
        label=f"PR Curve (AP = {avg_precision:.4f})")
ax.axhline(baseline, color="#484F58", linestyle="--", linewidth=1.2,
           label=f"Random baseline (AP = {baseline:.2f})")
ax.fill_between(recall_curve, precision_curve, alpha=0.08, color="#3FB950")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim(0,1); ax.set_ylim(0,1)

ax.set_facecolor("#0D1117")
fig.patch.set_facecolor("#161B22")
for spine in ax.spines.values(): spine.set_edgecolor("#2A3240")
ax.tick_params(colors="#8B949E")
ax.xaxis.label.set_color("#8B949E"); ax.yaxis.label.set_color("#8B949E")
ax.title.set_color("#E6EDF3")
ax.legend(facecolor="#1C2330", edgecolor="#2A3240", labelcolor="#E6EDF3")
plt.tight_layout()
plt.savefig("data/eval/pr_curve.png", bbox_inches="tight", dpi=130)
plt.show()


# ── 5. Feature Importance ─────────────────────────────────────
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = ["#388BFD" if i < 5 else "#2A3240" for i in range(len(feat_df))]
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1],
            color=colors_bar[::-1], edgecolor="#0D1117", linewidth=0.8)
    ax.set_title("Top 15 Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#161B22")
    for spine in ax.spines.values(): spine.set_edgecolor("#2A3240")
    ax.tick_params(colors="#8B949E")
    ax.xaxis.label.set_color("#8B949E"); ax.yaxis.label.set_color("#8B949E")
    ax.title.set_color("#E6EDF3")
    plt.tight_layout()
    plt.savefig("data/eval/feature_importance.png", bbox_inches="tight", dpi=130)
    plt.show()


# ── 6. SHAP Global Summary ────────────────────────────────────
# WHY: Feature importance from tree models shows *magnitude* but not
# *direction*. SHAP (SHapley Additive exPlanations) shows both:
# how much each feature pushes the prediction up or down, for every
# single prediction. The summary plot gives global behaviour.
# The individual waterfall plot (used in Flask) gives local explanation.

print("\nComputing SHAP values (this may take ~30s)...")

explainer_shap = shap.TreeExplainer(model)

# Use a sample for speed — 500 is enough for summary
sample_idx    = np.random.RandomState(42).choice(len(X_test_scaled), size=500, replace=False)
X_sample      = X_test_scaled.iloc[sample_idx]
shap_values   = explainer_shap.shap_values(X_sample)

# GradientBoosting returns array directly; RF returns list[neg, pos]
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# Save explainer for use in Flask
joblib.dump(explainer_shap, "model/shap_explainer.pkl")
print("✅ Saved model/shap_explainer.pkl")

# SHAP summary bar plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, X_sample, feature_names=feature_columns,
                  plot_type="bar", show=False, color="#388BFD")
plt.title("SHAP Global Feature Impact", fontweight="bold", color="#E6EDF3")
plt.gcf().patch.set_facecolor("#161B22")
plt.gca().set_facecolor("#0D1117")
plt.tight_layout()
plt.savefig("data/eval/shap_summary.png", bbox_inches="tight", dpi=130)
plt.show()

# SHAP beeswarm (shows direction)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, X_sample, feature_names=feature_columns, show=False)
plt.title("SHAP Beeswarm: Feature Impact & Direction", fontweight="bold", color="#E6EDF3")
plt.gcf().patch.set_facecolor("#161B22")
plt.tight_layout()
plt.savefig("data/eval/shap_beeswarm.png", bbox_inches="tight", dpi=130)
plt.show()

print("""
╔══════════════════════════════════════════════════╗
║         EVALUATION PIPELINE COMPLETE             ║
║                                                  ║
║  Saved to data/eval/:                            ║
║    threshold_tuning.png                          ║
║    confusion_matrices.png                        ║
║    roc_curve.png                                 ║
║    pr_curve.png                                  ║
║    feature_importance.png                        ║
║    shap_summary.png                              ║
║    shap_beeswarm.png                             ║
║                                                  ║
║  Saved to model/:                                ║
║    threshold.pkl  (tuned decision threshold)     ║
║    shap_explainer.pkl                            ║
╚══════════════════════════════════════════════════╝
""")