
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, f1_score
)

# Plotting config
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

# Output dirs
os.makedirs("model", exist_ok=True)
os.makedirs("data",  exist_ok=True)

print("✅ All imports successful.")
# ── Load Data ─────────────
df = pd.read_csv("data/telco_churn.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nFirst 5 rows:")
df.head()
# ── Basic Info ──────────────────────
print("── Data Types & Non-Null Counts ──")
df.info()

print("\n── Missing Values ──")
print(df.isnull().sum())

print("\n── Numeric Summary ──")
df.describe()
# Raw dataset has TotalCharges as object dtype; ~11 rows have blank strings
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

blanks_fixed = df["TotalCharges"].isnull().sum()
print(f"TotalCharges blanks (new customers with tenure=0): {blanks_fixed}")

# These customers have tenure=0 — they just joined, TotalCharges = 0 makes sense
df["TotalCharges"].fillna(0, inplace=True)
print("✅ TotalCharges fixed.")


# ── EDA — Target Distribution ─────────────────────────
churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count plot
axes[0].bar(churn_counts.index, churn_counts.values,
            color=["#4CAF82", "#E05C5C"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Churn Count")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Percentage pie
axes[1].pie(churn_pct.values, labels=churn_counts.index,
            autopct="%1.1f%%", colors=["#4CAF82", "#E05C5C"],
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Churn Split")

plt.suptitle("Target Variable: Churn Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/eda_target_distribution.png", bbox_inches="tight")
plt.show()

# ── OBSERVATION ─────────────────────────────
print("""
📌 OBSERVATION:
   The dataset is imbalanced — ~73% No Churn, ~27% Churn.
   This means accuracy alone is misleading as a metric.
   A dumb model that always predicts "No Churn" gets 73% accuracy.
   We will use F1-Score and Recall as primary metrics.
""")


# ──  EDA — Numeric Features vs Churn ──────────────────
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(numeric_cols):
    for label, color in zip(["No", "Yes"], ["#4CAF82", "#E05C5C"]):
        subset = df[df["Churn"] == label][col]
        axes[i].hist(subset, bins=30, alpha=0.65, label=label, color=color, edgecolor="white")
    axes[i].set_title(f"{col} by Churn")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].legend(title="Churn")

plt.suptitle("Numeric Feature Distributions by Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/eda_numeric_distributions.png", bbox_inches="tight")
plt.show()

print("""
📌 OBSERVATIONS:
   • tenure:         Churners have significantly LOWER tenure — they leave early.
   • MonthlyCharges: Churners tend to have HIGHER monthly charges.
   • TotalCharges:   Churners have lower total charges (correlates with low tenure).
   These patterns will drive our engineered features.
""")


# ── EDA — Churn Rate by Categorical Features ─────────
cat_cols = ["Contract", "PaymentMethod", "InternetService", "TechSupport",
            "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = (
        df.groupby(col)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
    )
    churn_rate.columns = [col, "ChurnRate"]
    churn_rate = churn_rate.sort_values("ChurnRate", ascending=False)

    bars = axes[i].barh(churn_rate[col], churn_rate["ChurnRate"],
                        color="#E05C5C", edgecolor="white", linewidth=1)
    axes[i].set_title(f"Churn Rate by {col}")
    axes[i].set_xlabel("Churn Rate (%)")
    axes[i].xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, churn_rate["ChurnRate"]):
        axes[i].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9)

# Hide empty subplot
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/eda_categorical_churn_rates.png", bbox_inches="tight")
plt.show()

print("""
📌 OBSERVATIONS:
   • Contract:         Month-to-month customers churn at ~42% vs ~3% for 2-year contracts.
   • PaymentMethod:    Electronic check users churn the most (~45%).
   • InternetService:  Fiber Optic customers churn at ~42% — possibly due to pricing.
   • TechSupport/OnlineSecurity: No-support customers churn far more.
   • SeniorCitizen:    Seniors churn at ~42% vs ~26% for non-seniors.
""")


# ── EDA — Correlation Heatmap ────────────────────────
# Temporary binary encode for correlation only
df_corr = df.copy()
df_corr["Churn_bin"] = (df_corr["Churn"] == "Yes").astype(int)

corr_cols = ["tenure", "MonthlyCharges", "TotalCharges",
             "SeniorCitizen", "Churn_bin"]
corr_matrix = df_corr[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 11}, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap (Numeric Features)", fontweight="bold")
plt.tight_layout()
plt.savefig("data/eda_correlation_heatmap.png", bbox_inches="tight")
plt.show()

print("""
📌 OBSERVATIONS:
   • tenure vs Churn: Negative correlation (-0.35) — longer tenure = less churn.
   • MonthlyCharges vs Churn: Positive correlation (+0.19) — higher charges = more churn.
   • tenure & TotalCharges are highly correlated (0.83) — expected, not multicollinearity problem here.
""")


# ── Preprocessing ────────────────────────────────────
df_model = df.copy()

# Drop customerID — it's just an identifier
df_model.drop(columns=["customerID"], inplace=True)

# Encode target
df_model["Churn"] = (df_model["Churn"] == "Yes").astype(int)

# ── Binary Yes/No columns → 0/1
binary_cols = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling"
]
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0, "No phone service": 0,
                                        "No internet service": 0})

# ── One-hot encode multi-category columns
ohe_cols = ["Contract", "PaymentMethod", "InternetService", "gender"]
df_model = pd.get_dummies(df_model, columns=ohe_cols, drop_first=False)

# Convert all bool columns (from get_dummies) to int
bool_cols = df_model.select_dtypes(include="bool").columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

print(f"Shape after encoding: {df_model.shape}")
print(f"\nColumn list:\n{list(df_model.columns)}")


# ── CELL 10: Feature Engineering ─────────────────────────────
print("Engineering new features...\n")

#    charges_per_month_of_tenure
#    High early spend relative to time spent = customer is paying a lot
#    without having built loyalty yet → churn risk
df_model["charges_per_tenure"] = (
    df_model["MonthlyCharges"] / (df_model["tenure"] + 1)
)

#  num_services_subscribed
#    More services = more integrated = stickier customer
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]
df_model["num_services"] = df_model[service_cols].sum(axis=1)

#  is_high_value_at_risk
#    High paying but relatively new customer — most valuable to retain
df_model["is_high_value_at_risk"] = (
    (df_model["MonthlyCharges"] > 70) & (df_model["tenure"] < 12)
).astype(int)

print("✅ Engineered features added:")
print(f"   charges_per_tenure       — range: {df_model['charges_per_tenure'].min():.2f} to {df_model['charges_per_tenure'].max():.2f}")
print(f"   num_services             — range: {df_model['num_services'].min()} to {df_model['num_services'].max()}")
print(f"   is_high_value_at_risk    — positive count: {df_model['is_high_value_at_risk'].sum()}")

# Visualize engineered features vs churn
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# charges_per_tenure
for label, color in zip([0, 1], ["#4CAF82", "#E05C5C"]):
    axes[0].hist(df_model[df_model["Churn"] == label]["charges_per_tenure"],
                 bins=30, alpha=0.65, label=["No Churn", "Churn"][label], color=color)
axes[0].set_title("charges_per_tenure vs Churn")
axes[0].legend()

# num_services
churn_by_services = df_model.groupby("num_services")["Churn"].mean() * 100
axes[1].bar(churn_by_services.index, churn_by_services.values, color="#5B8FD4", edgecolor="white")
axes[1].set_title("Churn Rate by # of Services")
axes[1].set_xlabel("num_services")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

# is_high_value_at_risk
churn_by_risk = df_model.groupby("is_high_value_at_risk")["Churn"].mean() * 100
axes[2].bar(["Regular", "High-Value at Risk"], churn_by_risk.values,
            color=["#4CAF82", "#E05C5C"], edgecolor="white")
axes[2].set_title("Churn Rate: High-Value at Risk Flag")
axes[2].set_ylabel("Churn Rate (%)")
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
for i, v in enumerate(churn_by_risk.values):
    axes[2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

plt.suptitle("Engineered Feature Validation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/eda_engineered_features.png", bbox_inches="tight")
plt.show()

print("""
📌 VALIDATION:
   • charges_per_tenure: Churners cluster at higher values — feature is meaningful.
   • num_services: Clear inverse relationship — more services, less churn.
   • is_high_value_at_risk: This segment churns at noticeably higher rates.
""")


# ── CLEAN FIRST ──
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.fillna(df_model.median(numeric_only=True))

# ── THEN SPLIT ──
X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train size : {X_train.shape[0]} rows")
print(f"Test  size : {X_test.shape[0]} rows")
print(f"Churn rate (train): {y_train.mean()*100:.1f}%")
print(f"Churn rate (test) : {y_test.mean()*100:.1f}%")

# Scale only numeric columns
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges",
              "charges_per_tenure", "num_services"]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

print("\n✅ Scaling done. Only fit on train set to prevent data leakage.")


# ── Model Training ───────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",    # handles class imbalance
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    f1     = f1_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "f1":        f1,
        "roc_auc":   roc,
        "precision": report["1"]["precision"],
        "recall":    report["1"]["recall"],
        "report":    report
    }

    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC Score: {roc:.4f}")


# ── Model Comparison ────────────────────────────────
comparison_df = pd.DataFrame([
    {
        "Model":     name,
        "F1 Score":  f"{res['f1']:.4f}",
        "ROC-AUC":   f"{res['roc_auc']:.4f}",
        "Precision": f"{res['precision']:.4f}",
        "Recall":    f"{res['recall']:.4f}"
    }
    for name, res in results.items()
])

print("\n── MODEL COMPARISON ──")
print(comparison_df.to_string(index=False))

# Visualize F1 and ROC-AUC
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

names   = list(results.keys())
f1s     = [results[n]["f1"]      for n in names]
rocaucs = [results[n]["roc_auc"] for n in names]
colors  = ["#5B8FD4", "#4CAF82", "#E05C5C"]

axes[0].barh(names, f1s, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("F1 Score (Churn Class)")
axes[0].set_xlabel("F1 Score")
axes[0].set_xlim(0.4, 0.75)
for i, v in enumerate(f1s):
    axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontweight="bold")

axes[1].barh(names, rocaucs, color=colors, edgecolor="white", linewidth=1.5)
axes[1].set_title("ROC-AUC Score")
axes[1].set_xlabel("ROC-AUC")
axes[1].set_xlim(0.70, 0.90)
for i, v in enumerate(rocaucs):
    axes[1].text(v + 0.001, i, f"{v:.4f}", va="center", fontweight="bold")

plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/model_comparison.png", bbox_inches="tight")
plt.show()


# ── Confusion Matrices ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontweight="bold")

plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/confusion_matrices.png", bbox_inches="tight")
plt.show()


# ── Feature Importance (Random Forest) ──────────────
rf_model      = results["Random Forest"]["model"]
importances   = rf_model.feature_importances_
feat_imp_df   = pd.DataFrame({
    "Feature":    feature_columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feat_imp_df["Feature"][::-1],
               feat_imp_df["Importance"][::-1],
               color="#5B8FD4", edgecolor="white", linewidth=1)
ax.set_title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("data/feature_importance.png", bbox_inches="tight")
plt.show()

print("\nTop 10 Features:")
print(feat_imp_df[["Feature", "Importance"]].head(10).to_string(index=False))


# ── Select Best Model ────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["f1"])
best_model = results[best_name]["model"]
best_f1    = results[best_name]["f1"]
best_roc   = results[best_name]["roc_auc"]

print(f"""
╔══════════════════════════════════════════╗
║         BEST MODEL SELECTED              ║
║                                          ║
║  Model   : {best_name:<30} ║
║  F1 Score: {best_f1:<30.4f} ║
║  ROC-AUC : {best_roc:<30.4f} ║
╚══════════════════════════════════════════╝

WHY F1 OVER ACCURACY?
  The dataset is imbalanced (73% No Churn, 27% Churn).
  A dumb model predicting "No Churn" always gets 73% accuracy.
  F1 balances Precision and Recall — penalizes the model for
  missing actual churners (bad recall) and false alarms (bad precision).

WHY RECALL MATTERS HERE?
  Missing a churner (False Negative) costs the company a customer.
  A false alarm (False Positive) just means we offer a discount unnecessarily.
  In business terms: FN >> FP in cost. So Recall is especially valuable.
""")


# ── Save Everything ──────────────────────────────────
joblib.dump(best_model,      "model/best_model.pkl")
joblib.dump(scaler,          "model/scaler.pkl")
joblib.dump(feature_columns, "model/feature_columns.pkl")
joblib.dump(scale_cols,      "model/scale_cols.pkl")

print("✅ Saved:")
print("   model/best_model.pkl")
print("   model/scaler.pkl")
print("   model/feature_columns.pkl")
print("   model/scale_cols.pkl")

print(f"\n{'='*50}")
print("DAY 1 COMPLETE.")
print(f"{'='*50}")
print("""
Summary:
  ✅ EDA done          — 5 insightful plots saved to data/
  ✅ Preprocessing     — binary encoding, OHE, TotalCharges fix
  ✅ Feature Engineering — 3 domain-driven features added
  ✅ 3 models trained  — Logistic Regression, RF, Gradient Boosting
  ✅ Evaluation done   — F1, ROC-AUC, Precision, Recall, Confusion Matrix
  ✅ Best model saved  — ready for Day 2 Flask app

Day 2: Flask app, explainer logic, retention suggestions, frontend UI.
""")