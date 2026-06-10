# ============================================================
# explainer.py
# Rule-based explainer + retention suggestion engine.
#
# Design principle: The ML model predicts probability.
# This module explains *why* using feature thresholds
# grounded in domain knowledge — no black box required.
# This mirrors real-world MLOps: model for scoring,
# rules for interpretability.
# ============================================================

def explain(raw: dict, risk: str) -> dict:
    """
    Given the raw feature values and risk tier,
    returns a list of human-readable reasons and
    a list of targeted retention suggestions.

    Each reason is tied to a concrete feature threshold,
    making the output auditable and trustworthy.
    """
    reasons     = []
    suggestions = []

    tenure          = raw["tenure"]
    monthly         = raw["MonthlyCharges"]
    contract        = raw["Contract"]
    payment         = raw["PaymentMethod"]
    internet        = raw["InternetService"]
    tech_support    = raw["TechSupport"]
    online_security = raw["OnlineSecurity"]
    num_services    = raw["num_services"]
    senior          = raw["SeniorCitizen"]
    paperless       = raw["PaperlessBilling"]
    partner         = raw["Partner"]
    dependents      = raw["Dependents"]
    high_val_risk   = raw["is_high_value_at_risk"]

    # ── REASONS ──────────────────────────────────────────────

    # Contract type — single strongest predictor
    if contract == "Month-to-month":
        reasons.append({
            "icon":  "📋",
            "text":  "Month-to-month contract — no long-term commitment locks customer in",
            "weight": "high"
        })
    elif contract == "One year":
        reasons.append({
            "icon":  "📋",
            "text":  "One-year contract — moderate commitment, still at risk at renewal",
            "weight": "medium"
        })

    # Tenure
    if tenure < 6:
        reasons.append({
            "icon":  "📅",
            "text":  f"Very low tenure ({int(tenure)} months) — customer hasn't built loyalty yet",
            "weight": "high"
        })
    elif tenure < 18:
        reasons.append({
            "icon":  "📅",
            "text":  f"Low-to-mid tenure ({int(tenure)} months) — still in early relationship stage",
            "weight": "medium"
        })

    # Monthly charges
    if monthly > 80:
        reasons.append({
            "icon":  "💸",
            "text":  f"High monthly charges (₹{monthly:.0f}) — customer may feel the value isn't justified",
            "weight": "high"
        })
    elif monthly > 65:
        reasons.append({
            "icon":  "💸",
            "text":  f"Above-average monthly charges (₹{monthly:.0f}) — price sensitivity is a risk",
            "weight": "medium"
        })

    # Payment method
    if payment == "Electronic check":
        reasons.append({
            "icon":  "💳",
            "text":  "Pays by electronic check — this group has the highest churn rate in our data",
            "weight": "medium"
        })

    # Internet service
    if internet == "Fiber optic":
        reasons.append({
            "icon":  "🌐",
            "text":  "Fiber optic internet user — higher churn segment, possibly due to pricing expectations",
            "weight": "medium"
        })
    elif internet == "No":
        reasons.append({
            "icon":  "🌐",
            "text":  "No internet service — limited product engagement reduces stickiness",
            "weight": "low"
        })

    # Support services
    if not tech_support:
        reasons.append({
            "icon":  "🔧",
            "text":  "No tech support subscription — service issues may go unresolved",
            "weight": "medium"
        })

    if not online_security:
        reasons.append({
            "icon":  "🔒",
            "text":  "No online security add-on — lower perceived value from the bundle",
            "weight": "low"
        })

    # Services count
    if num_services <= 2:
        reasons.append({
            "icon":  "📦",
            "text":  f"Only {int(num_services)} services subscribed — low product integration = easier to switch",
            "weight": "medium"
        })

    # Lifestyle / stability signals
    if not partner and not dependents:
        reasons.append({
            "icon":  "👤",
            "text":  "Single, no dependents — more mobile lifestyle, easier to switch providers",
            "weight": "low"
        })

    # Senior citizen
    if senior:
        reasons.append({
            "icon":  "🧓",
            "text":  "Senior citizen — this demographic shows above-average churn rates",
            "weight": "low"
        })

    # Paperless billing
    if paperless:
        reasons.append({
            "icon":  "📧",
            "text":  "Paperless billing enabled — correlated with higher churn (digital-savvy, comparison-shops more)",
            "weight": "low"
        })

    # High value at risk flag
    if high_val_risk:
        reasons.append({
            "icon":  "⚠️",
            "text":  "High-value customer in early tenure — paying premium pricing without built loyalty",
            "weight": "high"
        })

    # ── SUGGESTIONS ──────────────────────────────────────────
    # Suggestions are driven by the most impactful reasons,
    # ordered by business impact.

    if contract == "Month-to-month":
        suggestions.append({
            "icon":   "🤝",
            "action": "Offer a long-term contract incentive",
            "detail": "Provide 15–20% discount on annual or 2-year plan to reduce churn risk at its root cause"
        })

    if tenure < 12:
        suggestions.append({
            "icon":   "🎁",
            "action": "Launch an early-loyalty rewards program",
            "detail": "Send a personalised retention offer at Month 3 and Month 6 — critical churn windows"
        })

    if monthly > 65:
        suggestions.append({
            "icon":   "💰",
            "action": "Offer a targeted plan downgrade or bundle discount",
            "detail": "Present a slightly cheaper bundle that retains core services — losing margin beats losing the customer"
        })

    if payment == "Electronic check":
        suggestions.append({
            "icon":   "🏦",
            "action": "Encourage switch to auto-pay (bank transfer or credit card)",
            "detail": "Offer one month bill credit for switching — auto-pay customers churn significantly less"
        })

    if not tech_support or not online_security:
        suggestions.append({
            "icon":   "🛡️",
            "action": "Offer a free 1-month trial of Tech Support + Online Security",
            "detail": "Once customers experience these services, uptake and retention both improve"
        })

    if num_services <= 2:
        suggestions.append({
            "icon":   "📡",
            "action": "Cross-sell complementary services at a bundled rate",
            "detail": "Customers with 4+ services churn at half the rate — each add-on is a retention hook"
        })

    if senior:
        suggestions.append({
            "icon":   "📞",
            "action": "Assign a dedicated senior support contact",
            "detail": "Proactive phone outreach for senior customers significantly improves satisfaction scores"
        })

    if high_val_risk:
        suggestions.append({
            "icon":   "⭐",
            "action": "Escalate to a customer success manager immediately",
            "detail": "High-value early-tenure customers have the highest revenue impact if churned — prioritise personal outreach"
        })

    # Always include a generic but personalised-sounding fallback
    if not suggestions:
        suggestions.append({
            "icon":   "💬",
            "action": "Schedule a proactive satisfaction check-in call",
            "detail": "Even low-risk customers benefit from being heard — a 5-min call can surface hidden dissatisfaction"
        })

    # Sort reasons: high weight first
    weight_order = {"high": 0, "medium": 1, "low": 2}
    reasons.sort(key=lambda r: weight_order.get(r["weight"], 3))

    # Cap to top 5 reasons and top 4 suggestions for readability
    return {
        "reasons":     reasons[:5],
        "suggestions": suggestions[:4]
    }