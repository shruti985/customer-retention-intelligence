# ============================================================
# segmentation.py
#
# WHY THIS EXISTS:
#   Churn probability is a number. Segment labels are a story.
#   A retention team can't act on "0.74 churn probability" —
#   they can act on "High-value at-risk: offer loyalty plan."
#   Segments bridge ML output and business action.
#
#   These are BUSINESS segments derived from feature combinations,
#   not ML clusters. That's intentional: interpretable, auditable,
#   and directly tied to retention playbooks.
# ============================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    name:        str
    code:        str          # short identifier for CSS/logic
    description: str
    urgency:     str          # critical / high / medium / low
    icon:        str
    playbook:    str          # one-line action for retention team


# ── Segment definitions ───────────────────────────────────────
SEGMENTS = {
    "high_value_at_risk": Segment(
        name        = "High-Value at Risk",
        code        = "high_value_at_risk",
        description = "Premium-plan customer in early tenure. High revenue impact if lost.",
        urgency     = "critical",
        icon        = "⭐",
        playbook    = "Escalate to CSM. Offer annual plan with loyalty perks within 48 hrs."
    ),
    "price_sensitive": Segment(
        name        = "Price-Sensitive Churner",
        code        = "price_sensitive",
        description = "High monthly charges with month-to-month contract. Leaving over cost.",
        urgency     = "high",
        icon        = "💸",
        playbook    = "Present bundle downgrade or 3-month discount. Anchor on total savings."
    ),
    "new_disengaging": Segment(
        name        = "New Disengaging Customer",
        code        = "new_disengaging",
        description = "Low tenure with minimal service usage. Never fully onboarded.",
        urgency     = "high",
        icon        = "🚀",
        playbook    = "Trigger onboarding re-engagement. Offer free trial of 2 add-on services."
    ),
    "low_engagement": Segment(
        name        = "Low-Engagement Customer",
        code        = "low_engagement",
        description = "Long-tenured but uses few services. At risk from competitor offers.",
        urgency     = "medium",
        icon        = "📉",
        playbook    = "Cross-sell streaming or security bundle at loyalty discount."
    ),
    "contract_cliff": Segment(
        name        = "Contract Cliff Risk",
        code        = "contract_cliff",
        description = "Month-to-month contract with moderate-to-high churn probability.",
        urgency     = "high",
        icon        = "📋",
        playbook    = "Proactive contract upgrade outreach. Show 2-year total cost savings."
    ),
    "senior_at_risk": Segment(
        name        = "Senior Customer at Risk",
        code        = "senior_at_risk",
        description = "Senior citizen with elevated churn signals. May need concierge support.",
        urgency     = "medium",
        icon        = "🧓",
        playbook    = "Assign dedicated phone support contact. Simplify billing communication."
    ),
    "stable": Segment(
        name        = "Stable Customer",
        code        = "stable",
        description = "Low churn risk. Monitor at next contract renewal.",
        urgency     = "low",
        icon        = "✅",
        playbook    = "No immediate action. Flag for loyalty reward at tenure milestone."
    ),
}


# ── Segmentation logic ────────────────────────────────────────
def get_segment(raw: dict, probability: float) -> Segment:
    """
    Assign a customer to exactly one segment using priority-ordered rules.
    Priority matters: a customer can match multiple rules; we return
    the highest-urgency one that fits.

    Args:
        raw:         dict of raw feature values (from predictor._raw)
        probability: churn probability (0–100 scale from predictor)
    """
    p             = probability / 100.0
    tenure        = raw["tenure"]
    monthly       = raw["MonthlyCharges"]
    contract      = raw["Contract"]
    num_services  = raw["num_services"]
    senior        = raw["SeniorCitizen"]
    high_val_risk = raw["is_high_value_at_risk"]
    tech_support  = raw["TechSupport"]
    security      = raw["OnlineSecurity"]

    # Rule 1: High-value at risk (most urgent — highest revenue impact)
    if high_val_risk and p >= 0.45:
        return SEGMENTS["high_value_at_risk"]

    # Rule 2: Price sensitive — high charges + flexible contract + high prob
    if monthly > 70 and contract == "Month-to-month" and p >= 0.50:
        return SEGMENTS["price_sensitive"]

    # Rule 3: New customer, not engaging
    if tenure < 12 and num_services <= 2 and p >= 0.35:
        return SEGMENTS["new_disengaging"]

    # Rule 4: Contract cliff — on flexible contract with meaningful risk
    if contract == "Month-to-month" and p >= 0.40:
        return SEGMENTS["contract_cliff"]

    # Rule 5: Senior with elevated risk
    if senior and p >= 0.35:
        return SEGMENTS["senior_at_risk"]

    # Rule 6: Long-tenured but disengaged
    if tenure >= 24 and num_services <= 3 and p >= 0.30:
        return SEGMENTS["low_engagement"]

    # Default: stable
    return SEGMENTS["stable"]