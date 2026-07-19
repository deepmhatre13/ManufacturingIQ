"""
ManufacturingIQ - Shared Prediction Scoring  (H-1, H-2)

Single source of truth for health score, machine status, risk level, and
confidence.  Both app/predictor.py and agents/prediction_agent.py import
from here so the two code-paths are guaranteed to produce identical values.

H-1 fix:
    `app/predictor.py::calculate_health` and `agents/prediction_agent.py`
    had divergent thresholds (predictor had a "High Risk" band; agent had
    none) and slightly different formulae.  This module is the canonical
    implementation used by both.

H-2 fix:
    The previous `confidence = min(99.5, 85 + health_score * 0.15)` was
    prediction-independent — it just tracked health linearly and always
    returned ≥ 85, so the supervisor's human-review branch (confidence < 70)
    was dead code.

    Replaced with a real uncertainty signal: the *prediction margin* —
    the distance of the predicted probability from the decision boundary
    (p = 0.5).  At p = 0.5 the model is maximally uncertain (confidence = 0);
    at p = 0 or p = 1 it is maximally certain (confidence = 100).

    Formula:  confidence = |p − 0.5| / 0.5 × 100

    The supervisor CONFIDENCE_THRESHOLD is updated to 70 (percent) to match
    this scale.  The routing contract (state["next"]) is unchanged.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Health / status / risk thresholds
# ---------------------------------------------------------------------------
#
# Canonical boundary definitions — do not duplicate these elsewhere.
#
#   health ≥ 80  →  Healthy    / Low    risk
#   health ≥ 50  →  Warning    / Medium risk
#   health ≥ 20  →  High Risk  / High   risk
#   health <  20  →  Critical   / High   risk
#
_THRESHOLDS = [
    (80.0, "Healthy",   "Low"),
    (50.0, "Warning",   "Medium"),
    (20.0, "High Risk", "High"),
    ( 0.0, "Critical",  "High"),
]

# Supervisor routes to human-review when confidence is below this value (%).
CONFIDENCE_THRESHOLD_PCT = 70.0


def calculate_health_and_status(
    failure_probability: float,
) -> tuple[float, str, str]:
    """
    Return ``(health_score, machine_status, risk_level)`` from a raw
    failure probability in [0, 1].

    Parameters
    ----------
    failure_probability : float
        Model output probability of failure in [0.0, 1.0].

    Returns
    -------
    health_score : float
        Health score clamped to [0.0, 100.0], rounded to 2 d.p.
    machine_status : str
        One of "Healthy", "Warning", "High Risk", "Critical".
    risk_level : str
        One of "Low", "Medium", "High".
    """
    health_score = round(max(0.0, min(100.0, (1.0 - failure_probability) * 100)), 2)
    for threshold, status, risk in _THRESHOLDS:
        if health_score >= threshold:
            return health_score, status, risk
    # Should be unreachable (last threshold is 0.0), but satisfy type-checker:
    return health_score, "Critical", "High"


def calculate_confidence(failure_probability: float) -> float:
    """
    Return a real, prediction-dependent confidence score in [0.0, 100.0].

    This is the *prediction margin*: how far the model's predicted
    probability is from the decision boundary (0.5).  A probability of
    exactly 0.5 → confidence = 0 (maximally uncertain); a probability of 0
    or 1 → confidence = 100 (maximally certain).

    Formula:  ``|p − 0.5| / 0.5 × 100``

    The supervisor compares this value against ``CONFIDENCE_THRESHOLD_PCT``
    (70) to decide whether to route to human review.
    """
    margin = abs(failure_probability - 0.5)
    return round(margin / 0.5 * 100.0, 2)
