"""
ManufacturingIQ Agentic AI - Risk Assessment Agent

Estimates risk severity, business impact, and urgency.
"""

import logging
from typing import Any, Dict

from state.schema import RiskAssessment
from agents._utils import record_agent_error  # H-5

logger = logging.getLogger(__name__)


def run_risk(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        if not prediction:
            state["risk_assessment"] = RiskAssessment(
                risk_level="Unknown",
                severity="Unknown",
                business_impact="Unknown",
                rationale="No prediction available.",
                urgency="Unknown",
            )
            return state

        risk = prediction.get("risk_level", "")
        status = prediction.get("machine_status", "")
        prob = prediction.get("failure_probability", 0.0)

        if status == "Critical" or risk == "High" or prob > 0.7:
            level = "Critical"
            severity = "High"
            impact = "Critical operational impact"
            urgency = "Immediate"
            rationale = "Very high failure probability and degraded health score indicate imminent failure risk."
        elif status == "Warning" or risk == "Medium" or prob > 0.4:
            level = "Medium"
            severity = "Medium"
            impact = "High impact"
            urgency = "Within 24 hours"
            rationale = "Elevated failure probability suggests accelerated degradation; prompt maintenance advisable."
        else:
            level = "Low"
            severity = "Low"
            impact = "Low impact"
            urgency = "Scheduled"
            rationale = "Operating parameters are within acceptable ranges; routine monitoring sufficient."

        state["risk_assessment"] = RiskAssessment(
            risk_level=level,
            severity=severity,
            business_impact=impact,
            rationale=rationale,
            urgency=urgency,
        )
    except Exception as exc:
        logger.exception("Risk agent failed: %s", exc)
        record_agent_error(state, "node_risk", exc)  # H-5
        state["risk_assessment"] = RiskAssessment()
    return state