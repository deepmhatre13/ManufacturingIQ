"""
ManufacturingIQ Agentic AI - Operational Impact Agent

Classifies operational impact and maintenance priority without fabricating exact costs.
"""

import logging
from typing import Any, Dict

from state.schema import OperationalImpact
from agents._utils import record_agent_error  # H-5

logger = logging.getLogger(__name__)


def run_operational_impact(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        risk = state.get("risk_assessment")
        if not prediction:
            state["operational_impact"] = OperationalImpact(
                impact_category="Unknown",
                estimated_priority="Unknown",
                severity_score=0.0,
                notes="No prediction available.",
            )
            return state

        prob = prediction.get("failure_probability", 0.0)
        if prediction.get("machine_status", "") == "Critical" or prob > 0.7:
            category = "Critical operational impact"
            priority = "Immediate"
            score = 9.0
            notes = "Immediate maintenance required to avoid unplanned downtime."
        elif prediction.get("machine_status", "") == "Warning" or prob > 0.4:
            category = "High impact"
            priority = "Within 24 hours"
            score = 6.5
            notes = "Deferred maintenance may lead to increased downtime and production loss."
        else:
            category = "Low impact"
            priority = "Scheduled"
            score = 3.0
            notes = "Routine maintenance window acceptable."

        state["operational_impact"] = OperationalImpact(
            impact_category=category,
            estimated_priority=priority,
            severity_score=score,
            notes=notes,
        )
    except Exception as exc:
        logger.exception("Operational impact agent failed: %s", exc)
        record_agent_error(state, "node_operational_impact", exc)  # H-5
        state["operational_impact"] = OperationalImpact()
    return state