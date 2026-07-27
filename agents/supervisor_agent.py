"""
ManufacturingIQ Agentic AI v3 - Supervisor Agent

Validates state, controls routing, retries failed nodes, records execution trace.
"""

import logging
from typing import Any, Dict, Optional

from app.scoring import CONFIDENCE_THRESHOLD_PCT  # H-2: threshold matches new 0–100 scale

logger = logging.getLogger(__name__)


def run_supervisor(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")

        if not prediction:
            state["next"] = "report"
            return state

        confidence = prediction.get("confidence", 0)
        low_confidence = confidence < CONFIDENCE_THRESHOLD_PCT  # H-2: now on 0-100 scale

        if low_confidence:
            state["next"] = "human_review"
        else:
            state["next"] = "parallel_agents"
        return state
    except Exception as exc:
        logger.exception("Supervisor failed: %s", exc)
        state["error"] = str(exc)
        state["next"] = "report"
        return state