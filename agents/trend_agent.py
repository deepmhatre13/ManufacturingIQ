"""
ManufacturingIQ Agentic AI - Trend Analysis Agent

Compares current prediction against history to detect degradation or improvement.
"""

import logging
from typing import Any, Dict

from state.schema import TrendAnalysis
from agents._utils import record_agent_error  # H-5

logger = logging.getLogger(__name__)


def run_trend(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        history = state.get("prediction_history", [])

        if not prediction or len(history) < 2:
            state["trend_analysis"] = TrendAnalysis(
                direction="Stable",
                health_trend="Insufficient history",
                risk_trend="Insufficient history",
                summary="Not enough prediction history to determine trend.",
            )
            return state

        recent = history[-5:]
        healths = [float(h.get("health_score", 0)) for h in recent if "health_score" in h]
        risks = [float(h.get("failure_probability", 0)) for h in recent if "failure_probability" in h]

        if len(healths) >= 2:
            health_delta = healths[-1] - healths[0]
            if health_delta < -10:
                health_trend = "Decreasing"
            elif health_delta > 10:
                health_trend = "Improving"
            else:
                health_trend = "Stable"
        else:
            health_trend = "Stable"

        if len(risks) >= 2:
            risk_delta = risks[-1] - risks[0]
            if risk_delta > 0.1:
                risk_trend = "Increasing"
            elif risk_delta < -0.1:
                risk_trend = "Decreasing"
            else:
                risk_trend = "Stable"
        else:
            risk_trend = "Stable"

        if health_trend == "Decreasing" and risk_trend == "Increasing":
            direction = "Rapid degradation"
        elif health_trend == "Improving" and risk_trend == "Decreasing":
            direction = "Improving"
        elif health_trend == "Stable" and risk_trend == "Stable":
            direction = "Stable"
        else:
            direction = "Mixed"

        summary = (
            f"Health trend is {health_trend.lower()} and risk trend is {risk_trend.lower()} "
            f"over the last {len(recent)} predictions."
        )

        state["trend_analysis"] = TrendAnalysis(
            direction=direction,
            health_trend=health_trend,
            risk_trend=risk_trend,
            summary=summary,
        )
    except Exception as exc:
        logger.exception("Trend agent failed: %s", exc)
        record_agent_error(state, "node_trend", exc)  # H-5
        state["trend_analysis"] = TrendAnalysis()
    return state