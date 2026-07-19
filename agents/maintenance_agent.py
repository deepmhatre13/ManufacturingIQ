"""
ManufacturingIQ Agentic AI - Maintenance Recommendation Agent

Generates prioritized maintenance actions supported by retrieved knowledge.
"""

import logging
from typing import Any, Dict, List

from state.schema import MaintenanceRecommendation, RetrievedDocument
from agents._utils import record_agent_error  # H-5

logger = logging.getLogger(__name__)


def run_maintenance(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        retrieved = state.get("retrieved_documents", [])
        recommendations: List[MaintenanceRecommendation] = []

        if not prediction:
            state["maintenance_recommendations"] = recommendations
            return state

        status = prediction.get("machine_status", "")
        risk = prediction.get("risk_level", "")

        if status == "Critical" or risk == "High":
            recommendations.append(
                MaintenanceRecommendation(
                    action="Schedule immediate maintenance inspection within 24 hours.",
                    priority="High",
                    rationale="High failure probability and risk level indicate imminent failure risk.",
                    references=[
                        _to_ref(d)
                        for d in retrieved
                        if (d.get("title") if isinstance(d, dict) else d.title) in ["Thermal Failure Mode", "Mechanical Stress Indicators", "Tool Wear Failure Mode"]
                    ][:3],
                )
            )

        if prediction.get("failure_probability", 0) > 0.4:
            recommendations.append(
                MaintenanceRecommendation(
                    action="Inspect tooling and consider replacement.",
                    priority="Medium",
                    rationale="Elevated failure probability suggests significant wear or thermal exposure.",
                    references=[
                        _to_ref(d)
                        for d in retrieved
                        if (d.get("title") if isinstance(d, dict) else d.title) in ["Tool Wear Failure Mode", "Predictive Maintenance Principles"]
                    ][:2],
                )
            )

        # Generic fallback recommendation
        if not recommendations:
            recommendations.append(
                MaintenanceRecommendation(
                    action="Continue monitoring; no immediate action required.",
                    priority="Low",
                    rationale="Prediction indicates healthy operating range.",
                    references=[],
                )
            )

        state["maintenance_recommendations"] = recommendations
    except Exception as exc:
        logger.exception("Maintenance agent failed: %s", exc)
        record_agent_error(state, "node_maintenance", exc)  # H-5
        state["maintenance_recommendations"] = []
    return state


def _to_ref(doc: Any) -> RetrievedDocument:
    if isinstance(doc, dict):
        return RetrievedDocument(
            title=doc.get("title", ""),
            source=doc.get("source", ""),
            section=doc.get("section"),
            excerpt=doc.get("excerpt", ""),
            confidence=doc.get("confidence", 0.0),
        )
    return RetrievedDocument(
        title=doc.title,
        source=doc.source,
        section=doc.section,
        excerpt=doc.excerpt,
        confidence=doc.confidence,
    )