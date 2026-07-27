"""
ManufacturingIQ Agentic AI - Engineering Report Agent

Aggregates all agent outputs into a structured engineering report.
State values are TypedDicts (plain dicts at runtime) — use .get() for all field access.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _gd(obj: Any, key: str, default=None):
    """Get a field from a dict or object safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_report(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        shap_exp = state.get("shap_explanation")
        retrieved = state.get("retrieved_documents", [])
        recommendations = state.get("maintenance_recommendations", [])
        risk = state.get("risk_assessment")
        trend = state.get("trend_analysis")
        impact = state.get("operational_impact")

        if not prediction:
            state["engineering_report"] = {
                "prediction_summary": "No prediction available.",
                "technical_explanation": "",
                "primary_drivers": [],
                "retrieved_evidence": [],
                "maintenance_recommendations": [],
                "risk_assessment": risk or {},
                "confidence": 0.0,
                "final_recommendation": "Unable to generate report without prediction.",
            }
            return state

        machine_status = _gd(prediction, "machine_status", "Unknown")
        health_score = _gd(prediction, "health_score", 0.0)
        failure_prob = _gd(prediction, "failure_probability", 0.0)
        confidence = _gd(prediction, "confidence", 0.0)

        summary = (
            f"Machine classified as {machine_status} with health score {health_score} "
            f"and failure probability {failure_prob:.2f}."
        )

        explanation_text = _gd(shap_exp, "explanation_text") or "No explanation available."
        drivers = _gd(shap_exp, "top_contributors") or []
        recs = recommendations or []
        trend_summary = _gd(trend, "summary") or "Trend analysis not available."
        impact_notes = _gd(impact, "notes") or "Operational impact not available."

        final_recommendation = _build_final_recommendation(prediction, risk, recommendations)

        # H-3: collect citation strings from retrieved documents
        citations = [
            _gd(doc, "citation")
            for doc in retrieved
            if _gd(doc, "citation")
        ]

        state["engineering_report"] = {
            "prediction_summary": summary,
            "technical_explanation": explanation_text,
            "primary_drivers": list(drivers),
            "retrieved_evidence": list(retrieved),
            "retrieved_evidence_citations": citations,   # H-3
            "maintenance_recommendations": list(recs),
            "trend_analysis": trend,
            "risk_assessment": risk or {},
            "confidence": float(confidence),
            "final_recommendation": final_recommendation,
        }
    except Exception as exc:
        logger.exception("Report agent failed: %s", exc)
        state["engineering_report"] = {
            "prediction_summary": "Report generation failed.",
            "technical_explanation": "",
            "primary_drivers": [],
            "retrieved_evidence": [],
            "maintenance_recommendations": [],
            "risk_assessment": {},
            "confidence": 0.0,
            "final_recommendation": "Unable to generate report due to an internal error.",
        }
    return state


def _build_final_recommendation(
    prediction: Any,
    risk: Any,
    recommendations: List[Any],
) -> str:
    if recommendations:
        actions = [
            _gd(r, "action", "No action specified")
            for r in recommendations[:2]
        ]
        action_text = "; ".join(a for a in actions if a)
    else:
        action_text = "Continue routine monitoring."

    risk_text = _gd(risk, "urgency") or "Unknown urgency"
    confidence = _gd(prediction, "confidence", 0.0)
    return f"{action_text} | Urgency: {risk_text} | Confidence: {confidence:.1f}%"