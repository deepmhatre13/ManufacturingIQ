"""
ManufacturingIQ Agentic AI - FastAPI Integration

Wraps the LangGraph workflow as a callable service.
"""

import logging
from typing import Any, Dict

from state.schema import MachineInput
from graph.graph import run_graph
from history.utils import append_history
from reports.generator import to_json, to_markdown

logger = logging.getLogger(__name__)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value if value is None else dict(value)


def _get_attr_or_key(value, key):
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def run_agentic_pipeline(raw_input: Dict[str, Any], history: list[dict]) -> Dict[str, Any]:
    try:
        final_state = run_graph(raw_input, history)
        report = final_state.get("engineering_report")
        prediction = final_state.get("prediction")
        shap_exp = final_state.get("shap_explanation")
        risk = final_state.get("risk_assessment")
        trend = final_state.get("trend_analysis")
        impact = final_state.get("operational_impact")
        recs = final_state.get("maintenance_recommendations", [])
        retrieved = final_state.get("retrieved_documents", [])
        trace = final_state.get("execution_trace")
        node_status = final_state.get("node_status", {})

        if report and prediction:
            append_history({
                "timestamp": _get_attr_or_key(report, "prediction_summary") or "",
                "machine_type": raw_input.get("Type", ""),
                "health_score": _get_attr_or_key(prediction, "health_score"),
                "failure_probability": _get_attr_or_key(prediction, "failure_probability"),
                "risk_level": _get_attr_or_key(prediction, "risk_level"),
                "status": _get_attr_or_key(prediction, "machine_status"),
                "confidence": _get_attr_or_key(prediction, "confidence"),
                "recommendation": _get_attr_or_key(report, "final_recommendation") or "",
                "trend": _get_attr_or_key(trend, "direction") if trend else "",
            })
            return {
                "prediction": _as_dict(prediction),
                "shap_explanation": _as_dict(shap_exp) if shap_exp else {},
                "risk_assessment": _as_dict(risk) if risk else {},
                "trend_analysis": _as_dict(trend) if trend else {},
                "operational_impact": _as_dict(impact) if impact else {},
                "maintenance_recommendations": [_as_dict(r) for r in recs],
                "retrieved_documents": [_as_dict(d) for d in retrieved],
                "report": to_json(report),
                "markdown_report": to_markdown(report),
                "execution_trace": _as_dict(trace),
                "node_status": node_status,
            }
        return {}
    except Exception as exc:
        logger.exception("Agentic pipeline failed: %s", exc)
        return {}