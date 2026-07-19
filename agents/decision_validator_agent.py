import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def run_decision_validator(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        warnings = []
        pred = state.get("prediction")
        risk = state.get("risk_assessment")
        recs = state.get("maintenance_recommendations", [])
        report = state.get("engineering_report")

        if pred and risk:
            if pred.get("machine_status", "") == "Critical" and risk.get("urgency", "") == "Scheduled":
                warnings.append("Inconsistency: Critical status but scheduled urgency.")
            if pred.get("failure_probability", 0) > 0.8 and risk.get("severity", "") in {"Low", "Medium"}:
                warnings.append("Inconsistency: High failure prob but low severity.")

        trace = state.get("execution_trace")
        if isinstance(trace, dict):
            trace.setdefault("warnings", []).extend(warnings)
        state["node_status"] = state.get("node_status", {})
        state["node_status"]["validator"] = "success"
        return state
    except Exception as exc:
        logger.exception("Validator failed: %s", exc)
        state["node_status"] = state.get("node_status", {})
        state["node_status"]["validator"] = "failure"
        state["error"] = str(exc)
        return state