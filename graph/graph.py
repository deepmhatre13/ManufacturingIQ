"""
ManufacturingIQ Agentic AI v3 - Production-Grade LangGraph Workflow

Parallel conditional graph with supervisor, retries, and execution tracing.

Node names are prefixed with 'node_' to avoid clashing with ManufacturingIQState
key names in newer LangGraph versions (which enforce this uniqueness constraint).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from state.schema import ManufacturingIQState
from app.scoring import CONFIDENCE_THRESHOLD_PCT  # H-2
from agents.prediction_agent import run_prediction
from agents.supervisor_agent import run_supervisor
from agents.explanation_agent import run_explanation
from agents.retrieval_agent import run_retrieval
from agents.maintenance_agent import run_maintenance
from agents.risk_agent import run_risk
from agents.trend_agent import run_trend
from agents.operational_impact_agent import run_operational_impact
from agents.decision_validator_agent import run_decision_validator
from agents.report_agent import run_report

logger = logging.getLogger(__name__)

_graph = None

MAX_RETRIES = 2


def _retryable_node(node_name: str, fn, max_retries: int = MAX_RETRIES):
    """Wrap a node with retry logic."""
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        attempts = 0
        while attempts <= max_retries:
            try:
                result = fn(state)
                if result.get("error"):
                    attempts += 1
                    if attempts > max_retries:
                        _log_node_failure(state, node_name, result.get("error"))
                        return result
                    continue
                _log_node_success(state, node_name)
                return result
            except Exception as exc:
                attempts += 1
                if attempts > max_retries:
                    _log_node_failure(state, node_name, str(exc))
                    state["error"] = str(exc)
                    return state
    return wrapper


def _log_node_success(state: Dict[str, Any], node: str) -> None:
    logs = state.setdefault("execution_logs", [])
    logs.append({
        "node": node,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    state["node_status"] = state.get("node_status", {})
    state["node_status"][node] = "success"


def _log_node_failure(state: Dict[str, Any], node: str, error: str) -> None:
    logs = state.setdefault("execution_logs", [])
    logs.append({
        "node": node,
        "status": "failure",
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    state["node_status"] = state.get("node_status", {})
    state["node_status"][node] = "failure"


def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        logger.error("LangGraph is required: %s", exc)
        raise

    builder = StateGraph(ManufacturingIQState)

    # Prediction is wrapped with retry — it's the critical node; failure here means
    # no downstream processing is possible.  All other nodes degrade gracefully.
    builder.add_node("node_prediction", _retryable_node("node_prediction", run_prediction))
    builder.add_node("node_supervisor", run_supervisor)
    builder.add_node("node_explanation", run_explanation)
    builder.add_node("node_retrieval", run_retrieval)
    builder.add_node("node_maintenance", run_maintenance)
    builder.add_node("node_risk", run_risk)
    builder.add_node("node_trend", run_trend)
    builder.add_node("node_operational_impact", run_operational_impact)
    builder.add_node("node_validator", run_decision_validator)
    builder.add_node("node_report", run_report)

    # Entry: prediction first, then supervisor decides
    builder.set_entry_point("node_prediction")
    builder.add_edge("node_prediction", "node_supervisor")

    # Supervisor conditional edges
    builder.add_conditional_edges(
        "node_supervisor",
        _route_from_supervisor,
        {
            "parallel_agents": "node_explanation",
            "human_review": "node_report",
            "retry_retrieval": "node_retrieval",
            "report": "node_report",
        },
    )

    # Sequential pipeline: explanation → retrieval → maintenance → risk → trend → operational_impact → validator → report
    builder.add_edge("node_explanation", "node_retrieval")
    builder.add_edge("node_retrieval", "node_maintenance")
    builder.add_edge("node_maintenance", "node_risk")
    builder.add_edge("node_risk", "node_trend")
    builder.add_edge("node_trend", "node_operational_impact")
    builder.add_edge("node_operational_impact", "node_validator")
    builder.add_edge("node_validator", "node_report")
    builder.add_edge("node_report", END)

    return builder.compile()


def _route_from_supervisor(state: Dict[str, Any]) -> str:
    try:
        next_step = state.get("next")
        if next_step:
            return next_step
        prediction = state.get("prediction")
        if prediction and prediction.get("confidence", 0) < CONFIDENCE_THRESHOLD_PCT:
            return "human_review"
        return "parallel_agents"
    except Exception:
        return "report"


def run_graph(raw_input: Dict[str, Any], history: list[dict]) -> Dict[str, Any]:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    initial_state = {
        "raw_input": raw_input,
        "prediction_history": history,
        "retrieved_documents": [],
        "maintenance_recommendations": [],
        "execution_logs": [],
        "node_status": {},
    }
    final_state = _graph.invoke(initial_state)
    return dict(final_state)