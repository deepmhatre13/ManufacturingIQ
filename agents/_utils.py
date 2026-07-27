"""
ManufacturingIQ - Agent Error Recording Utility  (H-5)

Agents catch their own exceptions for graceful degradation.  Without this
helper those failures were invisible to the graph's execution_logs and
node_status, making it impossible to distinguish "ran successfully" from
"failed silently and returned a fallback".

Usage inside an agent's except block:

    from agents._utils import record_agent_error
    ...
    except Exception as exc:
        record_agent_error(state, "maintenance", exc)
        state["maintenance_recommendations"] = []
    return state

The helper is intentionally minimal — it never raises and never sets
state["error"] (which would trigger the graph-level retry wrapper and break
graceful degradation).  It only appends to execution_logs and node_status
so the dashboard and reports can surface the failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def record_agent_error(
    state: Dict[str, Any],
    node_name: str,
    exc: Exception,
) -> None:
    """
    Append a failure entry to ``state["execution_logs"]`` and mark
    ``state["node_status"][node_name]`` as ``"failure"``.

    Does NOT set ``state["error"]`` — that field is reserved for the
    graph-level retry wrapper (prediction_agent).  All other agents degrade
    gracefully; their failures must be *visible* but must not trigger retry.

    Parameters
    ----------
    state : dict
        Live LangGraph state dict.
    node_name : str
        Logical name of the failing agent/node (e.g. ``"maintenance"``).
    exc : Exception
        The caught exception.
    """
    try:
        error_msg = f"{type(exc).__name__}: {exc}"
        logs = state.setdefault("execution_logs", [])
        logs.append({
            "node": node_name,
            "status": "failure",
            "error": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        node_status = state.setdefault("node_status", {})
        node_status[node_name] = "failure"
        logger.error("[%s] agent error recorded in execution_logs: %s", node_name, error_msg)
    except Exception as inner:
        # Never let the error recorder itself crash the agent
        logger.error("record_agent_error helper itself failed: %s", inner)
