"""
ManufacturingIQ Agentic AI - Knowledge Retrieval Agent  (H-3)

Retrieves relevant knowledge passages for the current prediction using
context-driven query construction and hybrid retrieval.

H-3 improvements:
  - Query is built from the most informative prediction signals rather than
    raw numeric values: status label, active failure modes, and SHAP top
    contributors.  This gives the semantic model a much more relevant query.
  - Citation strings from corpus docs are preserved on RetrievedDocument.
  - Tags are preserved for downstream keyword-based filtering.
  - `top_k` increased to 4 to give report agent more evidence to draw from.
"""

import logging
from typing import Any, Dict, List

from retriever.retriever import retriever
from state.schema import RetrievedDocument
from agents._utils import record_agent_error  # H-5

logger = logging.getLogger(__name__)


def _build_query(state: Dict[str, Any]) -> str:
    """
    Construct a semantically informative retrieval query from prediction context.

    Priority order:
      1. Machine status + risk level (most discriminating)
      2. Inferred failure mode hints (from engineered feature values)
      3. Top SHAP contributors (if available from prior explanation_agent run)
      4. Machine type
    """
    prediction = state.get("prediction") or {}
    raw = state.get("raw_input") or {}
    shap_exp = state.get("shap_explanation") or {}
    eng_feats = state.get("engineered_features") or {}

    parts: List[str] = []

    # --- Status and risk ---
    status = prediction.get("machine_status", "")
    risk = prediction.get("risk_level", "")
    prob = prediction.get("failure_probability", 0.0)
    if status:
        parts.append(f"{status} machine")
    if risk:
        parts.append(f"{risk.lower()} risk")

    # --- Infer likely failure modes from feature values ---
    tool_wear = raw.get("Tool_wear_min", 0.0)
    torque = raw.get("Torque_Nm", 0.0)
    rpm = raw.get("Rotational_speed_rpm", 1500.0)
    air_temp = raw.get("Air_temperature_K", 300.0)
    proc_temp = raw.get("Process_temperature_K", 310.0)
    temp_diff = proc_temp - air_temp

    failure_mode_hints: List[str] = []
    if tool_wear > 180:
        failure_mode_hints.append("tool wear failure")
    if temp_diff < 9.0 and rpm < 1400:
        failure_mode_hints.append("heat dissipation failure thermal")
    if torque * (rpm * 3.14159 / 30) < 3600 or torque * (rpm * 3.14159 / 30) > 8800:
        failure_mode_hints.append("power failure electrical")
    wear_intensity = tool_wear * torque
    machine_type = raw.get("Type", "L")
    osf_thresholds = {"H": 11000, "M": 12000, "L": 13000}
    if wear_intensity > osf_thresholds.get(machine_type, 13000) * 0.85:
        failure_mode_hints.append("overstrain mechanical stress bearing")

    if failure_mode_hints:
        parts.extend(failure_mode_hints)
    elif status in ("Warning", "Critical", "High Risk"):
        parts.append("predictive maintenance inspection")

    # --- SHAP top contributors (best signal if available) ---
    top_contributors = shap_exp.get("top_contributors") or []
    if top_contributors:
        # Map feature names to human-readable terms the corpus covers
        contributor_terms = _map_features_to_terms(top_contributors[:3])
        if contributor_terms:
            parts.append("feature drivers: " + ", ".join(contributor_terms))

    # --- Machine type ---
    type_names = {"L": "low-capacity", "M": "medium-capacity", "H": "high-capacity"}
    type_label = type_names.get(machine_type, "")
    if type_label:
        parts.append(f"{type_label} machine type {machine_type}")

    query = ". ".join(parts) if parts else "predictive maintenance machine failure inspection"
    logger.debug("Retrieval query: %r", query)
    return query


def _map_features_to_terms(feature_names: List[str]) -> List[str]:
    """Map raw feature column names to corpus-searchable terms."""
    mapping = {
        "Torque [Nm]":              "torque mechanical stress",
        "Tool wear [min]":          "tool wear degradation",
        "Rotational speed [rpm]":   "rotational speed RPM bearing",
        "Process temperature [K]":  "process temperature thermal",
        "Air temperature [K]":      "air temperature cooling",
        "machine_stress_index":     "machine stress index overstrain",
        "thermal_risk_index":       "thermal risk heat dissipation",
        "wear_intensity":           "wear intensity overstrain tool",
        "wear_efficiency_index":    "wear efficiency degradation",
        "torque_speed_ratio":       "torque speed ratio bearing load",
        "temperature_difference":   "temperature difference heat dissipation",
    }
    return [mapping[f] for f in feature_names if f in mapping]


def _doc_to_retrieved(d: Any) -> RetrievedDocument:
    """Convert a retriever result (RetrievedDocument TypedDict or object) to a state dict."""
    if isinstance(d, dict):
        return RetrievedDocument(
            title=d.get("title", ""),
            source=d.get("source", ""),
            section=d.get("section"),
            excerpt=d.get("excerpt", ""),
            confidence=round(float(d.get("confidence", 0.0)), 3),
            citation=d.get("citation"),
            tags=d.get("tags", []),
        )
    # Fallback for object-style (should not occur after C-1 fix, but be defensive)
    return RetrievedDocument(
        title=getattr(d, "title", ""),
        source=getattr(d, "source", ""),
        section=getattr(d, "section", None),
        excerpt=getattr(d, "excerpt", ""),
        confidence=round(float(getattr(d, "confidence", 0.0)), 3),
        citation=getattr(d, "citation", None),
        tags=getattr(d, "tags", []),
    )


def run_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        query = _build_query(state)
        docs = retriever.retrieve(query, top_k=4)
        state["retrieved_documents"] = [_doc_to_retrieved(d) for d in docs]
        logger.info(
            "Retrieved %d documents (query: %r)",
            len(state["retrieved_documents"]),
            query[:60],
        )
    except Exception as exc:
        logger.exception("Retrieval agent failed: %s", exc)
        record_agent_error(state, "node_retrieval", exc)  # H-5
        state["retrieved_documents"] = []
    return state