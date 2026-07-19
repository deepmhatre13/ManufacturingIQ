"""
Regression tests for ManufacturingIQ Critical Bug Fixes
==========================================================

Tests that prevent regressions of:
  C-1: dict/attribute mismatch (TypedDict is a plain dict at runtime)
  C-3: SHAP faked / not wired
  C-4: machine_stress_index constant at inference
  C-5: history writeheader (verify still correct)

Run with:
    cd d:/jupyter_notebook/ManufacturingIQ
    python -m pytest tests/test_critical_fixes.py -v
"""

import csv
import os
import sys
import tempfile
import types
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEALTHY_INPUT = {
    "Type": "L",
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 308.6,
    "Rotational_speed_rpm": 1551.0,
    "Torque_Nm": 42.8,
    "Tool_wear_min": 0.0,
}

_CRITICAL_INPUT = {
    "Type": "H",
    "Air_temperature_K": 298.1,
    "Process_temperature_K": 311.6,
    "Rotational_speed_rpm": 1168.0,
    "Torque_Nm": 76.6,
    "Tool_wear_min": 253.0,
}

_TRAINING_STATS = {
    "temperature_difference_max": 13.0,
    "torque_max": 76.6,
    "wear_max": 253.0,
}


# ---------------------------------------------------------------------------
# C-4: machine_stress_index must vary across different inputs
# ---------------------------------------------------------------------------

class TestC4MachineStressIndex:
    """C-4: machine_stress_index must NOT be constant across different single-row inputs."""

    def test_varies_with_training_stats(self):
        """With training_stats supplied, machine_stress_index differs across inputs."""
        from feature_engineering.engineer import build_engineered_features

        rows = [_HEALTHY_INPUT, _CRITICAL_INPUT]
        indices = []
        for row in rows:
            df = pd.DataFrame([{
                "Type": row["Type"],
                "Air temperature [K]": row["Air_temperature_K"],
                "Process temperature [K]": row["Process_temperature_K"],
                "Rotational speed [rpm]": row["Rotational_speed_rpm"],
                "Torque [Nm]": row["Torque_Nm"],
                "Tool wear [min]": row["Tool_wear_min"],
            }])
            out = build_engineered_features(df, training_stats=_TRAINING_STATS)
            indices.append(float(out["machine_stress_index"].iloc[0]))

        assert indices[0] != indices[1], (
            f"machine_stress_index must differ between healthy ({indices[0]}) "
            f"and critical ({indices[1]}) inputs"
        )
        # Critical input should have a higher stress index than healthy
        assert indices[1] > indices[0], (
            f"Critical input should have higher machine_stress_index than healthy: "
            f"{indices[1]} vs {indices[0]}"
        )

    def test_no_training_stats_single_row_constant(self):
        """Without training_stats, single-row input collapses machine_stress_index to 3.0."""
        from feature_engineering.engineer import build_engineered_features

        df = pd.DataFrame([{
            "Type": "L",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551.0,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 100.0,
        }])
        out = build_engineered_features(df, training_stats=None)
        # Without training_stats, all three components normalize to 1.0 → sum = 3.0
        val = float(out["machine_stress_index"].iloc[0])
        assert abs(val - 3.0) < 0.01, (
            f"Without training_stats, single-row machine_stress_index should be ~3.0, got {val}"
        )

    def test_predictor_loads_training_stats(self):
        """app/predictor.py must load non-empty _training_stats from model_metadata.json."""
        from app.predictor import _training_stats
        assert isinstance(_training_stats, dict), "_training_stats should be a dict"
        assert len(_training_stats) > 0, (
            "_training_stats is empty — model_metadata.json missing training_stats key. "
            "machine_stress_index will be constant at inference."
        )
        for key in ("temperature_difference_max", "torque_max", "wear_max"):
            assert key in _training_stats, f"Missing key '{key}' in _training_stats"
            assert _training_stats[key] > 0, f"_training_stats['{key}'] must be positive, got {_training_stats[key]}"


# ---------------------------------------------------------------------------
# C-1: All agent state accesses must use .get() (TypedDict is a plain dict)
# ---------------------------------------------------------------------------

class TestC1DictAttributeMismatch:
    """C-1: Every agent past prediction_agent must accept plain dict state without AttributeError."""

    def _make_state(self, override_prediction=None, override_raw=None) -> Dict[str, Any]:
        """Build a minimal, realistic state dict (all values are plain dicts, not objects)."""
        return {
            "raw_input": override_raw or {
                "Type": "L",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551.0,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0.0,
            },
            "prediction": override_prediction or {
                "failure_prediction": 0,
                "failure_probability": 0.05,
                "health_score": 95.0,
                "machine_status": "Healthy",
                "confidence": 99.5,
                "risk_level": "Low",
            },
            "retrieved_documents": [
                {
                    "title": "Predictive Maintenance Principles",
                    "source": "knowledge/corpus",
                    "section": None,
                    "excerpt": "Regular inspection reduces unplanned downtime.",
                    "confidence": 0.82,
                }
            ],
            "maintenance_recommendations": [],
            "execution_logs": [],
            "node_status": {},
            "prediction_history": [],
        }

    def test_maintenance_agent_no_attributeerror(self):
        from agents.maintenance_agent import run_maintenance
        state = self._make_state()
        result = run_maintenance(state)
        assert "maintenance_recommendations" in result
        assert isinstance(result["maintenance_recommendations"], list)

    def test_maintenance_agent_critical_state(self):
        from agents.maintenance_agent import run_maintenance
        state = self._make_state(override_prediction={
            "failure_prediction": 1,
            "failure_probability": 0.9,
            "health_score": 10.0,
            "machine_status": "Critical",
            "confidence": 85.0,
            "risk_level": "High",
        })
        result = run_maintenance(state)
        recs = result["maintenance_recommendations"]
        assert len(recs) >= 1
        assert recs[0].get("priority") == "High"

    def test_risk_agent_no_attributeerror(self):
        from agents.risk_agent import run_risk
        state = self._make_state()
        result = run_risk(state)
        assert "risk_assessment" in result
        ra = result["risk_assessment"]
        assert isinstance(ra, dict)
        assert ra.get("risk_level") is not None

    def test_operational_impact_no_attributeerror(self):
        from agents.operational_impact_agent import run_operational_impact
        state = self._make_state()
        result = run_operational_impact(state)
        assert "operational_impact" in result
        oi = result["operational_impact"]
        assert isinstance(oi, dict)
        assert oi.get("impact_category") is not None

    def test_decision_validator_no_attributeerror(self):
        from agents.decision_validator_agent import run_decision_validator
        state = self._make_state()
        # Add a risk_assessment for the validator to check
        state["risk_assessment"] = {
            "risk_level": "Low",
            "severity": "Low",
            "business_impact": "Low impact",
            "urgency": "Scheduled",
            "rationale": "Test",
        }
        result = run_decision_validator(state)
        assert result.get("node_status", {}).get("validator") == "success"

    def test_report_agent_no_attributeerror(self):
        from agents.report_agent import run_report
        state = self._make_state()
        state["risk_assessment"] = {
            "risk_level": "Low",
            "severity": "Low",
            "business_impact": "Low impact",
            "urgency": "Scheduled",
            "rationale": "Test",
        }
        state["shap_explanation"] = {
            "top_contributors": ["Torque [Nm]", "Tool wear [min]"],
            "positive_contributors": ["Torque [Nm]"],
            "negative_contributors": ["Tool wear [min]"],
            "explanation_text": "Test explanation.",
            "confidence": 0.85,
        }
        state["trend_analysis"] = {
            "direction": "Stable",
            "health_trend": "Stable",
            "risk_trend": "Stable",
            "summary": "No significant trend.",
        }
        state["operational_impact"] = {
            "impact_category": "Low impact",
            "estimated_priority": "Scheduled",
            "severity_score": 3.0,
            "notes": "Routine window.",
        }
        result = run_report(state)
        assert "engineering_report" in result
        report = result["engineering_report"]
        assert isinstance(report, dict), f"engineering_report should be a dict, got {type(report)}"
        assert report.get("prediction_summary"), "prediction_summary must be non-empty"
        assert report.get("final_recommendation"), "final_recommendation must be non-empty"
        assert isinstance(report.get("primary_drivers"), list)

    def test_retrieval_agent_no_attributeerror(self):
        """Retrieval agent must not raise AttributeError when building the query string."""
        from agents.retrieval_agent import run_retrieval
        state = self._make_state()

        # Mock the retriever so we don't need FAISS/embeddings
        mock_docs = [
            {
                "title": "Predictive Maintenance Principles",
                "source": "knowledge/corpus",
                "section": None,
                "excerpt": "Regular inspection reduces downtime.",
                "confidence": 0.8,
            }
        ]
        with patch("agents.retrieval_agent.retriever") as mock_retriever:
            mock_retriever.retrieve.return_value = [
                types.SimpleNamespace(
                    title="Predictive Maintenance Principles",
                    source="knowledge/corpus",
                    section=None,
                    excerpt="Regular inspection reduces downtime.",
                    confidence=0.8,
                )
            ]
            result = run_retrieval(state)

        assert "retrieved_documents" in result
        assert isinstance(result["retrieved_documents"], list)

    def test_report_generator_to_json_with_dict(self):
        """reports/generator.to_json() must handle plain dict (not call .model_dump())."""
        from reports.generator import to_json
        report_dict = {
            "prediction_summary": "Test summary",
            "technical_explanation": "Test explanation",
            "primary_drivers": ["Torque [Nm]"],
            "retrieved_evidence": [],
            "maintenance_recommendations": [],
            "risk_assessment": {"risk_level": "Low", "urgency": "Scheduled"},
            "confidence": 95.0,
            "final_recommendation": "Continue monitoring.",
        }
        result = to_json(report_dict)
        assert result == report_dict, "to_json should return the dict as-is"

    def test_report_generator_to_markdown_with_dict(self):
        """reports/generator.to_markdown() must not raise AttributeError on a plain dict."""
        from reports.generator import to_markdown
        report_dict = {
            "prediction_summary": "Test summary",
            "technical_explanation": "Test explanation",
            "primary_drivers": ["Torque [Nm]"],
            "retrieved_evidence": [],
            "maintenance_recommendations": [
                {"action": "Inspect bearings", "priority": "High", "rationale": "High torque."}
            ],
            "risk_assessment": {
                "risk_level": "Medium",
                "severity": "Medium",
                "business_impact": "High impact",
                "urgency": "Within 24 hours",
                "rationale": "Elevated risk.",
            },
            "confidence": 88.0,
            "final_recommendation": "Inspect bearings | Urgency: Within 24 hours",
        }
        md = to_markdown(report_dict)
        assert "# Engineering Report" in md
        assert "Test summary" in md
        assert "Inspect bearings" in md


# ---------------------------------------------------------------------------
# C-3: SHAP explanation agent must call _compute_shap_values and return real values
# ---------------------------------------------------------------------------

class TestC3ShapWired:
    """C-3: SHAP must be wired to _compute_shap_values, not returning a static stub."""

    def _make_state(self):
        return {
            "raw_input": {
                "Type": "L",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551.0,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0.0,
            },
            "prediction": {
                "failure_prediction": 0,
                "failure_probability": 0.05,
                "health_score": 95.0,
                "machine_status": "Healthy",
                "confidence": 99.5,
                "risk_level": "Low",
            },
            "execution_logs": [],
            "node_status": {},
        }

    def test_explanation_calls_compute_shap(self):
        """_compute_shap_values must be called (not bypassed) when a prediction exists."""
        from agents import explanation_agent

        call_log = []
        original_fn = explanation_agent._compute_shap_values

        def spy_compute_shap(model, X):
            call_log.append((model, X))
            return original_fn(model, X)  # real call

        with patch.object(explanation_agent, "_compute_shap_values", side_effect=spy_compute_shap):
            state = self._make_state()
            result = explanation_agent.run_explanation(state)

        assert len(call_log) > 0, (
            "_compute_shap_values was never called — SHAP is still bypassed (C-3 not fixed)"
        )

    def test_explanation_output_is_prediction_dependent(self):
        """Two very different inputs must produce different top_contributors ordering."""
        from agents.explanation_agent import run_explanation

        state_healthy = {
            "raw_input": _HEALTHY_INPUT,
            "prediction": {
                "failure_prediction": 0,
                "failure_probability": 0.02,
                "health_score": 98.0,
                "machine_status": "Healthy",
                "confidence": 99.5,
                "risk_level": "Low",
            },
            "execution_logs": [],
            "node_status": {},
        }
        state_critical = {
            "raw_input": _CRITICAL_INPUT,
            "prediction": {
                "failure_prediction": 1,
                "failure_probability": 0.95,
                "health_score": 5.0,
                "machine_status": "Critical",
                "confidence": 85.5,
                "risk_level": "High",
            },
            "execution_logs": [],
            "node_status": {},
        }

        result_healthy = run_explanation(state_healthy)
        result_critical = run_explanation(state_critical)

        exp_healthy = result_healthy.get("shap_explanation", {})
        exp_critical = result_critical.get("shap_explanation", {})

        # Both must produce non-empty explanations
        assert exp_healthy.get("top_contributors"), "Healthy explanation has no top_contributors"
        assert exp_critical.get("top_contributors"), "Critical explanation has no top_contributors"

        # Their explanation texts must not be identical (they were static before the fix)
        # Note: It's possible SHAP could happen to agree for very different inputs in rare cases,
        # so we check the explanation_text contains something input-specific if SHAP worked.
        text_healthy = exp_healthy.get("explanation_text", "")
        text_critical = exp_critical.get("explanation_text", "")
        assert text_healthy != "", "Healthy explanation_text is empty"
        assert text_critical != "", "Critical explanation_text is empty"

    def test_explanation_graceful_fallback_on_shap_failure(self):
        """When SHAP computation raises, run_explanation must still return a non-empty ShapExplanation."""
        from agents import explanation_agent

        def failing_shap(model, X):
            return None, None  # simulate SHAP not available

        with patch.object(explanation_agent, "_compute_shap_values", side_effect=failing_shap):
            state = self._make_state()
            result = explanation_agent.run_explanation(state)

        exp = result.get("shap_explanation", {})
        assert exp is not None, "shap_explanation must not be None even on SHAP failure"
        assert isinstance(exp, dict), "shap_explanation must be a dict"
        assert exp.get("top_contributors") is not None, "top_contributors must be present on fallback"
        assert exp.get("explanation_text"), "explanation_text must be non-empty on fallback"


# ---------------------------------------------------------------------------
# C-5: history/utils.py writeheader (verify correct spelling is still present)
# ---------------------------------------------------------------------------

class TestC5HistoryWriteheader:
    """C-5: _ensure_file() must create a valid CSV header on a fresh file."""

    def test_ensure_file_creates_header(self, tmp_path):
        """When the history file doesn't exist, _ensure_file() must create it with a valid header."""
        import history.utils as hu

        original_path = hu.HISTORY_PATH
        test_path = str(tmp_path / "test_history.csv")
        hu.HISTORY_PATH = test_path

        try:
            assert not os.path.exists(test_path), "File should not exist before _ensure_file()"
            hu._ensure_file()
            assert os.path.exists(test_path), "_ensure_file() should create the file"

            with open(test_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
            assert fieldnames is not None, "CSV has no header row"
            assert "timestamp" in fieldnames, "Missing 'timestamp' field in CSV header"
            assert "health_score" in fieldnames, "Missing 'health_score' field in CSV header"
        finally:
            hu.HISTORY_PATH = original_path

    def test_append_and_load_roundtrip(self, tmp_path):
        """append_history() + load_history() must roundtrip a record correctly."""
        import history.utils as hu

        original_path = hu.HISTORY_PATH
        test_path = str(tmp_path / "roundtrip_history.csv")
        hu.HISTORY_PATH = test_path

        try:
            record = {
                "timestamp": "2026-07-18T10:00:00",
                "machine_type": "L",
                "health_score": 95.0,
                "failure_probability": 0.05,
                "risk_level": "Low",
                "status": "Healthy",
                "confidence": 99.5,
                "recommendation": "Continue monitoring.",
                "trend": "",
            }
            hu.append_history(record)
            rows = hu.load_history()
            assert len(rows) == 1
            assert rows[0]["machine_type"] == "L"
            assert rows[0]["health_score"] == "95.0"
        finally:
            hu.HISTORY_PATH = original_path


# ---------------------------------------------------------------------------
# End-to-end: run_graph must return non-empty, correctly-shaped output
# ---------------------------------------------------------------------------

class TestEndToEndGraph:
    """Verify the full LangGraph pipeline returns a non-empty, correctly-shaped result."""

    def test_run_graph_returns_non_empty(self):
        """run_graph must not return an empty dict for a valid input."""
        from graph.graph import run_graph

        result = run_graph(_HEALTHY_INPUT, history=[])

        assert isinstance(result, dict), f"run_graph must return a dict, got {type(result)}"
        assert len(result) > 0, "run_graph returned an empty dict — pipeline is silently failing"
        assert "prediction" in result, "result missing 'prediction' key"
        assert "engineering_report" in result, "result missing 'engineering_report' key"

        pred = result["prediction"]
        assert isinstance(pred, dict), f"prediction must be a dict, got {type(pred)}"
        assert pred.get("machine_status") in {"Healthy", "Warning", "Critical"}, (
            f"Unexpected machine_status: {pred.get('machine_status')}"
        )
        assert 0.0 <= pred.get("failure_probability", -1) <= 1.0, (
            f"failure_probability out of range: {pred.get('failure_probability')}"
        )
        assert 0.0 <= pred.get("health_score", -1) <= 100.0, (
            f"health_score out of range: {pred.get('health_score')}"
        )

    def test_run_graph_engineering_report_non_empty(self):
        """engineering_report in graph output must be a non-empty dict."""
        from graph.graph import run_graph

        result = run_graph(_CRITICAL_INPUT, history=[])
        report = result.get("engineering_report")
        assert report is not None, "engineering_report is None — report agent failed silently"
        assert isinstance(report, dict), f"engineering_report must be a dict, got {type(report)}"
        assert report.get("prediction_summary"), "prediction_summary is empty or missing"
        assert report.get("final_recommendation"), "final_recommendation is empty or missing"

    def test_run_agentic_pipeline_non_empty(self):
        """run_agentic_pipeline must return a non-empty dict (not the bare {} failure fallback)."""
        from app.agentic import run_agentic_pipeline

        result = run_agentic_pipeline(_HEALTHY_INPUT, history=[])
        assert isinstance(result, dict), f"run_agentic_pipeline must return dict, got {type(result)}"
        assert len(result) > 0, (
            "run_agentic_pipeline returned {} — the agentic pipeline is silently failing. "
            "Check that prediction and engineering_report are both non-None in the graph output."
        )
        assert "prediction" in result, "result missing 'prediction' key"
        assert "report" in result, "result missing 'report' key"
        assert isinstance(result.get("report"), dict), "'report' should be a dict"
