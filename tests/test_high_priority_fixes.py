"""
Regression tests for ManufacturingIQ High-Priority Bug Fixes
=============================================================

Covers:
  H-1  — Unified health/status/risk scoring (app/scoring.py)
  H-2  — Real confidence signal (prediction margin, not fabricated formula)
  H-5  — Agent failure visibility (execution_logs / node_status)
  H-6  — Model integrity checksum

Run with:
    cd d:/jupyter_notebook/ManufacturingIQ
    python -m pytest tests/test_high_priority_fixes.py -v
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# H-1: Shared scoring — unified thresholds, no divergence
# ---------------------------------------------------------------------------

class TestH1UnifiedScoring:
    """H-1: calculate_health_and_status must be the single source of truth."""

    def test_healthy_range(self):
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.05)
        assert h == pytest.approx(95.0, abs=0.01)
        assert s == "Healthy"
        assert r == "Low"

    def test_warning_range(self):
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.35)
        assert h == pytest.approx(65.0, abs=0.01)
        assert s == "Warning"
        assert r == "Medium"

    def test_high_risk_range(self):
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.70)
        assert h == pytest.approx(30.0, abs=0.01)
        assert s == "High Risk"
        assert r == "High"

    def test_critical_range(self):
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.95)
        assert h == pytest.approx(5.0, abs=0.01)
        assert s == "Critical"
        assert r == "High"

    def test_boundary_exactly_80(self):
        """health == 80.0 should be Healthy (≥ 80 threshold is inclusive)."""
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.20)
        assert h == pytest.approx(80.0, abs=0.01)
        assert s == "Healthy"

    def test_boundary_just_below_80(self):
        from app.scoring import calculate_health_and_status
        h, s, r = calculate_health_and_status(0.201)
        assert s == "Warning"

    def test_probability_clamped_to_0_100(self):
        from app.scoring import calculate_health_and_status
        h, _, _ = calculate_health_and_status(1.5)   # over 100%
        assert h == 0.0
        h2, _, _ = calculate_health_and_status(-0.1)  # negative
        assert h2 == 100.0

    def test_predictor_calculate_health_shim_consistent(self):
        """app.predictor.calculate_health (shim) must agree with app.scoring."""
        from app.predictor import calculate_health
        from app.scoring import calculate_health_and_status

        for prob in [0.05, 0.25, 0.55, 0.75, 0.95]:
            shim_h, shim_s = calculate_health(prob)
            canon_h, canon_s, _ = calculate_health_and_status(prob)
            assert shim_h == canon_h, (
                f"p={prob}: predictor.calculate_health returned {shim_h} "
                f"but app.scoring returned {canon_h}"
            )
            assert shim_s == canon_s, (
                f"p={prob}: predictor status {shim_s!r} != scoring status {canon_s!r}"
            )

    def test_prediction_agent_uses_shared_scoring(self):
        """prediction_agent must produce health/status that matches app.scoring."""
        from agents.prediction_agent import run_prediction
        from app.scoring import calculate_health_and_status

        state = {
            "raw_input": {
                "Type": "L",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551.0,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0.0,
            },
            "execution_logs": [],
            "node_status": {},
        }
        result = run_prediction(state)
        pred = result.get("prediction", {})
        prob = pred.get("failure_probability")
        assert prob is not None

        canon_h, canon_s, canon_r = calculate_health_and_status(prob)
        assert pred["health_score"] == canon_h, (
            f"Agent health {pred['health_score']} != canonical {canon_h} for p={prob}"
        )
        assert pred["machine_status"] == canon_s, (
            f"Agent status {pred['machine_status']!r} != canonical {canon_s!r}"
        )
        assert pred["risk_level"] == canon_r, (
            f"Agent risk {pred['risk_level']!r} != canonical {canon_r!r}"
        )


# ---------------------------------------------------------------------------
# H-2: Real confidence signal
# ---------------------------------------------------------------------------

class TestH2RealConfidence:
    """H-2: confidence must be a prediction-margin signal, not the fabricated formula."""

    def test_max_confidence_at_boundary(self):
        """p = 0 or 1 → confidence = 100 (maximally certain)."""
        from app.scoring import calculate_confidence
        assert calculate_confidence(0.0) == pytest.approx(100.0)
        assert calculate_confidence(1.0) == pytest.approx(100.0)

    def test_min_confidence_at_decision_boundary(self):
        """p = 0.5 → confidence = 0 (maximally uncertain)."""
        from app.scoring import calculate_confidence
        assert calculate_confidence(0.5) == pytest.approx(0.0)

    def test_confidence_monotonic_from_boundary(self):
        """Confidence should increase as probability moves away from 0.5."""
        from app.scoring import calculate_confidence
        probs = [0.5, 0.4, 0.3, 0.1, 0.0]
        confs = [calculate_confidence(p) for p in probs]
        for i in range(len(confs) - 1):
            assert confs[i] <= confs[i + 1], (
                f"Confidence not monotonic: p={probs[i]}→{confs[i]}, "
                f"p={probs[i+1]}→{confs[i+1]}"
            )

    def test_confidence_not_fabricated(self):
        """Confidence must NOT follow the old `85 + health * 0.15` formula."""
        from app.scoring import calculate_confidence, calculate_health_and_status

        for prob in [0.05, 0.3, 0.7, 0.95]:
            health, _, _ = calculate_health_and_status(prob)
            old_formula = min(99.5, 85 + health * 0.15)
            real_conf = calculate_confidence(prob)
            assert real_conf != pytest.approx(old_formula, abs=0.5), (
                f"p={prob}: confidence {real_conf} matches the old fabricated formula "
                f"{old_formula} — H-2 fix may not have applied"
            )

    def test_prediction_agent_confidence_is_real(self):
        """Confidence stored by prediction_agent must match calculate_confidence."""
        from agents.prediction_agent import run_prediction
        from app.scoring import calculate_confidence

        state = {
            "raw_input": {
                "Type": "L",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551.0,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0.0,
            },
            "execution_logs": [],
            "node_status": {},
        }
        result = run_prediction(state)
        pred = result.get("prediction", {})
        prob = pred.get("failure_probability")
        stored_conf = pred.get("confidence")

        expected_conf = calculate_confidence(prob)
        assert stored_conf == pytest.approx(expected_conf, abs=0.01), (
            f"Agent stored confidence {stored_conf} != "
            f"calculate_confidence({prob}) = {expected_conf}"
        )

    def test_supervisor_threshold_is_70(self):
        """CONFIDENCE_THRESHOLD_PCT must equal 70 (the 0–100 scale equivalent)."""
        from app.scoring import CONFIDENCE_THRESHOLD_PCT
        assert CONFIDENCE_THRESHOLD_PCT == pytest.approx(70.0), (
            f"CONFIDENCE_THRESHOLD_PCT is {CONFIDENCE_THRESHOLD_PCT}, expected 70.0"
        )

    def test_supervisor_routes_low_confidence_to_human_review(self):
        """Supervisor must route to human_review when confidence < 70."""
        from agents.supervisor_agent import run_supervisor

        # p ≈ 0.5 → confidence ≈ 0 → low confidence
        state = {
            "prediction": {
                "failure_prediction": 0,
                "failure_probability": 0.48,
                "health_score": 52.0,
                "machine_status": "Warning",
                "confidence": 4.0,   # |0.48-0.5|/0.5*100 = 4
                "risk_level": "Medium",
            },
            "execution_logs": [],
            "node_status": {},
        }
        result = run_supervisor(state)
        assert result.get("next") == "human_review", (
            f"Expected human_review for low-confidence prediction, got {result.get('next')!r}"
        )

    def test_supervisor_routes_high_confidence_to_parallel_agents(self):
        """Supervisor must route to parallel_agents when confidence ≥ 70."""
        from agents.supervisor_agent import run_supervisor

        # p ≈ 0.02 → confidence ≈ 96 → high confidence
        state = {
            "prediction": {
                "failure_prediction": 0,
                "failure_probability": 0.02,
                "health_score": 98.0,
                "machine_status": "Healthy",
                "confidence": 96.0,
                "risk_level": "Low",
            },
            "execution_logs": [],
            "node_status": {},
        }
        result = run_supervisor(state)
        assert result.get("next") == "parallel_agents", (
            f"Expected parallel_agents for high-confidence prediction, got {result.get('next')!r}"
        )


# ---------------------------------------------------------------------------
# H-5: Agent failure visibility
# ---------------------------------------------------------------------------

class TestH5AgentFailureVisibility:
    """H-5: When agents fail, execution_logs and node_status must record the failure."""

    def _base_state(self) -> Dict[str, Any]:
        return {
            "raw_input": {"Type": "L", "Air_temperature_K": 298.0,
                          "Process_temperature_K": 308.0, "Rotational_speed_rpm": 1500.0,
                          "Torque_Nm": 40.0, "Tool_wear_min": 0.0},
            "prediction": {"failure_probability": 0.1, "health_score": 90.0,
                           "machine_status": "Healthy", "confidence": 80.0, "risk_level": "Low"},
            "retrieved_documents": [],
            "maintenance_recommendations": [],
            "execution_logs": [],
            "node_status": {},
        }

    def _check_failure_recorded(self, state, node_name):
        assert state["node_status"].get(node_name) == "failure", (
            f"node_status['{node_name}'] not set to 'failure'. Got: {state['node_status']}"
        )
        error_entries = [e for e in state.get("execution_logs", [])
                         if e.get("node") == node_name and e.get("status") == "failure"]
        assert len(error_entries) >= 1, (
            f"No failure log entry for '{node_name}' in execution_logs. "
            f"Got: {state.get('execution_logs')}"
        )
        assert error_entries[0].get("error"), "Failure log entry has no 'error' message"

    def test_retrieval_failure_visible(self):
        from agents.retrieval_agent import run_retrieval
        state = self._base_state()
        with patch("agents.retrieval_agent.retriever") as mock:
            mock.retrieve.side_effect = RuntimeError("FAISS index corrupt")
            result = run_retrieval(state)
        assert result["retrieved_documents"] == [], "Should gracefully return empty list"
        self._check_failure_recorded(result, "node_retrieval")

    def test_maintenance_failure_visible(self):
        from agents import maintenance_agent
        state = self._base_state()
        with patch.object(maintenance_agent, "_to_ref", side_effect=RuntimeError("ref error")):
            # Force a failure by breaking the internal helper
            state["prediction"]["machine_status"] = "Critical"
            state["prediction"]["risk_level"] = "High"
            state["retrieved_documents"] = [
                {"title": "Thermal Failure Mode", "source": "x", "section": None,
                 "excerpt": "y", "confidence": 0.9}
            ]
            result = maintenance_agent.run_maintenance(state)
        assert result["maintenance_recommendations"] == []
        self._check_failure_recorded(result, "node_maintenance")

    def test_risk_failure_visible(self):
        """When run_risk raises in the main branch, failure must appear in logs."""
        from agents import risk_agent
        state = self._base_state()

        # Force a failure inside the try-block (not the except fallback) by
        # patching the internal state write to raise. The except-block fallback
        # calls RiskAssessment() with no args which is fine — only the first
        # (keyword-arg) call in the main path raises.
        call_count = {"n": 0}
        original = risk_agent.RiskAssessment

        def side_effect_once(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1 and kwargs:  # first call with kwargs → main branch
                raise RuntimeError("schema error")
            return original(*args, **kwargs)   # subsequent/fallback calls succeed

        with patch.object(risk_agent, "RiskAssessment", side_effect=side_effect_once):
            state["prediction"]["risk_level"] = "High"  # ensures main branch is taken
            result = risk_agent.run_risk(state)

        self._check_failure_recorded(result, "node_risk")

    def test_record_agent_error_helper(self):
        """record_agent_error must update execution_logs and node_status."""
        from agents._utils import record_agent_error
        state: Dict[str, Any] = {"execution_logs": [], "node_status": {}}
        record_agent_error(state, "node_test", ValueError("something broke"))
        assert state["node_status"]["node_test"] == "failure"
        assert len(state["execution_logs"]) == 1
        entry = state["execution_logs"][0]
        assert entry["node"] == "node_test"
        assert entry["status"] == "failure"
        assert "ValueError" in entry["error"]

    def test_record_agent_error_does_not_set_global_error(self):
        """record_agent_error must NOT set state['error'] (reserved for retry trigger)."""
        from agents._utils import record_agent_error
        state: Dict[str, Any] = {"execution_logs": [], "node_status": {}}
        record_agent_error(state, "node_retrieval", RuntimeError("oops"))
        assert "error" not in state, (
            "record_agent_error set state['error'] — this would trigger retry "
            "on graceful-degradation nodes, breaking the degradation contract"
        )


# ---------------------------------------------------------------------------
# H-6: Model integrity check
# ---------------------------------------------------------------------------

class TestH6ModelIntegrity:
    """H-6: Model must be verified against sha256 in model_metadata.json before loading."""

    def test_metadata_has_sha256(self):
        metadata_path = REPO_ROOT / "models" / "model_metadata.json"
        with open(metadata_path) as f:
            meta = json.load(f)
        assert "sha256" in meta, (
            "model_metadata.json missing 'sha256' key — integrity check cannot run"
        )
        assert len(meta["sha256"]) == 64, (
            f"sha256 looks wrong (length {len(meta['sha256'])}); expected 64-char hex"
        )

    def test_model_checksum_matches_metadata(self):
        metadata_path = REPO_ROOT / "models" / "model_metadata.json"
        model_path = REPO_ROOT / "models" / "production_model.pkl"
        with open(metadata_path) as f:
            meta = json.load(f)
        expected = meta.get("sha256", "")
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"SHA-256 mismatch!\n  metadata: {expected}\n  actual:   {actual}\n"
            "The model file may have changed since metadata was recorded. "
            "Update 'sha256' in model_metadata.json."
        )

    def test_verify_model_checksum_passes_on_correct_file(self):
        """_verify_model_checksum must not raise when sha256 is correct."""
        from app.predictor import _verify_model_checksum
        model_path = REPO_ROOT / "models" / "production_model.pkl"
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        # Should not raise
        _verify_model_checksum(model_path, {"sha256": actual})

    def test_verify_model_checksum_raises_on_wrong_hash(self, tmp_path):
        """_verify_model_checksum must raise RuntimeError on hash mismatch."""
        from app.predictor import _verify_model_checksum
        fake_model = tmp_path / "fake_model.pkl"
        fake_model.write_bytes(b"tampered content")
        with pytest.raises(RuntimeError, match="integrity check FAILED"):
            _verify_model_checksum(fake_model, {"sha256": "a" * 64})

    def test_verify_model_checksum_skips_when_no_sha256(self, caplog):
        """_verify_model_checksum must only warn (not raise) when sha256 key is absent."""
        import logging
        from app.predictor import _verify_model_checksum
        model_path = REPO_ROOT / "models" / "production_model.pkl"
        with caplog.at_level(logging.WARNING, logger="app.predictor"):
            _verify_model_checksum(model_path, {})  # no sha256 key → should only warn
        assert any("skipping integrity check" in r.message for r in caplog.records), (
            "Expected a warning when sha256 is absent, but none was emitted"
        )
