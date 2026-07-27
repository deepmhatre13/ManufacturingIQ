"""
Tests for the centralized feature engineering module.

Validates that:
- All 6 engineered features compute correctly
- Input validation works (missing columns raise ValueError)
- Edge cases (zero values) don't cause division by zero or inf/NaN
- Feature values match between the centralized module and the legacy notebook implementation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from feature_engineering.engineer import build_engineered_features


class TestFeatureEngineeringOutputs:
    """Test that engineered features match hand-computed expected values."""

    def test_temperature_difference(self):
        """Process - Air temperature."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        expected = 310.0 - 300.0
        assert np.isclose(result["temperature_difference"].iloc[0], expected)

    def test_torque_speed_ratio(self):
        """Torque / (Rotational speed + epsilon)."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        expected = 40.0 / (1500.0 + 1e-6)
        assert np.isclose(result["torque_speed_ratio"].iloc[0], expected)

    def test_wear_intensity(self):
        """Tool wear * Torque."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        expected = 10.0 * 40.0
        assert np.isclose(result["wear_intensity"].iloc[0], expected)

    def test_machine_stress_index(self):
        """Composite normalized index."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        temp_diff = result["temperature_difference"].iloc[0]
        expected = (temp_diff / temp_diff) + (40.0 / 40.0) + (10.0 / 10.0)
        assert np.isclose(result["machine_stress_index"].iloc[0], expected)

    def test_thermal_risk_index(self):
        """Process / (Air + epsilon)."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        expected = 310.0 / (300.0 + 1e-6)
        assert np.isclose(result["thermal_risk_index"].iloc[0], expected)

    def test_wear_efficiency_index(self):
        """Rotational speed / (Tool wear + 1)."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        expected = 1500.0 / (10.0 + 1)
        assert np.isclose(result["wear_efficiency_index"].iloc[0], expected)


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
        }])
        with pytest.raises(ValueError, match="Missing required columns"):
            build_engineered_features(df)

    def test_all_columns_present(self):
        """Valid input with all required columns should succeed."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        assert "temperature_difference" in result.columns
        assert "torque_speed_ratio" in result.columns
        assert "wear_intensity" in result.columns
        assert "machine_stress_index" in result.columns
        assert "thermal_risk_index" in result.columns
        assert "wear_efficiency_index" in result.columns


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_rotational_speed_no_inf(self):
        """Zero Rotational speed should not produce inf or NaN."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 0.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        assert np.isfinite(result["torque_speed_ratio"].iloc[0])
        assert np.isfinite(result["wear_efficiency_index"].iloc[0])

    def test_zero_tool_wear_no_inf(self):
        """Zero Tool wear should not produce inf or NaN."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 0.0,
        }])
        result = build_engineered_features(df)
        assert np.isfinite(result["wear_efficiency_index"].iloc[0])
        assert np.isfinite(result["wear_intensity"].iloc[0])

    def test_zero_air_temperature_no_inf(self):
        """Zero Air temperature should not produce inf or NaN."""
        df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 0.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 10.0,
        }])
        result = build_engineered_features(df)
        assert np.isfinite(result["thermal_risk_index"].iloc[0])

    def test_batch_processing(self):
        """Multiple rows should all compute correctly."""
        df = pd.DataFrame([
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 10.0,
            },
            {
                "Type": "H",
                "Air temperature [K]": 298.0,
                "Process temperature [K]": 320.0,
                "Rotational speed [rpm]": 1200.0,
                "Torque [Nm]": 80.0,
                "Tool wear [min]": 200.0,
            },
        ])
        result = build_engineered_features(df)
        assert len(result) == 2
        for col in [
            "temperature_difference",
            "torque_speed_ratio",
            "wear_intensity",
            "machine_stress_index",
            "thermal_risk_index",
            "wear_efficiency_index",
        ]:
            assert result[col].notna().all(), f"{col} contains NaN"
            assert np.isfinite(result[col]).all(), f"{col} contains non-finite values"


class TestFeatureEngineeringConsistency:
    """Test that the centralized module matches notebook's original formulas."""

    def _legacy_build_engineered_features(self, dataframe):
        """Legacy implementation from notebook cell 20-24."""
        engineered_df = dataframe.copy()
        engineered_df["temperature_difference"] = (
            engineered_df["Process temperature [K]"]
            - engineered_df["Air temperature [K]"]
        )
        engineered_df["torque_speed_ratio"] = (
            engineered_df["Torque [Nm]"]
            / engineered_df["Rotational speed [rpm]"]
        )
        engineered_df["wear_intensity"] = (
            engineered_df["Tool wear [min]"]
            * engineered_df["Torque [Nm]"]
        )
        return engineered_df

    def _legacy_build_inference_features(self, data):
        """Legacy inference implementation from notebook cell 74."""
        dataframe = pd.DataFrame([data])
        dataframe["temperature_difference"] = (
            dataframe["Process temperature [K]"]
            - dataframe["Air temperature [K]"]
        )
        dataframe["torque_speed_ratio"] = (
            dataframe["Torque [Nm]"]
            / dataframe["Rotational speed [rpm]"]
        )
        dataframe["wear_intensity"] = (
            dataframe["Tool wear [min]"]
            * dataframe["Torque [Nm]"]
        )
        dataframe["machine_stress_index"] = (
            (dataframe["temperature_difference"] / 100)
            + (dataframe["Torque [Nm]"] / 100)
            + (dataframe["Tool wear [min]"] / 100)
        )
        dataframe["thermal_risk_index"] = (
            dataframe["Process temperature [K]"]
            / dataframe["Air temperature [K]"]
        )
        dataframe["wear_efficiency_index"] = (
            dataframe["Rotational speed [rpm]"]
            / (dataframe["Tool wear [min]"] + 1)
        )
        return dataframe

    def test_centralized_matches_legacy_training(self):
        """Centralized module should match legacy training formulas (cells 20-24)."""
        df = pd.DataFrame([
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 10.0,
            },
            {
                "Type": "L",
                "Air temperature [K]": 295.0,
                "Process temperature [K]": 305.0,
                "Rotational speed [rpm]": 1400.0,
                "Torque [Nm]": 35.0,
                "Tool wear [min]": 5.0,
            },
        ])
        
        legacy = self._legacy_build_engineered_features(df.copy())
        centralized = build_engineered_features(df.copy())
        
        # Compare common features
        for col in ["temperature_difference", "torque_speed_ratio", "wear_intensity"]:
            assert np.allclose(legacy[col], centralized[col], rtol=1e-6), \
                f"Mismatch in {col} between legacy and centralized"

    def test_centralized_matches_legacy_inference(self):
        """Centralized inference wrapper should match legacy inference formulas."""
        sample = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 12,
        }
        
        legacy = self._legacy_build_inference_features(sample)
        df = pd.DataFrame([sample])
        centralized = build_engineered_features(df)
        
        # First three features should match exactly or very closely
        for col in [
            "temperature_difference",
            "torque_speed_ratio",
            "wear_intensity",
        ]:
            assert np.allclose(legacy[col], centralized[col], rtol=1e-6), \
                f"Mismatch in {col} between legacy and centralized inference"
        
        # machine_stress_index differs: legacy used hardcoded /100 divisors,
        # while canonical uses proper in-batch max normalization.
        # We assert the canonical value is numerically stable, not equal to legacy.
        assert np.isfinite(centralized["machine_stress_index"].iloc[0])
        
        # thermal_risk_index and wear_efficiency_index should match exactly
        for col in [
            "thermal_risk_index",
            "wear_efficiency_index",
        ]:
            assert np.allclose(legacy[col], centralized[col], rtol=1e-6), \
                f"Mismatch in {col} between legacy and centralized inference"


class TestPredictorIntegration:
    """Integration tests ensuring predictor.py produces identical outputs."""

    def test_predictor_matches_centralized_features(self):
        """predict_machine_failure via API and notebook inference should produce identical features."""
        from app.predictor import create_features
        
        sample = {
            "Type": "M",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 12,
        }
        
        predictor_df = create_features(sample)
        
        # Build expected using centralized module directly, with the same training_stats
        # that create_features uses (C-4 fix: predictor passes training_stats).
        from app.predictor import _training_stats
        expected_df = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 12,
        }])
        expected_df = build_engineered_features(expected_df, training_stats=_training_stats or None)
        
        # Compare feature values
        feature_cols = [
            "temperature_difference",
            "torque_speed_ratio",
            "wear_intensity",
            "machine_stress_index",
            "thermal_risk_index",
            "wear_efficiency_index",
        ]
        
        for col in feature_cols:
            assert np.isclose(predictor_df[col].iloc[0], expected_df[col].iloc[0], rtol=1e-6), \
                f"Feature {col} mismatch between predictor and centralized module"