import numpy as np
import pandas as pd
import pytest

from monitoring.drift_monitor import (
    calculate_ks,
    calculate_psi,
    calculate_drift,
    detect_feature_drift,
    generate_drift_report,
)


class TestCalculatePSI:
    """Tests for calculate_psi function."""

    def test_no_drift(self):
        """Similar distributions should have low PSI."""
        ref = pd.Series(np.random.normal(50, 10, 1000))
        curr = pd.Series(np.random.normal(50, 10, 1000))
        psi = calculate_psi(ref, curr)
        assert psi < 0.1, f"Expected no drift, got PSI={psi}"

    def test_moderate_drift(self):
        """Moderate distribution shift should yield PSI > 0.1."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 1000))
        curr = pd.Series(np.random.normal(55, 12, 1000))
        psi = calculate_psi(ref, curr)
        assert psi > 0.1, f"Expected PSI > 0.1, got PSI={psi}"

    def test_high_drift(self):
        """Large distribution shift should yield high PSI."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 1000))
        curr = pd.Series(np.random.normal(80, 5, 1000))
        psi = calculate_psi(ref, curr)
        assert psi > 0.25, f"Expected high drift, got PSI={psi}"

    def test_empty_reference_raises(self):
        """Empty reference should raise ValueError."""
        with pytest.raises(
            ValueError, match="Reference and current data must not be empty"
        ):
            calculate_psi(pd.Series([], dtype=float), pd.Series([1, 2, 3]))

    def test_empty_current_raises(self):
        """Empty current should raise ValueError."""
        with pytest.raises(
            ValueError, match="Reference and current data must not be empty"
        ):
            calculate_psi(pd.Series([1, 2, 3]), pd.Series([], dtype=float))

    def test_custom_bins(self):
        """PSI calculation should work with different bin counts."""
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(50, 10, 500))
        psi_10 = calculate_psi(ref, curr, bins=10)
        psi_20 = calculate_psi(ref, curr, bins=20)
        # Both should be low
        assert psi_10 < 0.1
        assert psi_20 < 0.1

    def test_with_nan(self):
        """PSI should handle NaN values gracefully."""
        ref = pd.Series([1, 2, np.nan, 4, 5] * 20)
        curr = pd.Series([1, 2, 3, 4, 6] * 20)
        psi = calculate_psi(ref, curr)
        assert isinstance(psi, float)
        assert psi >= 0

    def test_constant_reference(self):
        """Constant reference should produce finite PSI."""
        ref = pd.Series([5.0] * 100)
        curr = pd.Series([5.0] * 50 + [6.0] * 50)
        psi = calculate_psi(ref, curr)
        assert isinstance(psi, float)
        assert np.isfinite(psi)


class TestCalculateKS:
    """Tests for calculate_ks function."""

    def test_same_distribution(self):
        """Samples from same distribution should have high p-value."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(50, 10, 500))
        statistic, p_value = calculate_ks(ref, curr)
        assert p_value > 0.05, f"Expected no drift, got p={p_value}"

    def test_different_distributions(self):
        """Samples from different distributions should have low p-value."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(70, 10, 500))
        statistic, p_value = calculate_ks(ref, curr)
        assert p_value < 0.05, f"Expected drift, got p={p_value}"

    def test_returns_float_types(self):
        """KS function should return float values."""
        ref = pd.Series([1, 2, 3, 4, 5])
        curr = pd.Series([1, 2, 3, 4, 6])
        statistic, p_value = calculate_ks(ref, curr)
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)

    def test_single_value_raises(self):
        """Samples with <2 values should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculate_ks(pd.Series([1]), pd.Series([2, 3]))

    def test_all_nan_raises(self):
        """Samples with all NaN should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            calculate_ks(pd.Series([np.nan, np.nan]), pd.Series([1, 2]))

    def test_with_nan(self):
        """KS should handle NaN by ignoring them."""
        ref = pd.Series([1, 2, np.nan, 4, 5] * 10)
        curr = pd.Series([1, 2, 3, 4, 6] * 10)
        statistic, p_value = calculate_ks(ref, curr)
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)


class TestDetectFeatureDrift:
    """Tests for detect_feature_drift function."""

    def test_no_drift(self):
        """Similar distributions should show no drift."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(50, 10, 500))
        result = detect_feature_drift(ref, curr, "feature1")
        assert result["drift"] is False
        assert result["severity"] == "No Drift"

    def test_moderate_drift(self):
        """Moderate drift should be detected."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(55, 12, 500))
        result = detect_feature_drift(ref, curr, "feature1")
        assert result["drift"] is True
        assert result["severity"] in ["Moderate", "High"]

    def test_high_drift(self):
        """High drift should be detected."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(50, 10, 500))
        curr = pd.Series(np.random.normal(80, 5, 500))
        result = detect_feature_drift(ref, curr, "feature1")
        assert result["drift"] is True
        assert result["severity"] == "High"

    def test_constant_reference_feature(self):
        """Constant reference should return no drift."""
        ref = pd.Series([5.0] * 100)
        curr = pd.Series([5.0] * 50 + [6.0] * 50)
        result = detect_feature_drift(ref, curr, "constant_feature")
        assert result["drift"] is False
        assert result["severity"] == "No Drift"

    def test_missing_feature_data(self):
        """All-NaN current feature should signal drift."""
        ref = pd.Series([1, 2, 3, 4, 5] * 20)
        curr = pd.Series([np.nan] * 100)
        result = detect_feature_drift(ref, curr, "missing_feature")
        assert result["drift"] is True
        assert result["severity"] == "High"
        assert result["psi"] == float("inf")

    def test_result_structure(self):
        """Result should have expected keys."""
        ref = pd.Series([1, 2, 3, 4, 5])
        curr = pd.Series([1, 2, 3, 4, 6])
        result = detect_feature_drift(ref, curr, "test_feature")
        expected_keys = {
            "feature",
            "psi",
            "ks_statistic",
            "p_value",
            "drift",
            "severity",
        }
        assert set(result.keys()) >= expected_keys


class TestGenerateDriftReport:
    """Tests for generate_drift_report function."""

    def test_multiple_features(self):
        """Report should include all common features."""
        np.random.seed(42)
        ref = pd.DataFrame(
            {
                "a": np.random.normal(50, 10, 500),
                "b": np.random.normal(100, 20, 500),
                "c": np.random.normal(10, 2, 500),
            }
        )
        curr = pd.DataFrame(
            {
                "a": np.random.normal(51, 10, 500),
                "b": np.random.normal(105, 22, 500),
                "c": np.random.normal(10, 2, 500),
            }
        )
        report = generate_drift_report(ref, curr)
        assert len(report) == 3
        assert "a" in report
        assert "b" in report
        assert "c" in report

    def test_feature_intersection(self):
        """Only common features should be analyzed."""
        ref = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        curr = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
        report = generate_drift_report(ref, curr)
        assert "a" in report
        assert "b" not in report
        assert "c" not in report

    def test_custom_features_list(self):
        """Explicit features list should be respected."""
        ref = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        curr = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = generate_drift_report(ref, curr, features=["a"])
        assert "a" in report
        assert "b" not in report

    def test_empty_common_features(self):
        """No common features should raise ValueError."""
        ref = pd.DataFrame({"a": [1, 2, 3]})
        curr = pd.DataFrame({"b": [4, 5, 6]})
        with pytest.raises(ValueError, match="No common features found"):
            generate_drift_report(ref, curr)


class TestCalculateDrift:
    """Tests for calculate_drift (backward-compatible API)."""

    def test_legacy_mode_without_reference(self):
        """Without reference data, should use legacy hardcoded values."""
        production_data = pd.DataFrame(
            {
                "Torque [Nm]": [39, 40, 41, 38, 42],
                "Tool wear [min]": [98, 102, 100, 99, 101],
                "Rotational speed [rpm]": [1480, 1520, 1500, 1490, 1510],
            }
        )
        result = calculate_drift(production_df=production_data)
        assert "drift_detected" in result
        assert "max_drift" in result
        assert "details" in result

    def test_dataframe_interface(self):
        """Should accept dataframes directly for statistical testing."""
        np.random.seed(42)
        ref = pd.DataFrame(
            {
                "feature1": np.random.normal(50, 10, 500),
            }
        )
        curr = pd.DataFrame(
            {
                "feature1": np.random.normal(51, 10, 500),
            }
        )
        result = calculate_drift(
            reference_df=ref, production_df=curr, features=["feature1"]
        )
        assert "drift_detected" in result
        assert "max_drift" in result
        assert "details" in result
        assert "full_report" in result

    def test_missing_production_data_path(self):
        """Missing production path should return error dict."""
        result = calculate_drift(
            reference_df=pd.DataFrame({"a": [1, 2, 3]}),
            production_path="nonexistent.csv",
        )
        assert result["drift_detected"] is False
        assert "message" in result

    def test_production_dataframe_without_reference(self):
        """Production df without reference should trigger legacy mode."""
        prod = pd.DataFrame({"Torque [Nm]": [39, 40, 41]})
        result = calculate_drift(production_df=prod)
        assert result["drift_detected"] is False  # small drift

    def test_full_report_in_output(self):
        """Output should include details with per-feature metrics."""
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(50, 10, 500)})
        curr = pd.DataFrame({"x": np.random.normal(80, 10, 500)})
        result = calculate_drift(reference_df=ref, production_df=curr, features=["x"])
        assert "details" in result
        assert "x" in result["details"]
        # In statistical mode, details contains full report dicts
        assert isinstance(result["details"]["x"], dict)
        assert "psi" in result["details"]["x"]
