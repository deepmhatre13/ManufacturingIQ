"""
Drift monitoring module for statistical feature drift detection.

This module provides tools to detect feature drift between reference (training)
and production datasets using statistical tests.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_psi(
    reference: pd.Series, current: pd.Series, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution (training data).
    current : pd.Series
        Current distribution (production data).
    bins : int, optional
        Number of bins for histogram, by default 10.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-6.

    Returns
    -------
    float
        PSI value. Interpretation: <0.1 no drift, 0.1-0.25 moderate, >0.25 high.

    Raises
    ------
    ValueError
        If data is empty after removing NaN.

    Examples
    --------
    >>> psi = calculate_psi(ref_series, curr_series)
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        raise ValueError("Reference and current data must not be empty")

    bin_edges = np.histogram_bin_edges(ref_clean, bins=bins)
    ref_hist, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_hist, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = ref_hist / ref_hist.sum()
    curr_pct = curr_hist / curr_hist.sum()

    # Check for constant reference
    if np.all(ref_pct == ref_pct[0]):
        logger.warning("Reference data is constant. PSI may not be meaningful.")
        return 0.0

    # Handle division by zero
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


def calculate_ks(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution.
    current : pd.Series
        Current distribution.

    Returns
    -------
    tuple[float, float]
        KS statistic and p-value.

    Raises
    ------
    ValueError
        If insufficient valid data.

    Examples
    --------
    >>> stat, p_value = calculate_ks(ref_series, curr_series)
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()

    if len(ref_clean) < 2 or len(curr_clean) < 2:
        raise ValueError("Each array must contain at least 2 non-NaN values")

    statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)
    return float(statistic), float(p_value)


def detect_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    psi_bins: int = 10,
    psi_epsilon: float = 1e-6,
    ks_threshold: float = 0.05,
) -> Dict:
    """
    Detect drift for a single feature.

    Parameters
    ----------
    reference : pd.Series
        Reference feature values.
    current : pd.Series
        Current feature values.
    feature_name : str
        Name of feature being analyzed.
    psi_bins : int, optional
        Number of bins for PSI, by default 10.
    psi_epsilon : float, optional
        Epsilon for PSI calculation, by default 1e-6.
    ks_threshold : float, optional
        KS significance threshold, by default 0.05.

    Returns
    -------
    Dict
        Results with keys: feature, psi, ks_statistic, p_value,
        drift, severity.
    """
    logger.info(f"Analyzing drift for feature: {feature_name}")

    # Check for constant or single-value features
    ref_unique = reference.dropna().unique()
    curr_unique = current.dropna().unique()

    if len(ref_unique) <= 1:
        logger.warning(
            f"Feature '{feature_name}' has constant values or insufficient variance"
        )
        return {
            "feature": feature_name,
            "psi": 0.0,
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift": False,
            "severity": "No Drift",
        }

    if len(curr_unique) == 0:
        logger.warning(f"Feature '{feature_name}' has all NaN values in current data")
        return {
            "feature": feature_name,
            "psi": float("inf"),
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift": True,
            "severity": "High",
        }

    try:
        psi = calculate_psi(reference, current, bins=psi_bins, epsilon=psi_epsilon)
    except ValueError as e:
        logger.error(f"Error calculating PSI for '{feature_name}': {e}")
        psi = 0.0

    try:
        ks_stat, p_value = calculate_ks(reference, current)
    except ValueError as e:
        logger.error(f"Error calculating KS for '{feature_name}': {e}")
        ks_stat = 0.0
        p_value = 1.0

    # Determine drift
    ks_drift = p_value < ks_threshold
    psi_high = psi > 0.25
    psi_moderate = 0.1 <= psi <= 0.25
    psi_no = psi < 0.1

    if psi_high or (ks_drift and psi_moderate) or (ks_drift and psi_no):
        drift_detected = True
        if psi_high:
            severity = "High"
        else:
            severity = "Moderate"
    else:
        drift_detected = False
        severity = "No Drift"

    logger.info(
        f"Drift detection for '{feature_name}': PSI={psi:.4f}, KS={ks_stat:.4f}, "
        f"p={p_value:.4f}, severity={severity}"
    )

    return {
        "feature": feature_name,
        "psi": round(psi, 6),
        "ks_statistic": round(ks_stat, 6),
        "p_value": round(p_value, 6),
        "drift": drift_detected,
        "severity": severity,
    }


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Generate drift report for multiple features.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Reference (training) data.
    current_data : pd.DataFrame
        Current (production) data.
    features : list of str, optional
        Features to analyze. If None, uses common columns.

    Returns
    -------
    Dict[str, Dict]
        Nested dict mapping feature names to drift results.

    Raises
    ------
    ValueError
        If no common features found when features=None.
    """
    logger.info("Starting drift detection analysis")

    if reference_data.empty or current_data.empty:
        raise ValueError(
            "Both reference_data and current_data must be non-empty DataFrames"
        )

    if features is None:
        features = sorted(set(reference_data.columns) & set(current_data.columns))

    if not features:
        raise ValueError("No common features found between reference and current data")

    logger.info(f"Analyzing {len(features)} features: {', '.join(features)}")

    drift_report = {}
    for feature in features:
        if feature not in reference_data.columns or feature not in current_data.columns:
            logger.warning(f"Feature '{feature}' not found in both datasets, skipping")
            continue

        result = detect_feature_drift(
            reference=reference_data[feature],
            current=current_data[feature],
            feature_name=feature,
        )
        drift_report[feature] = result

    return drift_report


def calculate_drift(
    reference_df: Optional[pd.DataFrame] = None,
    production_df: Optional[pd.DataFrame] = None,
    features: Optional[List[str]] = None,
    production_path: str = "data/production/production_data.csv",
    reference_data: Optional[pd.DataFrame] = None,
    current_data: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
) -> Dict:
    """
    Calculate feature drift between reference and production data.

    Parameters
    ----------
    reference_df : pd.DataFrame, optional
        Reference (training) dataframe. If None, uses legacy mode.
    production_df : pd.DataFrame, optional
        Production dataframe.
    features : list of str, optional
        Features to analyze.
    production_path : str, optional
        Path to production data CSV.
    reference_data : pd.DataFrame, optional
        Alias for reference_df for backward compatibility.
    current_data : pd.DataFrame, optional
        Alias for production_df for backward compatibility.
    columns : list of str, optional
        Alias for features for backward compatibility.

    Returns
    -------
    Dict
        Drift report with keys: drift_detected, max_drift, details, full_report.
    """
    logger.info("=== Drift Detection Started ===")

    # Handle backward compatibility aliases
    if reference_data is not None:
        reference_df = reference_data
    if current_data is not None:
        production_df = current_data
    if columns is not None:
        features = columns

    # If no reference, use legacy mode
    if reference_df is None:
        logger.warning("No reference data provided, using legacy hardcoded values")
        return _legacy_drift_calculation(production_df)

    # Load production data if needed
    if production_df is None:
        try:
            production_path_obj = Path(production_path)
            if not production_path_obj.exists():
                logger.error(f"Production data not found at {production_path}")
                return {"drift_detected": False, "message": "No production data"}
            production_df = pd.read_csv(production_path)
            logger.info(f"Loaded production data: {production_df.shape}")
        except Exception as e:
            logger.error(f"Could not load production data: {e}")
            return {"drift_detected": False, "message": "No production data available"}

    try:
        # Generate drift report
        drift_report = generate_drift_report(
            reference_data=reference_df,
            current_data=production_df,
            features=features,
        )

        # Calculate summary
        max_drift = max((v["psi"] for v in drift_report.values()), default=0.0)
        features_with_drift = [
            f
            for f, v in drift_report.items()
            if v.get("drift_detected", v.get("drift", False))
        ]

        return {
            "drift_detected": len(features_with_drift) > 0,
            "max_drift": round(max_drift, 2),
            "details": drift_report,
            "full_report": drift_report,
        }

    except Exception as e:
        logger.error(f"Error during drift detection: {e}", exc_info=True)
        return {
            "drift_detected": False,
            "message": f"Error during drift detection: {str(e)}",
        }


def _legacy_drift_calculation(production_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Legacy drift calculation using hardcoded references.

    Parameters
    ----------
    production_df : pd.DataFrame, optional
        Production data.

    Returns
    -------
    Dict
        Legacy format drift report.
    """
    logger.warning("Using legacy drift calculation with hardcoded reference values")

    if production_df is None:
        try:
            production_path = Path("data/production/production_data.csv")
            if not production_path.exists():
                return {"drift_detected": False, "message": "No production data"}
            production_df = pd.read_csv(production_path)
        except Exception:
            return {"drift_detected": False, "message": "No production data"}

    TRAINING_REFERENCE = {
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
        "Rotational speed [rpm]": 1500.0,
    }

    drift_report = {}

    for feature, training_mean in TRAINING_REFERENCE.items():
        if feature not in production_df.columns:
            logger.warning(f"Feature '{feature}' not found in production data.")
            continue

        production_mean = production_df[feature].mean()
        drift_percentage = abs((production_mean - training_mean) / training_mean) * 100
        drift_report[feature] = round(drift_percentage, 2)

    max_drift = max(drift_report.values()) if drift_report else 0.0

    return {
        "drift_detected": bool(max_drift > 10),
        "max_drift": float(round(max_drift, 2)),
        "details": {k: float(v) for k, v in drift_report.items()},
    }


if __name__ == "__main__":
    # Example usage
    report = calculate_drift()
    print(report)
