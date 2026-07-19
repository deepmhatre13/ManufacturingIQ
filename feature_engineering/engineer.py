"""
Canonical Feature Engineering for ManufacturingIQ.

This module provides the single source of truth for all engineered features.
All formulas match the original training-time logic from notebook.ipynb cells 20-24,
ensuring zero training/serving skew.
"""

import pandas as pd
from typing import Optional


def build_engineered_features(
    df: pd.DataFrame,
    training_stats: Optional[dict] = None
) -> pd.DataFrame:
    """
    Apply ManufacturingIQ feature engineering to a DataFrame.

    This is the canonical implementation used by both training (notebook.ipynb)
    and inference (app/predictor.py) to guarantee identical feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the six required raw sensor columns:
          - 'Type': Product type (L/M/H)
          - 'Air temperature [K]': Ambient temperature
          - 'Process temperature [K]': Machine process temperature
          - 'Rotational speed [rpm]': Motor RPM
          - 'Torque [Nm]': Applied torque
          - 'Tool wear [min]': Tool degradation time

    training_stats : dict, optional
        Pre-computed statistics from training data for consistent normalization.
        Keys expected: 'temperature_difference_max', 'torque_max', 'wear_max'.
        If None (training mode), computes max values from input df.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 6 engineered feature columns appended:

        Engineered Features (physical interpretation)
        -------------------------------------------
        - temperature_difference:
            Delta between process and ambient air temperature (K).
            Larger deltas indicate higher thermal stress on machine components.

        - torque_speed_ratio:
            Torque divided by rotational speed (Nm per RPM).
            High values at low speeds suggest increased mechanical load and strain.

        - wear_intensity:
            Interaction term: tool wear time multiplied by applied torque.
            Captures cumulative mechanical stress from prolonged high-torque operation.

        - machine_stress_index:
            Composite normalized index combining thermal, mechanical, and wear factors.
            Formula: (temp_diff/max_temp_diff) + (torque/max_torque) + (wear/max_wear)
            Provides holistic machine stress measurement on an approximately [0, 3] scale.

        - thermal_risk_index:
            Ratio of process temperature to ambient air temperature.
            Values above ~1.03 suggest elevated thermal loading and potential overheating.

        - wear_efficiency_index:
            Rotational speed normalized by tool wear time (+1 offset).
            Decreasing values indicate operational efficiency degradation as tools wear.

    Raises
    ------
    ValueError
        If any required raw column is missing from the input DataFrame.

    Notes
    -----
    All division operations use epsilon (1e-6) guarding to prevent ZeroDivisionError
    on production inputs. This matches the protective behavior expected in live traffic
    where sensor data quality is not guaranteed.

    By importing this module in both notebook.ipynb and app/predictor.py, we eliminate
    training/serving skew: the exact same feature values are computed during model
    training and during live prediction requests.
    """
    # Input validation
    required_columns = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"DataFrame must contain: {required_columns}"
        )

    # Work on a copy to avoid modifying the original
    engineered_df = df.copy()

    # Feature 1: temperature_difference
    # Physical reasoning: Delta between process and ambient temperature
    # indicates thermal stress on the machine
    engineered_df["temperature_difference"] = (
        engineered_df["Process temperature [K]"]
        - engineered_df["Air temperature [K]"]
    )

    # Feature 2: torque_speed_ratio
    # Physical reasoning: Higher torque at lower speeds indicates
    # increased mechanical load and potential strain on the drive train
    epsilon = 1e-6
    engineered_df["torque_speed_ratio"] = (
        engineered_df["Torque [Nm]"]
        / (engineered_df["Rotational speed [rpm]"] + epsilon)
    )

    # Feature 3: wear_intensity
    # Physical reasoning: Interaction between tool degradation time
    # and applied torque captures cumulative mechanical stress on cutting tools
    engineered_df["wear_intensity"] = (
        engineered_df["Tool wear [min]"]
        * engineered_df["Torque [Nm]"]
    )

    # Feature 4: machine_stress_index
    # Physical reasoning: Composite index combining thermal differential,
    # mechanical torque, and tool wear normalized by their respective max values
    # from training data. Produces an approximately [0, 3] scale where higher
    # values indicate compounded stress across multiple failure modes.
    if training_stats is None:
        # Training mode: compute statistics from current batch
        temp_diff_max = engineered_df["temperature_difference"].max()
        torque_max = engineered_df["Torque [Nm]"].max()
        wear_max = engineered_df["Tool wear [min]"].max()
    else:
        # Inference mode: use fixed training statistics for consistency
        temp_diff_max = training_stats.get("temperature_difference_max", 1.0)
        torque_max = training_stats.get("torque_max", 1.0)
        wear_max = training_stats.get("wear_max", 1.0)

    # Guard against zero denominators in normalization
    temp_diff_max = max(temp_diff_max, epsilon)
    torque_max = max(torque_max, epsilon)
    wear_max = max(wear_max, epsilon)

    engineered_df["machine_stress_index"] = (
        (engineered_df["temperature_difference"] / temp_diff_max)
        + (engineered_df["Torque [Nm]"] / torque_max)
        + (engineered_df["Tool wear [min]"] / wear_max)
    )

    # Feature 5: thermal_risk_index
    # Physical reasoning: Ratio of process to ambient temperature indicates
    # thermal loading on machine components. Values > ~1.03 suggest
    # elevated risk of heat-related failures (bearing degradation, insulation breakdown).
    engineered_df["thermal_risk_index"] = (
        engineered_df["Process temperature [K]"]
        / (engineered_df["Air temperature [K]"] + epsilon)
    )

    # Feature 6: wear_efficiency_index
    # Physical reasoning: Speed normalized by wear time (+1 offset) shows
    # operational efficiency degradation as cutting tools wear down.
    # Lower values indicate the machine must work harder (or slower) to compensate
    # for degraded tool geometry.
    engineered_df["wear_efficiency_index"] = (
        engineered_df["Rotational speed [rpm]"]
        / (engineered_df["Tool wear [min]"] + 1)  # +1 matches original notebook formula
    )

    return engineered_df