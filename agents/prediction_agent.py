"""
ManufacturingIQ Agentic AI v3 - Prediction Agent

Runs feature engineering + XGBoost prediction without an LLM.
"""

import logging
from typing import Any, Dict

import pandas as pd

from feature_engineering.engineer import build_engineered_features
from app.predictor import model as loaded_model, feature_columns, _training_stats
from app.scoring import calculate_health_and_status, calculate_confidence  # H-1, H-2

logger = logging.getLogger(__name__)


def run_prediction(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw = state.get("raw_input", {})
        df = pd.DataFrame([{
            "Type": raw.get("Type", "L"),
            "Air temperature [K]": raw.get("Air_temperature_K", 300.0),
            "Process temperature [K]": raw.get("Process_temperature_K", 320.0),
            "Rotational speed [rpm]": raw.get("Rotational_speed_rpm", 1500),
            "Torque [Nm]": raw.get("Torque_Nm", 75.0),
            "Tool wear [min]": raw.get("Tool_wear_min", 200),
        }])
        df = build_engineered_features(df, training_stats=_training_stats or None)
        X = df[feature_columns]
        probs = loaded_model.predict_proba(X)[:, 1]
        pred = int(loaded_model.predict(X)[0])
        failure_prob = round(float(probs[0]), 4)
        health_score, status, risk = calculate_health_and_status(failure_prob)  # H-1
        confidence = calculate_confidence(failure_prob)  # H-2: real margin-based signal

        state["prediction"] = {
            "failure_prediction": pred,
            "failure_probability": failure_prob,
            "health_score": health_score,
            "machine_status": status,
            "confidence": confidence,
            "risk_level": risk,
        }
        state["engineered_features"] = {
            "temperature_difference": float(df["temperature_difference"].iloc[0]),
            "torque_speed_ratio": float(df["torque_speed_ratio"].iloc[0]),
            "wear_intensity": float(df["wear_intensity"].iloc[0]),
            "machine_stress_index": float(df["machine_stress_index"].iloc[0]),
            "thermal_risk_index": float(df["thermal_risk_index"].iloc[0]),
            "wear_efficiency_index": float(df["wear_efficiency_index"].iloc[0]),
        }
        return state
    except Exception as exc:
        logger.exception("Prediction agent failed: %s", exc)
        state["error"] = str(exc)
        return state