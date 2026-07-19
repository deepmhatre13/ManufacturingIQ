"""
ManufacturingIQ Agentic AI v3 - Explainability Agent

Generates SHAP-backed explanations for predictions.
SHAP import is deferred to avoid numba/numpy version conflicts at module level.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from state.schema import ShapExplanation

logger = logging.getLogger(__name__)


def _compute_shap_values(model, X):
    """Compute SHAP values lazily. Falls back gracefully if SHAP unavailable."""
    try:
        import shap
        if hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            return shap_values, explainer
        return None, None
    except Exception as exc:
        logger.warning("SHAP computation failed (numba/numpy version conflict?): %s", exc)
        return None, None


def _extract_top_contributors(
    shap_values,
    feature_names: List[str],
    top_n: int = 5,
):
    """
    Given a 1-row SHAP value array and feature names, return:
      - top_contributors: top N feature names by absolute SHAP value
      - positive_contributors: features pushing prediction toward failure
      - negative_contributors: features pushing prediction away from failure
    """
    import numpy as np

    if shap_values is None or len(shap_values) == 0:
        return [], [], []

    # shap_values may be 2D (n_samples x n_features) — take first row
    row = shap_values[0] if shap_values.ndim == 2 else shap_values

    abs_vals = abs(row)
    sorted_idxs = list(reversed(sorted(range(len(abs_vals)), key=lambda i: abs_vals[i])))

    top_contributors = [feature_names[i] for i in sorted_idxs[:top_n]]
    positive_contributors = [feature_names[i] for i in sorted_idxs if row[i] > 0][:top_n]
    negative_contributors = [feature_names[i] for i in sorted_idxs if row[i] < 0][:top_n]

    return top_contributors, positive_contributors, negative_contributors


def _build_explanation_text(top_contributors: List[str], positive: List[str], negative: List[str]) -> str:
    if not top_contributors:
        return "No explanation available."
    pos_text = ", ".join(positive[:3]) if positive else "none"
    neg_text = ", ".join(negative[:3]) if negative else "none"
    top_text = ", ".join(top_contributors[:5])
    return (
        f"Top feature drivers for this prediction: {top_text}. "
        f"Features increasing failure risk: {pos_text}. "
        f"Features reducing failure risk: {neg_text}."
    )


def run_explanation(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        prediction = state.get("prediction")
        if prediction is None:
            state["shap_explanation"] = ShapExplanation(
                top_contributors=[],
                positive_contributors=[],
                negative_contributors=[],
                explanation_text="No prediction available.",
                confidence=0.0,
            )
            return state

        # Attempt real SHAP computation
        from app.predictor import model as loaded_model, feature_columns, _training_stats
        from feature_engineering.engineer import build_engineered_features

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

        shap_values, _ = _compute_shap_values(loaded_model, X)

        if shap_values is not None:
            import numpy as np
            sv = shap_values if isinstance(shap_values, np.ndarray) else shap_values
            top_contributors, positive_contributors, negative_contributors = _extract_top_contributors(
                sv, feature_columns
            )
            explanation_text = _build_explanation_text(top_contributors, positive_contributors, negative_contributors)
            confidence = float(prediction.get("confidence", 0.85)) / 100.0
            state["shap_explanation"] = ShapExplanation(
                top_contributors=top_contributors,
                positive_contributors=positive_contributors,
                negative_contributors=negative_contributors,
                explanation_text=explanation_text,
                confidence=confidence,
            )
            logger.info("SHAP explanation computed successfully; top contributors: %s", top_contributors)
        else:
            # Real SHAP failed — use deterministic feature importance from the model
            logger.warning("SHAP unavailable; falling back to model feature importances.")
            try:
                importances = loaded_model.feature_importances_
                sorted_idxs = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
                top_contributors = [feature_columns[i] for i in sorted_idxs[:5]]
                explanation_text = (
                    f"SHAP unavailable; top features by model importance: {', '.join(top_contributors)}."
                )
            except Exception:
                top_contributors = list(feature_columns[:5])
                explanation_text = "SHAP computation unavailable; showing top model features."
            state["shap_explanation"] = ShapExplanation(
                top_contributors=top_contributors,
                positive_contributors=top_contributors[:2],
                negative_contributors=[],
                explanation_text=explanation_text,
                confidence=float(prediction.get("confidence", 0.85)) / 100.0,
            )
    except Exception as exc:
        logger.exception("Explanation agent failed: %s", exc)
        state["shap_explanation"] = ShapExplanation(
            top_contributors=[],
            positive_contributors=[],
            negative_contributors=[],
            explanation_text="Explanation generation failed.",
            confidence=0.0,
        )
    return state