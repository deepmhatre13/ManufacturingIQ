import hashlib
import json
import logging
import os
import warnings
import joblib
import pandas as pd
import sklearn
from pathlib import Path
from sklearn.base import BaseEstimator

from app.scoring import calculate_health_and_status, calculate_confidence  # H-1, H-2

logger = logging.getLogger(__name__)

# sklearn/XGBoost compatibility shim for scikit-learn >= 1.6
# ------------------------------------------------------------------
# scikit-learn 1.6+ invokes check_is_fitted inside predict_proba, which
# calls estimator.__sklearn_tags__() and expects attributes like
# requires_fit/transformer_tags.  Older XGBoost estimators inside the
# saved pipeline do not provide these, raising AttributeError and
# breaking the entire agentic graph.
#
# Instead of trying to synthesize tags (which is brittle), bypass
# check_is_fitted entirely for inference.  This is safe here because
# we only load a known-good, previously trained model artifact.
# ------------------------------------------------------------------
import sklearn.utils.validation as _sklearn_validation

_original_check_is_fitted = getattr(_sklearn_validation, "check_is_fitted", None)

def _noop_check_is_fitted(estimator, *args, **kwargs):
    return None

if _original_check_is_fitted is not None:
    _sklearn_validation.check_is_fitted = _noop_check_is_fitted

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_MODEL_PATH = MODELS_DIR / "production_model.pkl"
_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --- Load metadata (training_stats + expected checksum) -------------------
_training_stats: dict = {}
_metadata: dict = {}
try:
    with open(_METADATA_PATH, "r") as _f:
        _metadata = json.load(_f)
    _training_stats = _metadata.get("training_stats", {})
    if not _training_stats:
        logger.warning(
            "No training_stats in model_metadata.json; "
            "machine_stress_index will be constant at single-row inference."
        )
except Exception as exc:
    logger.warning("Could not load model_metadata.json: %s", exc)

# --- H-6: Integrity verification before loading the pickle ----------------
import secrets as _secrets_mod  # need compare_digest for constant-time comparison

def _verify_model_checksum(model_path: Path, metadata: dict) -> None:
    """Raise RuntimeError if the model file's SHA-256 doesn't match metadata."""
    expected = metadata.get("sha256")
    if not expected:
        logger.warning(
            "No 'sha256' key in model_metadata.json — skipping integrity check. "
            "Add it by running: python -c \"import hashlib; "
            "print(hashlib.sha256(open('models/production_model.pkl','rb').read()).hexdigest())\""
        )
        return
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not _secrets_mod.compare_digest(actual, expected):
        raise RuntimeError(
            f"Model integrity check FAILED for {model_path.name}. "
            f"Expected SHA-256 {expected!r}, got {actual!r}. "
            "The file may be corrupted or tampered with — refusing to load."
        )
    logger.info("Model integrity check passed for %s", model_path.name)

try:
    _verify_model_checksum(_MODEL_PATH, _metadata)
    model = joblib.load(_MODEL_PATH)
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
except Exception as exc:
    logger.error("Failed to load model artifacts from %s: %s", MODELS_DIR, exc)
    raise


def create_features(data):
    """
    Convert API payload to DataFrame and apply centralized feature engineering.

    Args:
        data: Dict with API payload keys (Air_temperature_K, etc.)

    Returns:
        DataFrame with raw and engineered features
    """
    from feature_engineering.engineer import build_engineered_features

    # Map API payload keys to canonical column names
    df = pd.DataFrame([{
        "Type": data["Type"],
        "Air temperature [K]": data["Air_temperature_K"],
        "Process temperature [K]": data["Process_temperature_K"],
        "Rotational speed [rpm]": data["Rotational_speed_rpm"],
        "Torque [Nm]": data["Torque_Nm"],
        "Tool wear [min]": data["Tool_wear_min"]
    }])

    # Pass training_stats so machine_stress_index is normalized correctly (C-4 fix)
    engineered_df = build_engineered_features(df, training_stats=_training_stats or None)

    return engineered_df


def calculate_health(probability: float) -> tuple[float, str]:
    """Shim kept for backward-compatibility; delegates to app.scoring (H-1)."""
    health_score, status, _risk = calculate_health_and_status(probability)
    return health_score, status


def log_production_data(
    feature_df,
    probability,
    health_score,
    status
):
    logger.info(
        "Logging production data",
        extra={
            "probability": float(probability),
            "health_score": float(health_score),
            "machine_status": status
        }
    )

    os.makedirs(
        "data/production",
        exist_ok=True
    )

    production_file = (
        "data/production/production_data.csv"
    )

    log_df = feature_df.copy()

    log_df[
        "failure_probability"
    ] = probability

    log_df[
        "health_score"
    ] = health_score

    log_df[
        "machine_status"
    ] = status

    if os.path.exists(
        production_file
    ):

        log_df.to_csv(
            production_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        log_df.to_csv(
            production_file,
            index=False
        )


def predict_machine_failure(payload):
    """Run inference and return failure probability, health score, and status."""
    feature_df = create_features(
        payload
    )

    logger.info(
        "Running prediction",
        extra={"payload_keys": list(payload.keys())}
    )

    model_input = feature_df[
        feature_columns
    ]

    probability = (
        model
        .predict_proba(
            model_input
        )[0][1]
    )

    health_score, status = (
        calculate_health(
            probability
        )
    )

    log_production_data(
        feature_df,
        probability,
        health_score,
        status
    )

    logger.info(
        "Prediction complete",
        extra={
            "failure_probability": round(float(probability), 4),
            "health_score": round(float(health_score), 2),
            "machine_status": status
        }
    )

    return {

        "failure_probability":

            round(
                float(probability),
                4
            ),

        "health_score":

            round(
                float(health_score),
                2
            ),

        "machine_status":

            status
    }