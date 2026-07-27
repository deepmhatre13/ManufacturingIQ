import logging
import shutil

from logging_config import configure_logging

configure_logging(service_name="retraining")
logger = logging.getLogger(__name__)

try:
    shutil.copy(
        "models/candidate_model.pkl",
        "models/production_model.pkl"
    )
    logger.info("Production model updated successfully")
except Exception as exc:
    logger.exception("Failed to promote candidate model: %s", exc)
    raise
