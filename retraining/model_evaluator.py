import json
import logging

from logging_config import configure_logging

configure_logging(service_name="retraining")
logger = logging.getLogger(__name__)

PRODUCTION_AUC = 0.94

try:
    with open(
        "models/candidate_metrics.json"
    ) as f:

        candidate_metrics = json.load(
            f
        )

    candidate_auc = candidate_metrics[
        "roc_auc"
    ]

    if candidate_auc > PRODUCTION_AUC:

        logger.info(
            "PROMOTE CANDIDATE: candidate AUC %.4f > production %.4f",
            candidate_auc,
            PRODUCTION_AUC,
        )

    else:

        logger.info(
            "KEEP PRODUCTION MODEL: candidate AUC %.4f <= production %.4f",
            candidate_auc,
            PRODUCTION_AUC,
        )
except Exception as exc:
    logger.exception("Failed to evaluate candidate model: %s", exc)
    raise
