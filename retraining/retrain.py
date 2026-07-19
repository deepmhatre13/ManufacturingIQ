import logging
import subprocess
import sys

from logging_config import configure_logging
from monitoring.drift_monitor import (
    calculate_drift
)

configure_logging(service_name="retraining")
logger = logging.getLogger(__name__)

drift_report = (
    calculate_drift()
)

logger.info(
    "Drift report: %s",
    drift_report
)

if drift_report[
    "drift_detected"
]:

    logger.info("Retraining Triggered")

    subprocess.run(

        [
            sys.executable,
            "data/simulate_labels.py"
        ]
    )

    subprocess.run(

        [
            sys.executable,
            "retraining/train_candidate.py"
        ]
    )

else:

    logger.info("No Retraining Needed")
