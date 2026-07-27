import time
import logging
import subprocess
import sys

from logging_config import configure_logging

configure_logging(service_name="scheduler")
logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60


while True:

    logger.info("Checking drift and retraining pipeline...")

    try:
        result = subprocess.run(
            [sys.executable, "retraining/retrain.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Retraining cycle completed successfully")
        else:
            logger.error(
                "Retraining subprocess failed with return code %d: %s",
                result.returncode,
                result.stderr.strip(),
            )
    except Exception as exc:
        logger.exception("Failed to run retraining subprocess: %s", exc)

    logger.info("Sleeping for %d seconds until next check", CHECK_INTERVAL)
    time.sleep(CHECK_INTERVAL)
