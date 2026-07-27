"""
Structured Logging Configuration for ManufacturingIQ

Provides JSON-formatted logging for production and human-readable logging for development.
"""

import logging
import sys
import json
import os
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", "unknown"),
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Include extra fields
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated",
            "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "pathname", "filename", "module", "levelno", "levelname",
        }
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }
        if extra:
            log_entry["extra"] = extra

        return json.dumps(log_entry, default=str)


def configure_logging(
    service_name: str,
    json_output: Optional[bool] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Configure root logging once per process.

    Parameters
    ----------
    service_name : str
        Name of the service (e.g., 'api', 'dashboard', 'monitoring', 'retraining', 'scheduler').
    json_output : bool, optional
        If True, output JSON lines. If False, output human-readable colored logs.
        If None, infer from ENV environment variable ('production' -> JSON, else human-readable).
    log_level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, read from LOG_LEVEL env var.
    """
    if json_output is None:
        env = os.getenv("ENV", "development")
        json_output = env == "production"

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        log_level = log_level.upper()

    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level, logging.INFO))

    if json_output:
        formatter = JSONFormatter()
    else:
        # Human-readable format with service name
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log configuration event
    root_logger.info(
        "Logging configured",
        extra={
            "service": service_name,
            "json_output": json_output,
            "log_level": log_level,
        },
    )