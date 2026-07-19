"""
ManufacturingIQ Agentic AI - History Utilities

Append/load agent prediction history records.
"""

import csv
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

HISTORY_PATH = os.path.join("data", "agent_prediction_history.csv")
FIELDNAMES = [
    "timestamp",
    "machine_type",
    "health_score",
    "failure_probability",
    "risk_level",
    "status",
    "confidence",
    "recommendation",
    "trend",
]


def _ensure_file():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_history(record: Dict[str, Any]) -> None:
    _ensure_file()
    try:
        with open(HISTORY_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({k: record.get(k, "") for k in FIELDNAMES})
    except Exception as exc:
        logger.exception("Failed to append history: %s", exc)


def load_history(limit: int | None = None) -> List[Dict[str, Any]]:
    _ensure_file()
    rows: List[Dict[str, Any]] = []
    try:
        with open(HISTORY_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except FileNotFoundError:
        return []
    rows.reverse()
    if limit:
        rows = rows[:limit]
    return rows