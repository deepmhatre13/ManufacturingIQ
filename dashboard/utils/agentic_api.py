"""
ManufacturingIQ Dashboard - Agentic AI API Client
Calls the backend agentic prediction endpoint and formats responses.
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

from utils.api import API_BASE_URL, TIMEOUT

logger = logging.getLogger(__name__)


def _get_api_key() -> Optional[str]:
    try:
        import streamlit as st
        if "MANUFACTURINGIQ_API_KEY" in st.secrets:
            return st.secrets["MANUFACTURINGIQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("MANUFACTURINGIQ_API_KEY")


def agentic_predict(
    machine_type: str,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    history: Optional[list[dict]] = None,
) -> Dict[str, Any]:
    payload = {
        "Type": machine_type,
        "Air_temperature_K": air_temp,
        "Process_temperature_K": process_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear,
    }
    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.post(
        f"{API_BASE_URL}/agentic/predict",
        json=payload,
        headers=headers,
        timeout=TIMEOUT,
    )

    if response.status_code != 200:
        error_detail = response.text if response.text else "No response body"
        raise RuntimeError(f"Agentic API status {response.status_code}: {error_detail}")

    return response.json()