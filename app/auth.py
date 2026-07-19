"""
API key authentication for ManufacturingIQ backend.

Provides a FastAPI dependency that validates incoming requests using
the X-API-Key header against comma-separated keys configured via the
MANUFACTURINGIQ_API_KEYS environment variable.

No hardcoded keys; keys are rotated by changing the env var only.
"""

import os
import secrets

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _load_valid_api_keys() -> set[str]:
    """Load valid API keys from env var MANUFACTURINGIQ_API_KEYS."""
    keys_raw = os.getenv("MANUFACTURINGIQ_API_KEYS", "")
    if not keys_raw.strip():
        return set()
    return {key.strip() for key in keys_raw.split(",") if key.strip()}


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate the incoming API key against configured valid keys.

    Returns the valid key string on success.
    Raises 401 Unauthorized if missing or invalid.
    """
    valid_keys = _load_valid_api_keys()

    if not valid_keys:
        # No keys configured: deny all but provide a clear error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API keys not configured on the server. Set MANUFACTURINGIQ_API_KEYS.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing header '{API_KEY_NAME}'.",
        )

    # Use constant-time comparison to avoid timing attacks
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
    )