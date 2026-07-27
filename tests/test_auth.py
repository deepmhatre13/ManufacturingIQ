"""
Tests for API key authentication.

Validates that:
- /predict returns 401 with no key
- /predict returns 401 with wrong key
- /predict returns 200 with valid key
- / and /health are unauthenticated
"""

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.auth import verify_api_key, _load_valid_api_keys


client = TestClient(app)


class TestUnauthenticatedEndpoints:
    """Ensure health check and root are accessible without API key."""

    def test_root_is_public(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_is_public(self):
        response = client.get("/health")
        assert response.status_code == 200


class TestProtectedEndpoints:
    """Ensure /predict requires a valid API key."""

    def test_predict_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "test-key-123")
        payload = {
            "Type": "M",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 12,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 401

    def test_predict_rejects_invalid_key(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "test-key-123")
        payload = {
            "Type": "M",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 12,
        }
        response = client.post(
            "/predict",
            json=payload,
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_predict_accepts_valid_key(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "test-key-123")
        payload = {
            "Type": "M",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 12,
        }
        with patch("app.predictor.model") as mock_model, \
             patch("app.predictor.feature_columns", ["temperature_difference"]):
            mock_model.predict_proba.return_value = [[0.8, 0.2]]
            response = client.post(
                "/predict",
                json=payload,
                headers={"X-API-Key": "test-key-123"},
            )
            assert response.status_code == 200

    def test_predict_rejects_wrong_valid_key(self, monkeypatch):
        """When multiple keys are configured, only matching key should work."""
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "key-a,key-b")
        payload = {
            "Type": "M",
            "Air_temperature_K": 298.1,
            "Process_temperature_K": 308.6,
            "Rotational_speed_rpm": 1551,
            "Torque_Nm": 42.8,
            "Tool_wear_min": 12,
        }
        with patch("app.predictor.model") as mock_model, \
             patch("app.predictor.feature_columns", ["temperature_difference"]):
            mock_model.predict_proba.return_value = [[0.8, 0.2]]
            response = client.post(
                "/predict",
                json=payload,
                headers={"X-API-Key": "key-a"},
            )
            assert response.status_code == 200
            response = client.post(
                "/predict",
                json=payload,
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401


class TestApiKeyLoader:
    """Unit tests for _load_valid_api_keys."""

    def test_empty_env_returns_empty_set(self, monkeypatch):
        monkeypatch.delenv("MANUFACTURINGIQ_API_KEYS", raising=False)
        assert _load_valid_api_keys() == set()

    def test_single_key(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "abc123")
        assert _load_valid_api_keys() == {"abc123"}

    def test_multiple_keys(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "a,b,c")
        assert _load_valid_api_keys() == {"a", "b", "c"}

    def test_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", " a , b , c ")
        assert _load_valid_api_keys() == {"a", "b", "c"}

    def test_blank_entries_ignored(self, monkeypatch):
        monkeypatch.setenv("MANUFACTURINGIQ_API_KEYS", "a,,b,")
        assert _load_valid_api_keys() == {"a", "b"}
