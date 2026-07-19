"""
Lightweight tests for dashboard authentication configuration.

Validates that .streamlit/secrets.toml.example has the expected keys present.
This guards against accidentally deleting required config fields.
"""

import tomlkit
from pathlib import Path


def test_secrets_example_exists():
    """Ensure .streamlit/secrets.toml.example file exists."""
    repo_root = Path(__file__).resolve().parents[1]
    secrets_example = repo_root / ".streamlit" / "secrets.toml.example"
    assert secrets_example.exists(), f"Missing {secrets_example}"


def test_secrets_example_has_auth_section():
    """Ensure [auth] section exists."""
    repo_root = Path(__file__).resolve().parents[1]
    secrets_example = repo_root / ".streamlit" / "secrets.toml.example"
    with open(secrets_example) as f:
        data = tomlkit.load(f)
    assert "auth" in data, "Missing [auth] section in secrets.toml.example"


def test_secrets_example_has_required_keys():
    """Ensure all required OAuth keys are present."""
    repo_root = Path(__file__).resolve().parents[1]
    secrets_example = repo_root / ".streamlit" / "secrets.toml.example"
    with open(secrets_example) as f:
        data = tomlkit.load(f)
    auth = data.get("auth", {})
    required_keys = [
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url",
    ]
    for key in required_keys:
        assert key in auth, f"Missing '{key}' in [auth] section"


def test_secrets_example_has_placeholders():
    """Ensure values are placeholders, not real secrets."""
    repo_root = Path(__file__).resolve().parents[1]
    secrets_example = repo_root / ".streamlit" / "secrets.toml.example"
    with open(secrets_example) as f:
        data = tomlkit.load(f)
    auth = data.get("auth", {})
    client_id = auth.get("client_id", "")
    client_secret = auth.get("client_secret", "")
    cookie_secret = auth.get("cookie_secret", "")
    assert "REPLACE_WITH" in client_id
    assert "REPLACE_WITH" in client_secret
    assert "REPLACE_WITH" in cookie_secret


def test_secrets_example_has_api_key():
    """Ensure MANUFACTURINGIQ_API_KEY is documented."""
    repo_root = Path(__file__).resolve().parents[1]
    secrets_example = repo_root / ".streamlit" / "secrets.toml.example"
    with open(secrets_example) as f:
        data = tomlkit.load(f)
    auth = data.get("auth", {})
    assert "MANUFACTURINGIQ_API_KEY" in auth