"""Tests for the Flight Control 1.4 compatibility patch (branch fc-1.4-compat)."""

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

import resource_queries
from resource_queries import (
    AuthenticationError,
    Configuration,
    FlightControlClient,
    FlightControlError,
)

FAKE_CA_PEM = b"-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n"
ORG = "00000000-0000-0000-0000-000000000000"


def fc14_client_yaml(tmp_path, **overrides):
    """A client.yaml in the layout `flightctl login` 1.4.1 writes (fake secrets)."""
    ca_file = tmp_path / "ca-bundle.crt"
    ca_file.write_bytes(FAKE_CA_PEM)
    data = {
        "service": {
            "server": "https://fc.lab.local:3443/",
            "certificate-authority-data": base64.b64encode(FAKE_CA_PEM).decode(),
        },
        "organization": ORG,
        "authentication": {
            "refresh-token": "fake-refresh",
            "access-token": "fake-cached-access",
            "access-token-expiry": "2099-01-01T00:00:00.123456789Z",
            "token-to-use": "id",
            "auth-provider": {
                "ca-file": str(ca_file),
                "auth-provider": {
                    "apiVersion": "v1beta1",
                    "kind": "AuthProvider",
                    "spec": {
                        "issuer": "https://fc.lab.local/_/pam-issuer/api/v1/auth/",
                        "clientId": "flightctl-client",
                        "clientSecret": "",
                        "providerType": "oidc",
                    },
                },
            },
        },
    }
    data.update(overrides)
    path = tmp_path / "client.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def load_config(path, env=None):
    """Build a Configuration from a given client.yaml path, with a controlled environment."""
    with patch.object(Configuration, "__init__", lambda self: None):
        config = Configuration()
    config.logger = Mock()
    config.config_path = Path(path)
    config.certs_path = Path("/nonexistent")
    with patch.dict(os.environ, env or {}, clear=True):
        config._load_config()
    return config


class TestFc14Config:
    def test_service_fields(self, tmp_path):
        c = load_config(fc14_client_yaml(tmp_path))
        assert c.api_base_url == "https://fc.lab.local:3443"
        assert Path(c.ca_cert_path).read_bytes() == FAKE_CA_PEM
        assert c.get_ssl_verify() == c.ca_cert_path
        assert c.organization == ORG

    def test_auth_fields(self, tmp_path):
        c = load_config(fc14_client_yaml(tmp_path))
        assert c.oidc_issuer == "https://fc.lab.local/_/pam-issuer/api/v1/auth"
        assert c.oidc_token_url is None
        assert c.client_id == "flightctl-client"
        assert c.client_secret is None
        assert c.refresh_token == "fake-refresh"
        assert c.auth_ca_cert_path == str(tmp_path / "ca-bundle.crt")
        assert c.get_auth_ssl_verify() == str(tmp_path / "ca-bundle.crt")

    def test_cached_access_token(self, tmp_path):
        c = load_config(fc14_client_yaml(tmp_path))
        assert c.cached_access_token == "fake-cached-access"
        assert c.cached_access_token_expiry == pytest.approx(4070908800.123456, abs=0.001)

    def test_env_overrides_still_win(self, tmp_path):
        env = {"API_BASE_URL": "https://other:3443", "REFRESH_TOKEN": "env-refresh", "OIDC_TOKEN_URL": "https://idp/token"}
        c = load_config(fc14_client_yaml(tmp_path), env)
        assert c.api_base_url == "https://other:3443"
        assert c.refresh_token == "env-refresh"
        assert c.oidc_token_url == "https://idp/token"

    def test_auth_ssl_verify_falls_back_to_api_verify(self, tmp_path):
        c = load_config(fc14_client_yaml(tmp_path))
        c.auth_ca_cert_path = None
        assert c.get_auth_ssl_verify() == c.ca_cert_path
        c.insecure_skip_verify = True
        assert c.get_auth_ssl_verify() is False


class TestParseExpiry:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2099-01-01T00:00:00Z", 4070908800.0),
            ("2099-01-01T00:00:00.123456789Z", 4070908800.123456),
            ("2099-01-01T02:00:00+02:00", 4070908800.0),
            ("not a time", 0.0),
            (None, 0.0),
            ("", 0.0),
        ],
    )
    def test_values(self, value, expected):
        assert resource_queries._parse_expiry(value) == pytest.approx(expected, abs=0.001)
