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


TOKEN_URL = "https://fc.lab.local/_/pam-issuer/api/v1/auth/token"


def fc14_mock_config(**kw):
    c = Mock()
    c.api_base_url = "https://fc.lab.local:3443"
    c.oidc_token_url = None
    c.oidc_issuer = "https://fc.lab.local/_/pam-issuer/api/v1/auth"
    c.client_id = "flightctl-client"
    c.client_secret = None
    c.refresh_token = "r1"
    c.cached_access_token = None
    c.cached_access_token_expiry = 0.0
    c.organization = None
    c.insecure_skip_verify = False
    c.get_ssl_verify.return_value = "/tmp/api-ca.crt"
    c.get_auth_ssl_verify.return_value = "/tmp/auth-ca.crt"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def client_with_token_url(**kw):
    return FlightControlClient(fc14_mock_config(oidc_token_url=TOKEN_URL, **kw))


def response(status=200, body=None):
    r = Mock()
    r.status_code = status
    r.json.return_value = body if body is not None else {}
    r.text = json.dumps(body if body is not None else {})
    if status >= 400:
        err = requests.exceptions.HTTPError(f"HTTP {status}")
        err.response = r
        r.raise_for_status.side_effect = err
    else:
        r.raise_for_status.return_value = None
    return r


INVALID_GRANT = {"error": "invalid_grant", "error_description": "token revoked"}


class TestDiscovery:
    @patch("resource_queries.requests.get")
    def test_token_url_from_discovery(self, mock_get):
        mock_get.return_value = response(200, {"token_endpoint": TOKEN_URL})
        client = FlightControlClient(fc14_mock_config())
        assert client._token_url() == TOKEN_URL
        assert mock_get.call_args.args[0] == (
            "https://fc.lab.local/_/pam-issuer/api/v1/auth/.well-known/openid-configuration"
        )
        assert mock_get.call_args.kwargs["verify"] == "/tmp/auth-ca.crt"
        client._token_url()
        mock_get.assert_called_once()  # cached

    @patch("resource_queries.requests.get")
    def test_keycloak_fallback_when_discovery_fails(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("boom")
        client = FlightControlClient(fc14_mock_config(oidc_issuer="https://kc/realms/flightctl"))
        assert client._token_url() == "https://kc/realms/flightctl/protocol/openid-connect/token"

    def test_explicit_token_url_skips_discovery(self):
        client = client_with_token_url()
        with patch("resource_queries.requests.get") as get:
            assert client._token_url() == TOKEN_URL
            get.assert_not_called()

    def test_init_needs_issuer_or_token_url(self):
        with pytest.raises(FlightControlError, match="OIDC_TOKEN_URL not configured"):
            FlightControlClient(fc14_mock_config(oidc_issuer=None))


class TestTokenLifecycle:
    def test_uses_cached_cli_token_while_valid(self):
        c = client_with_token_url(cached_access_token="cli-access", cached_access_token_expiry=time.time() + 600)
        with patch("resource_queries.requests.post") as post:
            assert c._get_access_token() == "cli-access"
            post.assert_not_called()

    def test_ignores_nearly_expired_cli_token(self):
        c = client_with_token_url(cached_access_token="old", cached_access_token_expiry=time.time() + 30)
        ok = response(200, {"access_token": "new", "expires_in": 3600})
        with patch("resource_queries.requests.post", return_value=ok) as post:
            assert c._get_access_token() == "new"
            post.assert_called_once()

    def test_refresh_request_and_rotation(self):
        c = client_with_token_url()
        ok = response(200, {"access_token": "a1", "refresh_token": "r2", "expires_in": 3600})
        with patch("resource_queries.requests.post", return_value=ok) as post:
            assert c._get_access_token() == "a1"
        assert c._refresh_token == "r2"
        kwargs = post.call_args.kwargs
        assert post.call_args.args[0] == TOKEN_URL
        assert kwargs["data"] == {"grant_type": "refresh_token", "refresh_token": "r1", "client_id": "flightctl-client"}
        assert kwargs["verify"] == "/tmp/auth-ca.crt"

    def test_sends_client_secret_when_set(self):
        c = client_with_token_url(client_secret="s3cret")
        ok = response(200, {"access_token": "a1", "expires_in": 3600})
        with patch("resource_queries.requests.post", return_value=ok) as post:
            c._get_access_token()
        assert post.call_args.kwargs["data"]["client_secret"] == "s3cret"

    def test_invalid_grant_rereads_client_yaml_once(self):
        c = client_with_token_url()
        fresh = Mock(refresh_token="r-new", cached_access_token=None, cached_access_token_expiry=0.0)
        posts = [response(400, INVALID_GRANT), response(200, {"access_token": "a2", "expires_in": 3600})]
        with patch("resource_queries.requests.post", side_effect=posts) as post, patch(
            "resource_queries.Configuration", return_value=fresh
        ):
            assert c._get_access_token() == "a2"
        assert [call.kwargs["data"]["refresh_token"] for call in post.call_args_list] == ["r1", "r-new"]

    def test_invalid_grant_adopts_fresh_cli_access_token(self):
        c = client_with_token_url()
        fresh = Mock(refresh_token="r-new", cached_access_token="cli-a", cached_access_token_expiry=time.time() + 600)
        with patch("resource_queries.requests.post", side_effect=[response(400, INVALID_GRANT)]) as post, patch(
            "resource_queries.Configuration", return_value=fresh
        ):
            assert c._get_access_token() == "cli-a"
        post.assert_called_once()

    def test_invalid_grant_twice_gives_clear_error(self):
        c = client_with_token_url()
        fresh = Mock(refresh_token="r-new", cached_access_token=None, cached_access_token_expiry=0.0)
        posts = [response(400, INVALID_GRANT), response(400, INVALID_GRANT)]
        with patch("resource_queries.requests.post", side_effect=posts), patch(
            "resource_queries.Configuration", return_value=fresh
        ):
            with pytest.raises(AuthenticationError, match="lab-env start"):
                c._get_access_token()

    def test_invalid_grant_with_unchanged_file_gives_clear_error(self):
        c = client_with_token_url()
        fresh = Mock(refresh_token="r1", cached_access_token=None, cached_access_token_expiry=0.0)
        with patch("resource_queries.requests.post", side_effect=[response(400, INVALID_GRANT)]) as post, patch(
            "resource_queries.Configuration", return_value=fresh
        ):
            with pytest.raises(AuthenticationError, match="lab-env start"):
                c._get_access_token()
        post.assert_called_once()

    def test_other_http_error_is_authentication_error(self):
        c = client_with_token_url()
        with patch("resource_queries.requests.post", return_value=response(500, {"error": "server_error"})):
            with pytest.raises(AuthenticationError, match="HTTP 500"):
                c._get_access_token()
