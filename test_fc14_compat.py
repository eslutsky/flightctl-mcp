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
    format_console_result,
    strip_cli_noise,
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

    @patch("resource_queries.requests.get")
    def test_discovery_retried_after_failure(self, mock_get):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("boom"),
            response(200, {"token_endpoint": TOKEN_URL}),
        ]
        client = FlightControlClient(fc14_mock_config())
        fallback = "https://fc.lab.local/_/pam-issuer/api/v1/auth/protocol/openid-connect/token"
        assert client._token_url() == fallback
        assert client._token_url() == TOKEN_URL
        assert mock_get.call_count == 2
        assert client._token_url() == TOKEN_URL
        assert mock_get.call_count == 2  # success is cached

    @patch("resource_queries.requests.get")
    def test_discovery_without_token_endpoint_falls_back(self, mock_get):
        mock_get.return_value = response(200, {})
        client = FlightControlClient(fc14_mock_config())
        assert client._token_url() == "https://fc.lab.local/_/pam-issuer/api/v1/auth/protocol/openid-connect/token"
        assert client._discovered_token_url is None


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


DEVICE = {"metadata": {"name": "o1top", "labels": {"alias": "ushift-1"}}}


class TestOrgAndDeviceLookup:
    def test_queries_send_org_id(self):
        c = client_with_token_url(organization="org-1")
        with patch.object(c, "_get_access_token", return_value="t"), patch(
            "resource_queries.requests.get", return_value=response(200, {"items": []})
        ) as get:
            c.query_devices(label_selector="fleet=lab")
        assert get.call_args.kwargs["params"] == {"labelSelector": "fleet=lab", "org_id": "org-1"}
        assert get.call_args.kwargs["verify"] == "/tmp/api-ca.crt"

    def test_no_org_id_when_unset(self):
        c = client_with_token_url()
        with patch.object(c, "_get_access_token", return_value="t"), patch(
            "resource_queries.requests.get", return_value=response(200, {"items": []})
        ) as get:
            c.query_devices(label_selector="fleet=lab")
        assert get.call_args.kwargs["params"] == {"labelSelector": "fleet=lab"}

    def test_get_device_found_and_missing(self):
        c = client_with_token_url(organization="org-1")
        with patch.object(c, "_get_access_token", return_value="t"), patch(
            "resource_queries.requests.get", side_effect=[response(200, DEVICE), response(404, {})]
        ) as get:
            assert c.get_device("o1top") == DEVICE
            assert c.get_device("nosuch") is None
        assert get.call_args_list[0].args[0] == "https://fc.lab.local:3443/api/v1/devices/o1top"
        assert get.call_args_list[0].kwargs["params"] == {"org_id": "org-1"}

    def test_get_device_auth_error(self):
        c = client_with_token_url()
        with patch.object(c, "_get_access_token", return_value="t"), patch(
            "resource_queries.requests.get", return_value=response(401, {})
        ):
            with pytest.raises(AuthenticationError):
                c.get_device("o1top")

    def test_resolve_by_alias(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices", return_value=[DEVICE]) as q, patch.object(c, "get_device") as g:
            assert c.resolve_device("ushift-1") is DEVICE
        q.assert_called_once_with(label_selector="alias=ushift-1", limit=2)
        g.assert_not_called()

    def test_resolve_by_name_when_no_alias_matches(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices", return_value=[]), patch.object(c, "get_device", return_value=DEVICE) as g:
            assert c.resolve_device("o1top") is DEVICE
        g.assert_called_once_with("o1top")

    def test_resolve_ambiguous_alias(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices", return_value=[DEVICE, DEVICE]):
            with pytest.raises(FlightControlError, match="more than one device"):
                c.resolve_device("ushift-1")

    def test_resolve_unknown(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices", return_value=[]), patch.object(c, "get_device", return_value=None):
            with pytest.raises(FlightControlError, match="No device named or aliased 'ghost'"):
                c.resolve_device("ghost")

    def test_resolve_skips_alias_lookup_for_non_label_values(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices") as q, patch.object(c, "get_device", return_value=None):
            with pytest.raises(FlightControlError):
                c.resolve_device("bad name!")
        q.assert_not_called()

    def test_resolve_rejects_trailing_newline_alias(self):
        c = client_with_token_url()
        with patch.object(c, "query_devices") as q, patch.object(c, "get_device", return_value=None):
            with pytest.raises(FlightControlError):
                c.resolve_device("ushift-1\n")
        q.assert_not_called()

    @pytest.mark.parametrize("name", [".", "..", "  ", " . "])
    def test_resolve_rejects_unusable_names(self, name):
        c = client_with_token_url()
        with patch.object(c, "query_devices") as q, patch.object(c, "get_device") as g:
            with pytest.raises(FlightControlError, match="Invalid device name"):
                c.resolve_device(name)
        q.assert_not_called()
        g.assert_not_called()


class TestTokenResponse:
    def test_null_expires_in_defaults(self):
        c = client_with_token_url()
        with patch("resource_queries.requests.post", return_value=response(200, {"access_token": "a", "expires_in": None})):
            before = time.time()
            assert c._refresh_access_token() == "a"
        assert 3590 < c._token_expiry - before < 3610

    def test_non_json_body(self):
        c = client_with_token_url()
        r = response(200, {})
        r.json.side_effect = ValueError("no json")
        with patch("resource_queries.requests.post", return_value=r):
            with pytest.raises(AuthenticationError, match="not JSON"):
                c._refresh_access_token()


def test_cli_subprocesses_do_not_inherit_stdin():
    from cli import FlightctlCLI

    with patch("cli.shutil.which", return_value=None), patch("cli.subprocess.run") as run:
        try:
            FlightctlCLI("https://api.test.com").download()
        except Exception:
            pass
    assert run.call_args_list
    assert all(c.kwargs["stdin"] is subprocess.DEVNULL for c in run.call_args_list)


ONLINE = {"metadata": {"name": "o1top"}, "status": {"summary": {"status": "Online", "info": "ok"}}}
OFFLINE = {
    "metadata": {"name": "eh5t"},
    "status": {"summary": {"status": "Unknown", "info": "Device is disconnected."}},
}
NOISE = 'time="2026-11-25T16:05:16Z" level=info msg="no refresh token found, skipping token refresh"\n'


def completed(rc=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr=err)


@pytest.fixture
def console_client(tmp_path):
    c = client_with_token_url(organization="org-1")
    c.console_config_dir = tmp_path / "fcmcp"
    return c


def run_console(c, side_effect, device=ONLINE, command="uptime", **kw):
    with patch.object(c, "resolve_device", return_value=device), patch.object(
        c, "_get_access_token", return_value="tok-1"
    ), patch("resource_queries.shutil.which", return_value="/usr/bin/flightctl"), patch(
        "resource_queries.subprocess.run", side_effect=side_effect
    ) as run:
        out = c.run_console_command("/usr/bin/flightctl", "ushift-1", command, **kw)
    return out, run


class TestConsole:
    def test_login_then_console_with_private_config_dir(self, console_client):
        seen = {}

        def fake_run(cmd, **kw):
            if cmd[1] == "login":
                cred = Path(cmd[cmd.index("--credentials-file") + 1])
                seen["cred"] = json.loads(cred.read_text())
                seen["mode"] = cred.stat().st_mode & 0o777
            seen.setdefault("stdin", []).append(kw.get("stdin"))
            return completed(0, "up 6:38\n", NOISE)

        out, run = run_console(console_client, fake_run, command="oc get pods -A | wc -l")
        d = str(console_client.console_config_dir)
        login, console = run.call_args_list[0].args[0], run.call_args_list[1].args[0]
        cred_path = Path(login[login.index("--credentials-file") + 1])
        assert cred_path.parent == console_client.console_config_dir
        assert cred_path.name.startswith("credentials-")
        assert login == [
            "/usr/bin/flightctl", "login", "https://fc.lab.local:3443",
            "--config-dir", d, "--credentials-file", str(cred_path),
            "--certificate-authority", "/tmp/api-ca.crt", "--org", "org-1",
        ]
        assert console == [
            "/usr/bin/flightctl", "console", "device/o1top", "--config-dir", d, "--notty",
            "--org", "org-1", "--", "sh", "-c", "oc get pods -A | wc -l",
        ]
        assert seen == {"cred": {"token": "tok-1"}, "mode": 0o600, "stdin": [subprocess.DEVNULL] * 2}
        assert not list(console_client.console_config_dir.glob("credentials-*"))
        assert console_client.console_config_dir.stat().st_mode & 0o777 == 0o700
        assert "tok-1" not in " ".join(login + console)
        assert out == "up 6:38"

    def test_logs_in_before_every_command(self, console_client):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd[1])
            return completed(0, "x")

        run_console(console_client, fake_run)
        run_console(console_client, fake_run)
        assert calls == ["login", "console", "login", "console"]

    def test_offline_device_refused_before_console(self, console_client):
        with patch.object(console_client, "resolve_device", return_value=OFFLINE), patch(
            "resource_queries.shutil.which", return_value="/usr/bin/flightctl"
        ), patch("resource_queries.subprocess.run") as run:
            with pytest.raises(FlightControlError, match=r"is Unknown: Device is disconnected\. Commands need.*last-known status"):
                console_client.run_console_command("/usr/bin/flightctl", "ushift-3", "uptime")
        run.assert_not_called()

    def test_nonzero_exit_returns_code_and_streams(self, console_client):
        def fake_run(cmd, **kw):
            return completed(0) if cmd[1] == "login" else completed(3, "out\n", NOISE + "err\n")

        out, _ = run_console(console_client, fake_run)
        assert out == "exit code: 3\n--- stdout ---\nout\n--- stderr ---\nerr"

    def test_timeout(self, console_client):
        def fake_run(cmd, **kw):
            if cmd[1] == "console":
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])
            return completed(0)

        with pytest.raises(FlightControlError, match="timed out after 5s"):
            run_console(console_client, fake_run, timeout=5)

    def test_login_failure_cleans_up_credentials(self, console_client):
        def fake_run(cmd, **kw):
            raise subprocess.CalledProcessError(1, cmd, stderr=NOISE + "Error: unauthorized\n")

        with pytest.raises(FlightControlError, match="Failed to login to flightctl console: Error: unauthorized"):
            run_console(console_client, fake_run)
        assert not list(console_client.console_config_dir.glob("credentials-*"))

    def test_empty_command_rejected(self, console_client):
        with patch("resource_queries.shutil.which", return_value="/usr/bin/flightctl"):
            with pytest.raises(FlightControlError, match="Command cannot be empty"):
                console_client.run_console_command("/usr/bin/flightctl", "ushift-1", "  ")


class TestConsoleHelpers:
    def test_strip_cli_noise(self):
        assert strip_cli_noise(NOISE + "real error\n" + 'time="t" level=debug msg="b"\n') == "real error\n"

    def test_clean_success_is_just_stdout(self):
        assert format_console_result(0, "hello\n", "", max_output=100) == "hello"

    def test_output_cap(self):
        assert format_console_result(0, "a" * 100, "", max_output=10) == "a" * 10 + "\n[... truncated: 90 more bytes]"


class TestCliStdoutClean:
    def test_skip_existing_prints_nothing(self, capsys):
        from cli import FlightctlCLI

        with patch("cli.shutil.which", return_value="/usr/bin/flightctl"):
            FlightctlCLI("https://api.test.com").download()
        assert capsys.readouterr().out == ""
