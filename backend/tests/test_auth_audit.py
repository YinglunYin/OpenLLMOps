from datetime import UTC, datetime, timedelta

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from app.core.config import Settings, get_settings
from app.core.request_context import resolve_source_ip
from app.core.security import (
    SessionTokenError,
    create_admin_session,
    verify_admin_session,
)


@pytest.fixture
def configured_admin(client: TestClient):  # type: ignore[no-untyped-def]
    settings = get_settings()
    previous = {
        "auth_enabled": settings.auth_enabled,
        "admin_username": settings.admin_username,
        "admin_password_hash": settings.admin_password_hash,
        "admin_api_key": settings.admin_api_key,
        "session_signing_key": settings.session_signing_key,
        "session_cookie_secure": settings.session_cookie_secure,
        "session_ttl_seconds": settings.session_ttl_seconds,
    }
    settings.auth_enabled = True
    settings.admin_username = "admin"
    settings.admin_password_hash = PasswordHasher().hash("correct horse battery staple")
    settings.admin_api_key = "bootstrap-test-key"
    settings.session_signing_key = "test-session-signing-key-which-is-long-enough-123456"
    settings.session_cookie_secure = True
    settings.session_ttl_seconds = 3600
    client.cookies.clear()
    try:
        yield settings
    finally:
        client.cookies.clear()
        for field, value in previous.items():
            setattr(settings, field, value)


def test_login_session_csrf_logout_and_audit(
    client: TestClient,
    configured_admin: Settings,
) -> None:
    failed = client.post(
        "/api/v1/auth/login",
        headers={"X-Request-ID": "login-failed-request"},
        json={"username": "admin", "password": "bad-password"},
    )
    assert failed.status_code == 401
    assert failed.headers["X-Request-ID"] == "login-failed-request"

    cross_site = client.post(
        "/api/v1/auth/login",
        headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
        json={"username": "admin", "password": "correct horse battery staple"},
    )
    assert cross_site.status_code == 403

    logged_in = client.post(
        "/api/v1/auth/login",
        headers={"X-Request-ID": "login-success-request"},
        json={"username": "admin", "password": "correct horse battery staple"},
    )
    assert logged_in.status_code == 200, logged_in.text
    cookie_header = logged_in.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "secure" in cookie_header
    assert "samesite=strict" in cookie_header
    assert "max-age=3600" in cookie_header
    csrf_token = logged_in.json()["csrf_token"]
    assert csrf_token

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["auth_method"] == "session"
    assert me.json()["csrf_token"] == csrf_token
    # 浏览器会话 Cookie 不能被拿来调用 OpenAI 兼容端点。
    assert (
        client.post(
            "/v1/completions",
            json={"model": "not-deployed", "prompt": "hello"},
        ).status_code
        == 401
    )

    model_payload = {
        "name": "audit model",
        "source_type": "manual",
        "local_path": "/srv/openllmops/models/audit-model",
        "model_kind": "instruct",
        "status": "ready",
    }
    missing_csrf = client.post(
        "/api/v1/model-assets",
        headers={"X-Request-ID": "csrf-rejected-request"},
        json=model_payload,
    )
    assert missing_csrf.status_code == 403

    created = client.post(
        "/api/v1/model-assets",
        headers={
            configured_admin.csrf_header: csrf_token,
            "Origin": "http://localhost:5173",
            "X-Request-ID": "model-create-request",
        },
        json=model_payload,
    )
    assert created.status_code == 201, created.text

    audit = client.get("/api/v1/audit-logs?request_id=login-failed-request")
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    failed_entry = audit.json()[0]
    assert failed_entry["action"] == "auth.login"
    assert failed_entry["status_code"] == 401
    assert failed_entry["succeeded"] is False
    assert "bad-password" not in str(failed_entry)
    assert "cookie" not in str(failed_entry).lower()

    created_audit = client.get("/api/v1/audit-logs?request_id=model-create-request").json()
    assert len(created_audit) == 1
    assert created_audit[0]["actor"] == "admin:admin"
    assert created_audit[0]["auth_method"] == "session"
    assert created_audit[0]["status_code"] == 201
    rejected_audit = client.get("/api/v1/audit-logs?request_id=csrf-rejected-request").json()
    assert len(rejected_audit) == 1
    assert rejected_audit[0]["actor"] == "admin:admin"
    assert rejected_audit[0]["status_code"] == 403

    logout_without_csrf = client.post("/api/v1/auth/logout")
    assert logout_without_csrf.status_code == 403
    logout = client.post(
        "/api/v1/auth/logout",
        headers={configured_admin.csrf_header: csrf_token},
    )
    assert logout.status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 401


def test_signed_session_expiry_and_tampering(configured_admin: Settings) -> None:
    issued_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    token, identity = create_admin_session("admin", now=issued_at)
    assert verify_admin_session(token, now=issued_at).session_id == identity.session_id

    version, payload, signature = token.split(".")
    replacement = "A" if payload[-1] != "A" else "B"
    tampered = f"{version}.{payload[:-1]}{replacement}.{signature}"
    with pytest.raises(SessionTokenError):
        verify_admin_session(tampered, now=issued_at)
    with pytest.raises(SessionTokenError, match="过期"):
        verify_admin_session(
            token,
            now=issued_at + timedelta(seconds=configured_admin.session_ttl_seconds + 1),
        )


def test_production_auth_configuration_has_no_plaintext_default() -> None:
    password_hash = PasswordHasher().hash("production-password")
    production = Settings(
        _env_file=None,
        environment="production",
        auth_enabled=True,
        admin_password_hash=password_hash,
        session_signing_key="p" * 48,
        cors_origins=["https://openllmops.internal"],
    )
    assert production.admin_password_hash.startswith("$argon2")
    assert not hasattr(production, "admin_password")

    with pytest.raises(ValidationError, match="ADMIN_PASSWORD_HASH"):
        Settings(
            _env_file=None,
            environment="production",
            auth_enabled=True,
            admin_password_hash=None,
            session_signing_key="p" * 48,
            cors_origins=["https://openllmops.internal"],
        )
    with pytest.raises(ValidationError, match="有效的 Argon2"):
        Settings(
            _env_file=None,
            environment="production",
            auth_enabled=True,
            admin_password_hash="$argon2id$invalid",
            session_signing_key="p" * 48,
            cors_origins=["https://openllmops.internal"],
        )


def _request(client_ip: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for:
        headers.append((b"x-forwarded-for", forwarded_for.encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
            "client": (client_ip, 12345),
            "server": ("openllmops.internal", 443),
        }
    )


def test_forwarded_ip_is_used_only_behind_trusted_proxy() -> None:
    trusted_settings = Settings(
        _env_file=None,
        environment="test",
        trusted_proxy_cidrs=["10.0.0.0/8"],
    )
    trusted_request = _request("10.0.0.2", "203.0.113.9, 10.0.0.3")
    assert resolve_source_ip(trusted_request, trusted_settings) == "203.0.113.9"

    untrusted_request = _request("192.0.2.20", "203.0.113.9")
    assert resolve_source_ip(untrusted_request, trusted_settings) == "192.0.2.20"
    malformed_request = _request("10.0.0.2", "not-an-ip")
    assert resolve_source_ip(malformed_request, trusted_settings) == "10.0.0.2"


def test_credentialed_cors_uses_explicit_origin(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-csrf-token",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert allowed.headers["access-control-allow-credentials"] == "true"

    denied = client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
