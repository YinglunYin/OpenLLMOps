import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import APIKey

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SESSION_VERSION = 1
password_hasher = PasswordHasher()


@dataclass(frozen=True)
class AdminIdentity:
    username: str
    auth_method: Literal["session", "bootstrap_key", "disabled"]
    expires_at: datetime | None = None
    session_id: str | None = None
    csrf_token: str | None = None


class SessionTokenError(ValueError):
    """签名、格式或有效期不合法时统一抛出，外部响应不泄漏具体原因。"""


def hash_api_key(raw_key: str) -> str:
    settings = get_settings()
    payload = f"{settings.api_key_pepper}:{raw_key}".encode()
    return hashlib.sha256(payload).hexdigest()


def issue_api_key() -> tuple[str, str, str]:
    """返回（明文、展示前缀、摘要）；调用者只持久化后两项。"""

    raw_key = f"ollm_{secrets.token_urlsafe(32)}"
    return raw_key, raw_key[:12], hash_api_key(raw_key)


def extract_api_key(request: Request) -> str | None:
    settings = get_settings()
    token = request.headers.get(settings.api_key_header)
    if token:
        return token.strip()
    authorization = request.headers.get("Authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer" and credentials:
        return credentials.strip()
    return None


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(f"{value}{padding}", altchars=b"-_", validate=True)


def _session_signing_key() -> bytes:
    signing_key = get_settings().session_signing_key
    if not signing_key or len(signing_key) < 32:
        raise SessionTokenError("管理员会话签名密钥未配置或长度不足")
    return signing_key.encode("utf-8")


def create_admin_session(username: str, now: datetime | None = None) -> tuple[str, AdminIdentity]:
    settings = get_settings()
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at.timestamp() + settings.session_ttl_seconds
    payload = {
        "v": SESSION_VERSION,
        "sub": username,
        "sid": secrets.token_urlsafe(18),
        "csrf": secrets.token_urlsafe(32),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at),
    }
    encoded_payload = _base64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signed_value = f"v{SESSION_VERSION}.{encoded_payload}"
    signature = hmac.new(_session_signing_key(), signed_value.encode("ascii"), hashlib.sha256).digest()
    token = f"{signed_value}.{_base64url_encode(signature)}"
    identity = AdminIdentity(
        username=username,
        auth_method="session",
        expires_at=datetime.fromtimestamp(expires_at, UTC),
        session_id=payload["sid"],
        csrf_token=payload["csrf"],
    )
    return token, identity


def verify_admin_session(token: str, now: datetime | None = None) -> AdminIdentity:
    if len(token) > 4096:
        raise SessionTokenError("会话 Cookie 过长")
    try:
        version, encoded_payload, encoded_signature = token.split(".", 2)
        if version != f"v{SESSION_VERSION}":
            raise SessionTokenError("会话版本不受支持")
        signed_value = f"{version}.{encoded_payload}"
        expected = hmac.new(_session_signing_key(), signed_value.encode("ascii"), hashlib.sha256).digest()
        supplied = _base64url_decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            raise SessionTokenError("会话签名无效")
        payload = json.loads(_base64url_decode(encoded_payload))
        current_timestamp = int((now or datetime.now(UTC)).timestamp())
        if payload.get("v") != SESSION_VERSION:
            raise SessionTokenError("会话载荷版本无效")
        if not isinstance(payload.get("iat"), int) or payload["iat"] > current_timestamp + 60:
            raise SessionTokenError("会话签发时间无效")
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= current_timestamp:
            raise SessionTokenError("会话已过期")
        settings = get_settings()
        if payload.get("sub") != settings.admin_username:
            raise SessionTokenError("会话管理员不匹配")
        if not isinstance(payload.get("sid"), str) or not isinstance(payload.get("csrf"), str):
            raise SessionTokenError("会话载荷不完整")
        return AdminIdentity(
            username=payload["sub"],
            auth_method="session",
            expires_at=datetime.fromtimestamp(payload["exp"], UTC),
            session_id=payload["sid"],
            csrf_token=payload["csrf"],
        )
    except SessionTokenError:
        raise
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SessionTokenError("会话格式无效") from exc


def verify_admin_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    password_hash = settings.admin_password_hash
    if not password_hash:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理员密码尚未配置",
        )
    try:
        password_matches = password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        password_matches = False
    except (InvalidHashError, VerificationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理员密码哈希配置无效",
        ) from exc
    return hmac.compare_digest(username, settings.admin_username) and password_matches


def validate_browser_origin(request: Request) -> None:
    settings = get_settings()
    fetch_site = request.headers.get("Sec-Fetch-Site", "").lower()
    if fetch_site == "cross-site":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="拒绝跨站请求")
    origin = request.headers.get("Origin")
    if origin is None:
        # 非浏览器客户端通常没有 Origin；会话写请求仍必须提供不可猜测的 CSRF token。
        return
    normalized_origin = origin.rstrip("/")
    allowed = {item.rstrip("/") for item in settings.cors_origins}
    if normalized_origin not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="请求来源不受信任")


async def require_api_key(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> None:
    """同时接受环境变量中的管理员密钥和数据库签发密钥。"""

    settings = get_settings()
    if not settings.auth_enabled:
        return

    token = extract_api_key(request)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少 API Key")

    if settings.admin_api_key and hmac.compare_digest(token, settings.admin_api_key):
        return

    key_hash = hash_api_key(token)
    api_key = await session.scalar(
        select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active.is_(True))
    )
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key 无效")

    api_key.last_used_at = datetime.now(UTC)
    await session.commit()


async def require_admin_auth(request: Request) -> AdminIdentity:
    """管理面仅接受 bootstrap 管理员密钥或已签名浏览器会话。"""

    existing = getattr(request.state, "admin_identity", None)
    if isinstance(existing, AdminIdentity):
        return existing

    settings = get_settings()
    if not settings.auth_enabled:
        identity = AdminIdentity(username=settings.admin_username, auth_method="disabled")
        request.state.admin_identity = identity
        request.state.audit_actor = f"admin:{settings.admin_username}"
        request.state.audit_auth_method = identity.auth_method
        return identity

    token = extract_api_key(request)
    if settings.admin_api_key and token and hmac.compare_digest(token, settings.admin_api_key):
        identity = AdminIdentity(
            username=settings.admin_username,
            auth_method="bootstrap_key",
        )
        request.state.admin_identity = identity
        request.state.audit_actor = f"admin:{settings.admin_username}"
        request.state.audit_auth_method = identity.auth_method
        return identity

    raw_session = request.cookies.get(settings.session_cookie_name)
    if raw_session:
        try:
            identity = verify_admin_session(raw_session)
        except SessionTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="管理员会话无效或已过期",
            ) from exc

        request.state.admin_identity = identity
        request.state.audit_actor = f"admin:{identity.username}"
        request.state.audit_auth_method = identity.auth_method
        if request.method.upper() in UNSAFE_METHODS:
            validate_browser_origin(request)
            supplied_csrf = request.headers.get(settings.csrf_header, "")
            if not identity.csrf_token or not hmac.compare_digest(supplied_csrf, identity.csrf_token):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token 无效")

        return identity

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要管理员认证")
