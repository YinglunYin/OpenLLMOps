from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.core.config import get_settings
from app.core.security import (
    AdminIdentity,
    SessionTokenError,
    create_admin_session,
    require_admin_auth,
    validate_browser_origin,
    verify_admin_credentials,
)
from app.schemas import AdminIdentityRead, LoginRequest, Message

router = APIRouter(prefix="/auth", tags=["管理员认证"])


def _identity_response(identity: AdminIdentity) -> AdminIdentityRead:
    return AdminIdentityRead(
        username=identity.username,
        auth_method=identity.auth_method,
        expires_at=identity.expires_at,
        csrf_token=identity.csrf_token,
    )


@router.post("/login", response_model=AdminIdentityRead, name="auth.login")
async def login(payload: LoginRequest, request: Request, response: Response) -> AdminIdentityRead:
    settings = get_settings()
    validate_browser_origin(request)
    if not settings.auth_enabled:
        raise HTTPException(status_code=503, detail="当前环境未启用管理员登录")
    if not verify_admin_credentials(payload.username, payload.password.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    try:
        token, identity = create_admin_session(settings.admin_username)
    except SessionTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理员会话签名尚未正确配置",
        ) from exc

    request.state.audit_actor = f"admin:{identity.username}"
    request.state.audit_auth_method = identity.auth_method
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_ttl_seconds,
        expires=identity.expires_at,
        # Cookie 仅发送到管理 API，浏览器访问 OpenAI `/v1/*` 时不会附带它。
        path=settings.api_prefix,
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return _identity_response(identity)


@router.post("/logout", response_model=Message, name="auth.logout")
async def logout(
    response: Response,
    _identity: AdminIdentity = Depends(require_admin_auth),
) -> Message:
    settings = get_settings()
    response.delete_cookie(
        key=settings.session_cookie_name,
        path=settings.api_prefix,
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return Message(message="已退出管理员会话")


@router.get("/me", response_model=AdminIdentityRead, name="auth.me")
async def me(identity: AdminIdentity = Depends(require_admin_auth)) -> AdminIdentityRead:
    return _identity_response(identity)
