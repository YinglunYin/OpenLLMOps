import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from ipaddress import ip_address, ip_network

from starlette.requests import Request
from starlette.responses import Response

from app.core.config import Settings, get_settings
from app.core.database import AsyncSessionFactory
from app.models import AuditLog

logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
AUDITED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def resolve_request_id(request: Request) -> str:
    supplied = request.headers.get(get_settings().request_id_header, "")
    if REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return str(uuid.uuid4())


def _is_trusted_proxy(value: str, settings: Settings) -> bool:
    try:
        address = ip_address(value)
    except ValueError:
        return False
    return any(address in ip_network(cidr, strict=False) for cidr in settings.trusted_proxy_cidrs)


def resolve_source_ip(request: Request, settings: Settings | None = None) -> str:
    """只在直连节点可信时读取 X-Forwarded-For，并从右向左剥离可信代理。"""

    active_settings = settings or get_settings()
    direct_peer = request.client.host if request.client else "unknown"
    if not _is_trusted_proxy(direct_peer, active_settings):
        return direct_peer[:64]

    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded:
        return direct_peer[:64]
    forwarded_chain = [item.strip() for item in forwarded.split(",") if item.strip()]
    if not forwarded_chain:
        return direct_peer[:64]
    try:
        # 整条链必须是合法 IP；任何畸形值都会让系统回退到已知的直连代理地址。
        normalized_chain = [str(ip_address(item)) for item in forwarded_chain]
    except ValueError:
        return direct_peer[:64]

    full_chain = [*normalized_chain, direct_peer]
    for candidate in reversed(full_chain):
        if _is_trusted_proxy(candidate, active_settings):
            continue
        return candidate[:64]
    return normalized_chain[0][:64]


def _should_audit(request: Request) -> bool:
    settings = get_settings()
    prefix = f"{settings.api_prefix}/"
    return request.method.upper() in AUDITED_METHODS and request.url.path.startswith(prefix)


def _audit_action(request: Request) -> str:
    route = request.scope.get("route")
    route_name = getattr(route, "name", None)
    if isinstance(route_name, str) and route_name:
        return route_name[:128]
    return f"{request.method.lower()}.unmatched"


async def _persist_audit_log(request: Request, status_code: int) -> None:
    audit_log = AuditLog(
        request_id=request.state.request_id,
        actor=getattr(request.state, "audit_actor", "anonymous"),
        auth_method=getattr(request.state, "audit_auth_method", None),
        action=_audit_action(request),
        method=request.method.upper(),
        path=request.url.path[:1024],
        status_code=status_code,
        succeeded=200 <= status_code < 400,
        source_ip=request.state.source_ip,
    )
    try:
        async with AsyncSessionFactory() as session:
            session.add(audit_log)
            await session.commit()
    except Exception:
        # 审计持久化失败不能把已经提交的管理操作伪装成未执行；生产监控应告警此日志。
        logger.exception(
            "audit persistence failed request_id=%s action=%s status=%s",
            audit_log.request_id,
            audit_log.action,
            audit_log.status_code,
        )


async def request_context_and_audit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request.state.request_id = resolve_request_id(request)
    request.state.source_ip = resolve_source_ip(request)
    request.state.audit_actor = "anonymous"
    request.state.audit_auth_method = None

    try:
        response = await call_next(request)
    except Exception:
        if _should_audit(request):
            await _persist_audit_log(request, 500)
        raise

    response.headers[get_settings().request_id_header] = request.state.request_id
    if _should_audit(request):
        await _persist_audit_log(request, response.status_code)
    return response
