import time
from collections.abc import Awaitable, Callable

from prometheus_client import Counter, Histogram
from starlette.requests import Request
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "openllmops_http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)
REQUEST_DURATION = Histogram(
    "openllmops_http_request_duration_seconds",
    "HTTP 请求耗时",
    ["method", "path"],
)


async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    # 使用路由模板而不是原始 UUID，避免 Prometheus 标签基数无限增长。
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_DURATION.labels(request.method, path).observe(time.perf_counter() - started)
    return response
