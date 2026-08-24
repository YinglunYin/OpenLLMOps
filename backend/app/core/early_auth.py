from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.core.security import require_admin_auth

DATASET_UPLOAD_PATH = "/api/v1/datasets/upload"


async def early_large_upload_auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """在 Starlette 解析/落盘大 multipart body 之前完成管理员鉴权。

    FastAPI 的路由依赖会在表单参数解析后执行，单靠 Depends 会允许未认证请求先占用
    最多 5 GiB 暂存盘。这里仅检查 header/cookie/CSRF，不读取 request body；后续依赖
    会复用写入 request.state 的身份。
    """

    if request.method == "POST" and request.url.path == DATASET_UPLOAD_PATH:
        try:
            await require_admin_auth(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
    return await call_next(request)
