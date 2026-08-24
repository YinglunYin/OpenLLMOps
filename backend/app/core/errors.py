from typing import Any

from fastapi import Request
from fastapi.exception_handlers import (
    http_exception_handler as fastapi_http_exception_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as fastapi_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException


def _openai_error(message: str, error_type: str, *, code: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _error_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


async def openllmops_http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """只为 OpenAI 兼容根路径改写错误，管理 API 保持 FastAPI 标准合同。"""

    if not request.url.path.startswith("/v1/"):
        return await fastapi_http_exception_handler(request, exc)
    if isinstance(exc.detail, dict) and isinstance(exc.detail.get("error"), dict):
        error = dict(exc.detail["error"])
        error.setdefault("param", None)
        error.setdefault("code", None)
        content = {"error": error}
    else:
        message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
        content = _openai_error(message, _error_type(exc.status_code))
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


async def openllmops_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    if not request.url.path.startswith("/v1/"):
        return await fastapi_validation_exception_handler(request, exc)
    first = exc.errors()[0] if exc.errors() else None
    message = first.get("msg", "请求参数无效") if isinstance(first, dict) else "请求参数无效"
    return JSONResponse(
        status_code=400,
        content=_openai_error(str(message), "invalid_request_error"),
    )
