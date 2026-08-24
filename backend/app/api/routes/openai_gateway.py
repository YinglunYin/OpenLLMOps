from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import Deployment
from app.models.enums import DeploymentState, DeploymentTaskType
from app.schemas import OpenAIProxyRequest

router = APIRouter(tags=["OpenAI Compatible API"])


async def _find_deployment(
    model_name: str,
    task_type: DeploymentTaskType,
    session: AsyncSession,
) -> Deployment:
    deployment = await session.scalar(
        select(Deployment).where(
            Deployment.served_model_name == model_name,
            Deployment.task_type == task_type,
            Deployment.actual_state == DeploymentState.RUNNING,
        )
    )
    if deployment is None or not deployment.internal_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"模型 {model_name} 没有可用的运行实例",
                    "type": "model_not_found",
                }
            },
        )
    return deployment


async def _proxy(
    endpoint: str,
    body: OpenAIProxyRequest,
    task_type: DeploymentTaskType,
    session: AsyncSession,
) -> Response:
    deployment = await _find_deployment(body.model, task_type, session)
    settings = get_settings()
    payload: dict[str, Any] = body.model_dump()
    is_streaming = bool(payload.get("stream", False))
    headers = {"Content-Type": "application/json"}
    if settings.vllm_internal_api_key:
        headers["Authorization"] = f"Bearer {settings.vllm_internal_api_key}"

    client = httpx.AsyncClient(timeout=settings.proxy_timeout_seconds)
    upstream_url = f"{deployment.internal_url.rstrip('/')}{endpoint}"
    try:
        request = client.build_request("POST", upstream_url, json=payload, headers=headers)
        upstream = await client.send(request, stream=is_streaming)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"推理服务不可达：{exc}") from exc

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() in {"content-type", "x-request-id"}
    }
    if not is_streaming:
        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


@router.post("/v1/completions")
async def completions(
    body: OpenAIProxyRequest,
    session: AsyncSession = Depends(get_db),
) -> Response:
    return await _proxy("/v1/completions", body, DeploymentTaskType.GENERATE, session)


@router.post("/v1/chat/completions")
async def chat_completions(
    body: OpenAIProxyRequest,
    session: AsyncSession = Depends(get_db),
) -> Response:
    return await _proxy("/v1/chat/completions", body, DeploymentTaskType.GENERATE, session)


@router.post("/v1/embeddings")
async def embeddings(
    body: OpenAIProxyRequest,
    session: AsyncSession = Depends(get_db),
) -> Response:
    return await _proxy("/v1/embeddings", body, DeploymentTaskType.EMBEDDING, session)
