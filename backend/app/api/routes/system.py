import math
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import GPULease
from app.schemas import GPUHistoryMetric, GPUHistoryRead, GPULeaseRead, GPUStatusRead
from app.services.gpu_monitoring import (
    MAX_HISTORY_POINTS,
    MAX_HISTORY_SPAN_SECONDS,
    MAX_HISTORY_STEP_SECONDS,
    MIN_HISTORY_STEP_SECONDS,
    get_gpu_history,
    get_gpu_statuses,
)
from app.services.prometheus import PrometheusClient, get_prometheus_client

router = APIRouter(prefix="/system", tags=["系统"])


@router.get("/capabilities", response_model=dict[str, Any])
async def get_capabilities() -> dict[str, Any]:
    settings = get_settings()
    return {
        "gpu_count": settings.gpu_count,
        "gpu_policy": "exclusive_non_preemptive",
        "model_format": ["safetensors"],
        "trust_remote_code": False,
        "training": {
            "cpt": ["lora"],
            "sft": ["freeze", "lora", "qlora"],
        },
        "evaluation": ["ceval", "cmmlu", "custom_jsonl"],
        "openai_endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/embeddings",
        ],
    }


@router.get("/gpu-leases", response_model=list[GPULeaseRead])
async def list_gpu_leases(
    session: AsyncSession = Depends(get_db),
) -> list[GPULease]:
    result = await session.scalars(select(GPULease).order_by(GPULease.gpu_index))
    return list(result)


@router.get("/gpus", response_model=list[GPUStatusRead], name="system.gpus")
async def list_gpu_statuses(
    session: AsyncSession = Depends(get_db),
    prometheus: PrometheusClient | None = Depends(get_prometheus_client),
) -> list[GPUStatusRead]:
    settings = get_settings()
    return await get_gpu_statuses(session, prometheus, settings.gpu_count)


@router.get(
    "/gpus/{gpu_index}/history",
    response_model=GPUHistoryRead,
    name="system.gpu_history",
)
async def query_gpu_history(
    gpu_index: int = Path(ge=0),
    metric: GPUHistoryMetric = Query(),
    start: datetime = Query(),
    end: datetime = Query(),
    step_seconds: int = Query(default=30, ge=MIN_HISTORY_STEP_SECONDS, le=MAX_HISTORY_STEP_SECONDS),
    prometheus: PrometheusClient | None = Depends(get_prometheus_client),
) -> GPUHistoryRead:
    settings = get_settings()
    if gpu_index >= settings.gpu_count:
        raise HTTPException(status_code=422, detail="GPU 编号超出本机范围")
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(status_code=422, detail="历史查询时间必须包含时区")
    normalized_start = start.astimezone(UTC)
    normalized_end = end.astimezone(UTC)
    span_seconds = (normalized_end - normalized_start).total_seconds()
    if span_seconds <= 0:
        raise HTTPException(status_code=422, detail="end 必须晚于 start")
    if span_seconds > MAX_HISTORY_SPAN_SECONDS:
        raise HTTPException(status_code=422, detail="历史查询时间跨度不能超过 7 天")
    point_count = math.floor(span_seconds / step_seconds) + 1
    if point_count > MAX_HISTORY_POINTS:
        raise HTTPException(status_code=422, detail=f"历史查询数据点不能超过 {MAX_HISTORY_POINTS}")
    return await get_gpu_history(
        prometheus,
        gpu_index=gpu_index,
        metric=metric,
        start=normalized_start,
        end=normalized_end,
        step_seconds=step_seconds,
        max_points=point_count,
    )
