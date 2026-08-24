from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.core.database import get_db
from app.models import AuditLog, ModelAsset, ModelImportJob

router = APIRouter(tags=["健康检查"])


@router.get("/health/live", response_model=dict[str, str])
async def live() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready", response_model=dict[str, Any])
async def ready(session: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    try:
        # 不只检查数据库连接，还检查首个迁移中的核心表，避免未迁移实例被误判为就绪。
        await session.execute(select(ModelAsset.id).limit(1))
        # 审计表缺失时必须拒绝就绪，避免管理写操作在无审计保护的实例上运行。
        await session.execute(select(AuditLog.id).limit(1))
        await session.execute(select(ModelImportJob.id).limit(1))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="数据库尚未就绪",
        ) from exc
    return {"status": "ready", "checks": {"database": "ok"}}
