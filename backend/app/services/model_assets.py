from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Deployment, EvaluationRun, ModelAsset, TrainingJob


async def get_active_model_asset(
    session: AsyncSession,
    asset_id: uuid.UUID,
    label: str = "模型资产",
    *,
    for_update: bool = False,
) -> ModelAsset:
    statement = select(ModelAsset).where(
        ModelAsset.id == asset_id,
        ModelAsset.deleted_at.is_(None),
    )
    if for_update:
        statement = statement.with_for_update()
    asset = await session.scalar(statement)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"{label}不存在")
    return asset


async def get_active_model_assets_for_update(
    session: AsyncSession,
    requested: dict[uuid.UUID, str],
) -> dict[uuid.UUID, ModelAsset]:
    """按稳定顺序锁定多个可见资产，避免评测创建与软删除竞态/死锁。"""

    asset_ids = sorted(requested, key=str)
    rows = list(
        await session.scalars(
            select(ModelAsset)
            .where(
                ModelAsset.id.in_(asset_ids),
                ModelAsset.deleted_at.is_(None),
            )
            .order_by(ModelAsset.id)
            .with_for_update()
        )
    )
    assets = {asset.id: asset for asset in rows}
    for asset_id in asset_ids:
        if asset_id not in assets:
            raise HTTPException(status_code=404, detail=f"{requested[asset_id]}不存在")
    return assets


async def model_asset_has_operational_references(
    session: AsyncSession,
    asset_id: uuid.UUID,
) -> bool:
    checks = (
        select(Deployment.id).where(Deployment.model_asset_id == asset_id).limit(1),
        select(TrainingJob.id)
        .where(
            or_(
                TrainingJob.model_asset_id == asset_id,
                TrainingJob.published_model_asset_id == asset_id,
            )
        )
        .limit(1),
        select(EvaluationRun.id)
        .where(
            or_(
                EvaluationRun.base_model_asset_id == asset_id,
                EvaluationRun.candidate_model_asset_id == asset_id,
            )
        )
        .limit(1),
    )
    for statement in checks:
        if await session.scalar(statement) is not None:
            return True
    return False
