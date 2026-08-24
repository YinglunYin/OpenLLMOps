import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import ModelAsset
from app.models.enums import AssetStatus
from app.schemas import ModelAssetRead, ModelAssetUpdate
from app.services.crud import commit_or_conflict
from app.services.model_assets import (
    get_active_model_asset,
    model_asset_has_operational_references,
)

router = APIRouter(prefix="/model-assets", tags=["模型资产"])


@router.get("", response_model=list[ModelAssetRead])
async def list_model_assets(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[ModelAsset]:
    result = await session.scalars(
        select(ModelAsset)
        .where(ModelAsset.deleted_at.is_(None))
        .order_by(ModelAsset.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result)


@router.get("/{asset_id}", response_model=ModelAssetRead)
async def get_model_asset(
    asset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    return await get_active_model_asset(session, asset_id)


@router.patch("/{asset_id}", response_model=ModelAssetRead)
async def update_model_asset(
    asset_id: uuid.UUID,
    payload: ModelAssetUpdate,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    asset = await get_active_model_asset(session, asset_id, for_update=True)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(asset, field, value)
    await commit_or_conflict(session, "模型资产更新冲突")
    await session.refresh(asset)
    return asset


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_asset(
    asset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Response:
    asset = await get_active_model_asset(session, asset_id, for_update=True)
    if await model_asset_has_operational_references(session, asset.id):
        raise HTTPException(status_code=409, detail="模型正在被部署、训练或评测引用，不能删除")

    # 管理界面的“删除”是软删除：目录可能很大且仍可能被管理员用于离线核验，
    # 因此这里只隐藏数据库资产，绝不在请求线程中递归删除实体文件。
    asset.deleted_at = datetime.now(UTC)
    asset.status = AssetStatus.FAILED
    asset.error_message = "管理员已软删除；模型实体文件仍保留在受控目录"
    await commit_or_conflict(session, "模型资产删除冲突")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
