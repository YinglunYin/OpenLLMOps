import uuid

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import ModelAsset
from app.schemas import ModelAssetCreate, ModelAssetRead, ModelAssetUpdate
from app.services.crud import commit_or_conflict, get_or_404

router = APIRouter(prefix="/model-assets", tags=["模型资产"])


@router.post("", response_model=ModelAssetRead, status_code=status.HTTP_201_CREATED)
async def create_model_asset(
    payload: ModelAssetCreate,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    # trust_remote_code 并未出现在可写 schema 中，因此 API 层无法绕过全局禁用策略。
    asset = ModelAsset(**payload.model_dump())
    session.add(asset)
    await commit_or_conflict(session, "模型本地路径已被登记")
    await session.refresh(asset)
    return asset


@router.get("", response_model=list[ModelAssetRead])
async def list_model_assets(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[ModelAsset]:
    result = await session.scalars(
        select(ModelAsset).order_by(ModelAsset.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{asset_id}", response_model=ModelAssetRead)
async def get_model_asset(
    asset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    return await get_or_404(session, ModelAsset, asset_id, "模型资产")


@router.patch("/{asset_id}", response_model=ModelAssetRead)
async def update_model_asset(
    asset_id: uuid.UUID,
    payload: ModelAssetUpdate,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    asset = await get_or_404(session, ModelAsset, asset_id, "模型资产")
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
    asset = await get_or_404(session, ModelAsset, asset_id, "模型资产")
    await session.delete(asset)
    await commit_or_conflict(session, "模型正在被部署、训练或评测引用，不能删除")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
