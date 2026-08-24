import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import issue_api_key
from app.models import APIKey
from app.schemas import APIKeyCreate, APIKeyCreated, APIKeyRead
from app.services.crud import commit_or_conflict, get_or_404

router = APIRouter(prefix="/api-keys", tags=["API Key"])


@router.post("", response_model=APIKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: APIKeyCreate,
    session: AsyncSession = Depends(get_db),
) -> APIKeyCreated:
    raw_key, prefix, key_hash = issue_api_key()
    api_key = APIKey(name=payload.name, prefix=prefix, key_hash=key_hash)
    session.add(api_key)
    await commit_or_conflict(session, "API Key 名称已存在")
    await session.refresh(api_key)
    return APIKeyCreated(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        key=raw_key,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at,
        created_at=api_key.created_at,
        updated_at=api_key.updated_at,
    )


@router.get("", response_model=list[APIKeyRead])
async def list_api_keys(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[APIKey]:
    result = await session.scalars(
        select(APIKey).order_by(APIKey.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.post("/{key_id}/revoke", response_model=APIKeyRead)
async def revoke_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> APIKey:
    api_key = await get_or_404(session, APIKey, key_id, "API Key")
    if not api_key.is_active:
        return api_key
    api_key.is_active = False
    await session.commit()
    await session.refresh(api_key)
    return api_key


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Response:
    api_key = await get_or_404(session, APIKey, key_id, "API Key")
    if api_key.is_active:
        raise HTTPException(status_code=409, detail="请先撤销 API Key 再删除")
    await session.delete(api_key)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
