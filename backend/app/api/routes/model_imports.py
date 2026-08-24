import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from openllmops_model_importer import scan_inbox
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import ModelImportJob
from app.models.enums import ModelImportSource, ModelImportStatus
from app.schemas import InboxCandidateRead, ModelImportCreate, ModelImportRead
from app.services.crud import get_or_404
from app.services.model_import_coordinator import request_model_import_cancel

router = APIRouter(prefix="/model-imports", tags=["模型导入"])
inbox_router = APIRouter(prefix="/model-inbox", tags=["模型导入"])


async def _validate_controlled_candidate(directory_name: str) -> None:
    root = get_settings().model_inbox_root
    try:
        candidates = await asyncio.to_thread(scan_inbox, root)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="模型 inbox 目录不可用") from exc
    candidate = next((item for item in candidates if item.name == directory_name), None)
    if candidate is None:
        raise HTTPException(status_code=422, detail="受控目录候选不存在")
    if not candidate.ready_for_import:
        raise HTTPException(status_code=422, detail=candidate.reason or "受控目录尚不可导入")


@router.post(
    "",
    response_model=ModelImportRead,
    status_code=status.HTTP_201_CREATED,
    name="model_import.create",
)
async def create_model_import(
    payload: ModelImportCreate,
    session: AsyncSession = Depends(get_db),
) -> ModelImportJob:
    if payload.source == ModelImportSource.CONTROLLED_DIRECTORY:
        # worker 会在执行时再次做 resolve/软链接检查；API 预检只负责尽早给出可读错误。
        await _validate_controlled_candidate(payload.source_directory or "")
    job = ModelImportJob(
        **payload.model_dump(),
        status=ModelImportStatus.PENDING,
        progress_completed=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


@router.get("", response_model=list[ModelImportRead], name="model_import.list")
async def list_model_imports(
    import_status: ModelImportStatus | None = Query(default=None, alias="status"),
    source: ModelImportSource | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[ModelImportJob]:
    statement = select(ModelImportJob)
    if import_status is not None:
        statement = statement.where(ModelImportJob.status == import_status)
    if source is not None:
        statement = statement.where(ModelImportJob.source == source)
    result = await session.scalars(
        statement.order_by(ModelImportJob.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{job_id}", response_model=ModelImportRead, name="model_import.detail")
async def get_model_import(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ModelImportJob:
    return await get_or_404(session, ModelImportJob, job_id, "模型导入任务")


@router.post("/{job_id}/cancel", response_model=ModelImportRead, name="model_import.cancel")
async def cancel_model_import(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ModelImportJob:
    job = await request_model_import_cancel(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="模型导入任务不存在")
    return job


@inbox_router.get("", response_model=list[InboxCandidateRead], name="model_inbox.list")
async def list_model_inbox(
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[InboxCandidateRead]:
    try:
        candidates = await asyncio.to_thread(
            scan_inbox,
            get_settings().model_inbox_root,
            maximum_candidates=limit,
        )
    except OSError as exc:
        raise HTTPException(status_code=503, detail="模型 inbox 目录不可用") from exc
    return [InboxCandidateRead.model_validate(item) for item in candidates]
