import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import Dataset, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
    JobState,
    TrainingStage,
)
from app.schemas import StateActionResponse, TrainingJobCreate, TrainingJobRead
from app.services.crud import commit_or_conflict, get_or_404
from app.services.dataset_files import ensure_path_within

router = APIRouter(prefix="/training-jobs", tags=["模型训练"])
TERMINAL_STATES = {JobState.CANCELED, JobState.SUCCEEDED, JobState.FAILED}


@router.post("", response_model=TrainingJobRead, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    payload: TrainingJobCreate,
    session: AsyncSession = Depends(get_db),
) -> TrainingJob:
    settings = get_settings()
    invalid_gpu_ids = [gpu_id for gpu_id in payload.gpu_ids if gpu_id >= settings.gpu_count]
    if invalid_gpu_ids:
        raise HTTPException(status_code=422, detail=f"GPU 编号超出本机范围：{invalid_gpu_ids}")
    try:
        ensure_path_within(Path(payload.output_dir), settings.checkpoint_root)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    asset = await get_or_404(session, ModelAsset, payload.model_asset_id, "模型资产")
    dataset = await get_or_404(session, Dataset, payload.dataset_id, "数据集")
    if asset.status != AssetStatus.READY or dataset.status != DatasetStatus.READY:
        raise HTTPException(status_code=422, detail="模型资产和数据集必须处于 ready 状态")
    expected_dataset_type = DatasetType.CPT if payload.stage == TrainingStage.CPT else DatasetType.SFT
    if dataset.dataset_type != expected_dataset_type:
        raise HTTPException(status_code=422, detail=f"训练阶段需要 {expected_dataset_type.value} 数据集")

    job = TrainingJob(
        **payload.model_dump(),
        actual_state=JobState.QUEUED,
        queued_at=datetime.now(UTC),
    )
    session.add(job)
    await commit_or_conflict(session, "训练任务名称已存在")
    await session.refresh(job)
    return job


@router.get("", response_model=list[TrainingJobRead])
async def list_training_jobs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[TrainingJob]:
    result = await session.scalars(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{job_id}", response_model=TrainingJobRead)
async def get_training_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> TrainingJob:
    return await get_or_404(session, TrainingJob, job_id, "训练任务")


@router.post("/{job_id}/terminate", response_model=StateActionResponse)
async def terminate_training_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> StateActionResponse:
    job = await get_or_404(session, TrainingJob, job_id, "训练任务")
    job.desired_state = DesiredJobState.TERMINATED
    if job.actual_state in {JobState.CREATED, JobState.QUEUED}:
        job.actual_state = JobState.CANCELED
        job.queued_at = None
        job.finished_at = datetime.now(UTC)
    elif job.actual_state not in TERMINAL_STATES | {JobState.CANCELING}:
        job.actual_state = JobState.CANCELING
    job.state_version += 1
    await session.commit()
    return StateActionResponse(
        id=job.id,
        desired_state=job.desired_state.value,
        actual_state=job.actual_state.value,
        message="训练终止指令已记录",
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Response:
    job = await get_or_404(session, TrainingJob, job_id, "训练任务")
    if job.actual_state not in TERMINAL_STATES:
        raise HTTPException(status_code=409, detail="只能删除已结束的训练任务")
    await session.delete(job)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
