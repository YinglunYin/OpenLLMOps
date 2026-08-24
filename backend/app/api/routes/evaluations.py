import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import Dataset, EvaluationRun, ModelAsset
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
    JobState,
    ModelKind,
)
from app.schemas import EvaluationRunCreate, EvaluationRunRead
from app.services.crud import commit_or_conflict, get_or_404

router = APIRouter(prefix="/evaluation-runs", tags=["模型评测"])
TERMINAL_STATES = {JobState.CANCELED, JobState.SUCCEEDED, JobState.FAILED}


@router.post("", response_model=EvaluationRunRead, status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(
    payload: EvaluationRunCreate,
    session: AsyncSession = Depends(get_db),
) -> EvaluationRun:
    invalid = [gpu_id for gpu_id in payload.gpu_ids if gpu_id >= get_settings().gpu_count]
    if invalid:
        raise HTTPException(status_code=422, detail=f"GPU 编号超出本机范围：{invalid}")
    base = await get_or_404(session, ModelAsset, payload.base_model_asset_id, "基线模型")
    candidate = await get_or_404(session, ModelAsset, payload.candidate_model_asset_id, "候选模型")
    if base.status != AssetStatus.READY or candidate.status != AssetStatus.READY:
        raise HTTPException(status_code=422, detail="基线模型和候选模型必须处于 ready 状态")
    if ModelKind.EMBEDDING in {base.model_kind, candidate.model_kind}:
        raise HTTPException(status_code=422, detail="Embedding 模型不支持生成式评测")
    if payload.custom_dataset_id:
        dataset = await get_or_404(session, Dataset, payload.custom_dataset_id, "评测数据集")
        if dataset.dataset_type != DatasetType.EVALUATION or dataset.status != DatasetStatus.READY:
            raise HTTPException(status_code=422, detail="自定义数据集必须是 ready 的 evaluation 数据集")

    run = EvaluationRun(
        **payload.model_dump(),
        actual_state=JobState.QUEUED,
        queued_at=datetime.now(UTC),
    )
    session.add(run)
    await commit_or_conflict(session, "评测任务名称已存在")
    await session.refresh(run)
    return run


@router.get("", response_model=list[EvaluationRunRead])
async def list_evaluation_runs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[EvaluationRun]:
    result = await session.scalars(
        select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{run_id}", response_model=EvaluationRunRead)
async def get_evaluation_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> EvaluationRun:
    return await get_or_404(session, EvaluationRun, run_id, "评测任务")


@router.post("/{run_id}/cancel", response_model=EvaluationRunRead)
async def cancel_evaluation_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> EvaluationRun:
    run = await get_or_404(session, EvaluationRun, run_id, "评测任务")
    run.desired_state = DesiredJobState.TERMINATED
    if run.actual_state in {JobState.CREATED, JobState.QUEUED}:
        run.actual_state = JobState.CANCELED
        run.queued_at = None
    elif run.actual_state not in TERMINAL_STATES | {JobState.CANCELING}:
        run.actual_state = JobState.CANCELING
    run.state_version += 1
    await session.commit()
    await session.refresh(run)
    return run


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Response:
    run = await get_or_404(session, EvaluationRun, run_id, "评测任务")
    if run.actual_state not in TERMINAL_STATES:
        raise HTTPException(status_code=409, detail="只能删除已结束的评测任务")
    await session.delete(run)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
