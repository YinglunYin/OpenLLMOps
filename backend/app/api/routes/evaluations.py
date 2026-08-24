import uuid
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_cleanup_node_agent
from app.core.config import get_settings
from app.core.database import get_db
from app.models import Dataset, EvaluationRun
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
    JobState,
    LeaseOwnerType,
    ModelKind,
)
from app.schemas import EvaluationRunCreate, EvaluationRunRead
from app.services.crud import commit_or_conflict, get_or_404
from app.services.evaluation_control import (
    EvaluationControlError,
    build_evaluation_execution,
    derive_evaluation_output_dir,
    template_for_model,
)
from app.services.model_assets import get_active_model_assets_for_update
from app.services.workload_cleanup import (
    CleanupAgentGateway,
    WorkloadCleanupBlocked,
    WorkloadCleanupUnavailable,
    confirm_absent_and_release_leases,
)

router = APIRouter(prefix="/evaluation-runs", tags=["模型评测"])
TERMINAL_STATES = {JobState.CANCELED, JobState.SUCCEEDED, JobState.FAILED}


@router.post("", response_model=EvaluationRunRead, status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(
    payload: EvaluationRunCreate,
    session: AsyncSession = Depends(get_db),
) -> EvaluationRun:
    settings = get_settings()
    invalid = [gpu_id for gpu_id in payload.gpu_ids if gpu_id >= settings.gpu_count]
    if invalid:
        raise HTTPException(status_code=422, detail=f"GPU 编号超出本机范围：{invalid}")
    assets = await get_active_model_assets_for_update(
        session,
        {
            payload.base_model_asset_id: "基线模型",
            payload.candidate_model_asset_id: "候选模型",
        },
    )
    base = assets[payload.base_model_asset_id]
    candidate = assets[payload.candidate_model_asset_id]
    if base.status != AssetStatus.READY or candidate.status != AssetStatus.READY:
        raise HTTPException(status_code=422, detail="基线模型和候选模型必须处于 ready 状态")
    if ModelKind.EMBEDDING in {base.model_kind, candidate.model_kind}:
        raise HTTPException(status_code=422, detail="Embedding 模型不支持生成式评测")
    dataset: Dataset | None = None
    if payload.custom_dataset_id:
        dataset = await get_or_404(session, Dataset, payload.custom_dataset_id, "评测数据集")
        if dataset.dataset_type != DatasetType.EVALUATION or dataset.status != DatasetStatus.READY:
            raise HTTPException(status_code=422, detail="自定义数据集必须是 ready 的 evaluation 数据集")

    run_id = uuid.uuid4()
    try:
        run = EvaluationRun(
            id=run_id,
            name=payload.name,
            base_model_asset_id=payload.base_model_asset_id,
            candidate_model_asset_id=payload.candidate_model_asset_id,
            custom_dataset_id=payload.custom_dataset_id,
            builtin_datasets=list(payload.builtin_datasets),
            base_template=template_for_model(base.model_kind),
            candidate_template=template_for_model(candidate.model_kind),
            output_dir=str(derive_evaluation_output_dir(settings, run_id)),
            tensor_parallel_size=len(payload.gpu_ids),
            gpu_memory_utilization=settings.evaluation_gpu_memory_utilization,
            concurrency=settings.evaluation_concurrency,
            max_tokens=settings.evaluation_max_tokens,
            actual_state=JobState.QUEUED,
            gpu_ids=payload.gpu_ids,
            queued_at=datetime.now(UTC),
        )
        # 创建时先给出同步、可读的配置错误；调度前仍会再次检查，防止文件随后被移动。
        await anyio.to_thread.run_sync(
            build_evaluation_execution,
            run,
            base,
            candidate,
            dataset,
            settings,
        )
    except EvaluationControlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        run.finished_at = datetime.now(UTC)
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
    agent: CleanupAgentGateway | None = Depends(get_cleanup_node_agent),
) -> Response:
    run = await session.get(EvaluationRun, run_id, with_for_update=True)
    if run is None:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    if run.actual_state not in TERMINAL_STATES:
        raise HTTPException(status_code=409, detail="只能删除已结束的评测任务")
    if agent is None:
        raise HTTPException(status_code=503, detail="未配置 node-agent HMAC 密钥，无法确认容器 absent")
    try:
        await confirm_absent_and_release_leases(
            session,
            agent,
            owner_type=LeaseOwnerType.EVALUATION,
            owner_id=run.id,
            owner_name=run.name,
            generation=run.runtime_generation or run.state_version,
            gpu_ids=run.gpu_ids,
        )
    except WorkloadCleanupUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WorkloadCleanupBlocked as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.delete(run)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
