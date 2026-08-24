import uuid
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from app.core.config import get_settings
from app.core.database import get_db
from app.models import Dataset, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
    JobState,
    ModelKind,
    ModelSourceType,
    TrainingStage,
)
from app.schemas import (
    ModelAssetRead,
    StateActionResponse,
    TrainingArtifactKind,
    TrainingArtifactManifestRead,
    TrainingJobCreate,
    TrainingJobRead,
)
from app.services.crud import commit_or_conflict, get_or_404
from app.services.training_control import (
    TrainingControlError,
    build_training_archive,
    build_training_execution,
    derive_training_output_dir,
    list_training_artifacts,
    publish_training_model_files,
)

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
    asset = await get_or_404(session, ModelAsset, payload.model_asset_id, "模型资产")
    dataset = await get_or_404(session, Dataset, payload.dataset_id, "数据集")
    if asset.status != AssetStatus.READY or dataset.status != DatasetStatus.READY:
        raise HTTPException(status_code=422, detail="模型资产和数据集必须处于 ready 状态")
    if asset.model_kind == ModelKind.EMBEDDING:
        raise HTTPException(status_code=422, detail="Embedding 模型不支持生成式训练")
    expected_dataset_type = DatasetType.CPT if payload.stage == TrainingStage.CPT else DatasetType.SFT
    if dataset.dataset_type != expected_dataset_type:
        raise HTTPException(status_code=422, detail=f"训练阶段需要 {expected_dataset_type.value} 数据集")

    job_id = uuid.uuid4()
    try:
        job = TrainingJob(
            id=job_id,
            name=payload.name,
            model_asset_id=payload.model_asset_id,
            dataset_id=payload.dataset_id,
            stage=payload.stage,
            algorithm=payload.algorithm,
            gpu_ids=payload.gpu_ids,
            training_config=payload.training_config.model_dump(mode="json", exclude_none=True),
            output_dir=str(derive_training_output_dir(settings, job_id)),
            actual_state=JobState.QUEUED,
            queued_at=datetime.now(UTC),
        )
        # 创建时先返回可读的路径/配置错误；调度前会再校验一次，防止文件随后变化。
        await anyio.to_thread.run_sync(build_training_execution, job, asset, dataset, settings)
    except TrainingControlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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


@router.get("/{job_id}/artifacts", response_model=TrainingArtifactManifestRead)
async def get_training_artifacts(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> TrainingArtifactManifestRead:
    job = await get_or_404(session, TrainingJob, job_id, "训练任务")
    try:
        artifacts = await anyio.to_thread.run_sync(list_training_artifacts, job, get_settings())
    except TrainingControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return TrainingArtifactManifestRead(job_id=job.id, state=job.actual_state, artifacts=artifacts)


@router.get("/{job_id}/artifacts/{kind}/download", response_class=FileResponse)
async def download_training_artifact(
    job_id: uuid.UUID,
    kind: TrainingArtifactKind,
    session: AsyncSession = Depends(get_db),
) -> FileResponse:
    job = await get_or_404(session, TrainingJob, job_id, "训练任务")
    try:
        archive_path, artifact = await anyio.to_thread.run_sync(
            build_training_archive,
            job,
            kind,
            get_settings(),
        )
    except TrainingControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(
        archive_path,
        media_type="application/gzip",
        filename=artifact.archive_filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.post("/{job_id}/publish-model", response_model=ModelAssetRead)
async def publish_training_model(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ModelAsset:
    job = await session.get(TrainingJob, job_id, with_for_update=True)
    if job is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    if job.actual_state != JobState.SUCCEEDED:
        raise HTTPException(status_code=409, detail="只有 succeeded 训练任务可以发布模型")
    if job.published_model_asset_id is not None:
        published = await session.get(ModelAsset, job.published_model_asset_id)
        if published is None:  # pragma: no cover - 外键 SET NULL 是最终保障。
            raise HTTPException(status_code=409, detail="训练任务关联的已发布模型不存在")
        return published

    settings = get_settings()
    try:
        published_files = await anyio.to_thread.run_sync(
            publish_training_model_files,
            job,
            settings,
        )
    except TrainingControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    existing = await session.scalar(
        select(ModelAsset).where(ModelAsset.local_path == str(published_files.path))
    )
    if existing is not None:
        if existing.metadata_json.get("training_job_id") != str(job.id):
            raise HTTPException(status_code=409, detail="发布目录已被其他模型资产占用")
        job.published_model_asset_id = existing.id
        await session.commit()
        await session.refresh(existing)
        return existing

    base_asset = await session.get(ModelAsset, job.model_asset_id)
    if base_asset is None:  # pragma: no cover - 外键 RESTRICT 保证。
        raise HTTPException(status_code=409, detail="训练引用的基础模型不存在")
    published = ModelAsset(
        name=f"trained-{job.name[:80]}-{job.id.hex[:12]}",
        source_type=ModelSourceType.TRAINED,
        source_uri=f"training://{job.id}",
        revision=str(job.id),
        local_path=str(published_files.path),
        model_kind=base_asset.model_kind if job.stage == TrainingStage.CPT else ModelKind.INSTRUCT,
        format="safetensors",
        status=AssetStatus.READY,
        family=base_asset.family,
        parameter_count=base_asset.parameter_count,
        size_bytes=published_files.size_bytes,
        checksum=f"sha256:{published_files.checksum}",
        metadata_json={
            "training_job_id": str(job.id),
            "base_model_asset_id": str(job.model_asset_id),
            "stage": job.stage.value,
            "algorithm": job.algorithm.value,
            "artifact_kind": published_files.artifact_kind,
        },
    )
    session.add(published)
    try:
        await session.flush()
        job.published_model_asset_id = published.id
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="训练模型发布发生并发冲突，请重试") from exc
    await session.refresh(published)
    return published


@router.post("/{job_id}/terminate", response_model=StateActionResponse)
async def terminate_training_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> StateActionResponse:
    # 与 reconciler 的终态写入串行化，避免迟到的 terminate 用旧 RUNNING 快照
    # 覆盖已经提交的 SUCCEEDED/FAILED。
    job = await session.get(TrainingJob, job_id, with_for_update=True)
    if job is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    if job.actual_state in TERMINAL_STATES:
        return StateActionResponse(
            id=job.id,
            desired_state=job.desired_state.value,
            actual_state=job.actual_state.value,
            message="训练任务已结束，无需重复终止",
        )
    if job.desired_state == DesiredJobState.TERMINATED and job.actual_state == JobState.CANCELING:
        return StateActionResponse(
            id=job.id,
            desired_state=job.desired_state.value,
            actual_state=job.actual_state.value,
            message="训练终止指令已记录",
        )
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
