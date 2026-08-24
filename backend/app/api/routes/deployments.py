import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_cleanup_node_agent
from app.core.config import get_settings
from app.core.database import get_db
from app.models import Deployment
from app.models.enums import (
    AssetStatus,
    DeploymentState,
    DeploymentTaskType,
    DesiredServiceState,
    LeaseOwnerType,
    ModelKind,
)
from app.schemas import (
    DeploymentCreate,
    DeploymentRead,
    DeploymentUpdate,
    StateActionResponse,
)
from app.services.crud import commit_or_conflict, get_or_404
from app.services.model_assets import get_active_model_asset
from app.services.workload_cleanup import (
    CleanupAgentGateway,
    WorkloadCleanupBlocked,
    WorkloadCleanupUnavailable,
    confirm_absent_and_release_leases,
)

router = APIRouter(prefix="/deployments", tags=["模型部署"])
EDITABLE_STATES = {DeploymentState.CREATED, DeploymentState.STOPPED, DeploymentState.FAILED}


def _validate_gpu_ids(gpu_ids: list[int]) -> None:
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= get_settings().gpu_count]
    if invalid:
        raise HTTPException(status_code=422, detail=f"GPU 编号超出本机范围：{invalid}")


@router.post("", response_model=DeploymentRead, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    payload: DeploymentCreate,
    session: AsyncSession = Depends(get_db),
) -> Deployment:
    _validate_gpu_ids(payload.gpu_ids)
    asset = await get_active_model_asset(session, payload.model_asset_id, for_update=True)
    if asset.status != AssetStatus.READY:
        raise HTTPException(status_code=422, detail="模型资产尚未就绪")
    if payload.task_type == DeploymentTaskType.EMBEDDING and asset.model_kind != ModelKind.EMBEDDING:
        raise HTTPException(status_code=422, detail="Embedding 部署必须选择 embedding 模型")
    if payload.task_type == DeploymentTaskType.GENERATE and asset.model_kind == ModelKind.EMBEDDING:
        raise HTTPException(status_code=422, detail="生成部署不能选择 embedding 模型")

    deployment = Deployment(
        **payload.model_dump(),
        desired_state=DesiredServiceState.RUNNING,
        actual_state=DeploymentState.QUEUED,
        queued_at=datetime.now(UTC),
    )
    session.add(deployment)
    await commit_or_conflict(session, "部署名称或对外模型名已被占用")
    await session.refresh(deployment)
    return deployment


@router.get("", response_model=list[DeploymentRead])
async def list_deployments(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[Deployment]:
    result = await session.scalars(
        select(Deployment).order_by(Deployment.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{deployment_id}", response_model=DeploymentRead)
async def get_deployment(
    deployment_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Deployment:
    return await get_or_404(session, Deployment, deployment_id, "部署")


@router.patch("/{deployment_id}", response_model=DeploymentRead)
async def update_deployment(
    deployment_id: uuid.UUID,
    payload: DeploymentUpdate,
    session: AsyncSession = Depends(get_db),
) -> Deployment:
    deployment = await get_or_404(session, Deployment, deployment_id, "部署")
    if deployment.actual_state not in EDITABLE_STATES:
        raise HTTPException(status_code=409, detail="仅停止或失败的部署可以编辑")

    changes = payload.model_dump(exclude_unset=True)
    final_gpu_ids = changes.get("gpu_ids", deployment.gpu_ids)
    final_tp = changes.get("tensor_parallel_size", deployment.tensor_parallel_size)
    _validate_gpu_ids(final_gpu_ids)
    if len(final_gpu_ids) != final_tp:
        raise HTTPException(status_code=422, detail="tensor_parallel_size 必须等于所选 GPU 数量")
    for field, value in changes.items():
        setattr(deployment, field, value)
    deployment.state_version += 1
    await commit_or_conflict(session, "部署名称已被占用")
    await session.refresh(deployment)
    return deployment


@router.post("/{deployment_id}/start", response_model=StateActionResponse)
async def start_deployment(
    deployment_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> StateActionResponse:
    deployment = await get_or_404(session, Deployment, deployment_id, "部署")
    deployment.desired_state = DesiredServiceState.RUNNING
    if deployment.actual_state not in {
        DeploymentState.RUNNING,
        DeploymentState.STARTING,
        DeploymentState.QUEUED,
    }:
        # 请求只进入非抢占队列，后台协调器拿到全部整卡租约后才会切到 starting。
        deployment.actual_state = DeploymentState.QUEUED
        deployment.queued_at = datetime.now(UTC)
        # 新一轮启动不沿用上一代实例的端点、健康状态和运行起点。
        deployment.internal_url = None
        deployment.health_status = None
        deployment.started_at = None
        deployment.error_message = None
        deployment.state_version += 1
    await session.commit()
    return StateActionResponse(
        id=deployment.id,
        desired_state=deployment.desired_state.value,
        actual_state=deployment.actual_state.value,
        message="部署已进入启动队列",
    )


@router.post("/{deployment_id}/stop", response_model=StateActionResponse)
async def stop_deployment(
    deployment_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> StateActionResponse:
    deployment = await get_or_404(session, Deployment, deployment_id, "部署")
    deployment.desired_state = DesiredServiceState.STOPPED
    if deployment.actual_state in {
        DeploymentState.CREATED,
        DeploymentState.QUEUED,
        DeploymentState.FAILED,
    }:
        deployment.actual_state = DeploymentState.STOPPED
        deployment.queued_at = None
        deployment.internal_url = None
        deployment.health_status = None
    elif deployment.actual_state not in {DeploymentState.STOPPED, DeploymentState.STOPPING}:
        # 运行容器由协调器停止，确认退出前不能提前释放 GPU 租约。
        deployment.actual_state = DeploymentState.STOPPING
    deployment.state_version += 1
    await session.commit()
    return StateActionResponse(
        id=deployment.id,
        desired_state=deployment.desired_state.value,
        actual_state=deployment.actual_state.value,
        message="部署停止指令已记录",
    )


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    agent: CleanupAgentGateway | None = Depends(get_cleanup_node_agent),
) -> Response:
    deployment = await session.get(Deployment, deployment_id, with_for_update=True)
    if deployment is None:
        raise HTTPException(status_code=404, detail="部署不存在")
    if deployment.actual_state not in EDITABLE_STATES:
        raise HTTPException(status_code=409, detail="请先停止部署再删除")
    if agent is None:
        raise HTTPException(status_code=503, detail="未配置 node-agent HMAC 密钥，无法确认容器 absent")
    try:
        await confirm_absent_and_release_leases(
            session,
            agent,
            owner_type=LeaseOwnerType.DEPLOYMENT,
            owner_id=deployment.id,
            owner_name=deployment.name,
            generation=deployment.runtime_generation or deployment.state_version,
            gpu_ids=deployment.gpu_ids,
        )
    except WorkloadCleanupUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WorkloadCleanupBlocked as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.delete(deployment)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
