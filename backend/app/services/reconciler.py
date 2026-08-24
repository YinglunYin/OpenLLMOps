import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Deployment, EvaluationRun, GPULease, ModelAsset, TrainingJob
from app.models.entities import Dataset
from app.models.enums import (
    DeploymentState,
    DesiredJobState,
    DesiredServiceState,
    JobState,
    LeaseOwnerType,
)
from app.schemas.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
)
from app.services.gpu_scheduler import GPULeaseManager, LeaseOwner
from app.services.node_agent import NodeAgentError

logger = logging.getLogger(__name__)

ACTIVE_DEPLOYMENT_STATES = {
    DeploymentState.STARTING,
    DeploymentState.RUNNING,
    DeploymentState.STOPPING,
}
ACTIVE_JOB_STATES = {JobState.STARTING, JobState.RUNNING, JobState.CANCELING}
TERMINAL_DEPLOYMENT_STATES = {
    DeploymentState.CREATED,
    DeploymentState.STOPPED,
    DeploymentState.FAILED,
}
TERMINAL_JOB_STATES = {JobState.CANCELED, JobState.SUCCEEDED, JobState.FAILED}


class NodeAgentGateway(Protocol):
    async def execute(self, command: AgentCommand) -> AgentCommandResponse: ...


@dataclass(frozen=True, slots=True)
class WorkloadSnapshot:
    owner: LeaseOwner
    gpu_ids: tuple[int, ...]
    desired_state: DesiredServiceState | DesiredJobState
    actual_state: DeploymentState | JobState
    execution: dict[str, Any] = field(default_factory=dict)


class ScheduleOutcome(StrEnum):
    EMPTY = "empty"
    BLOCKED = "blocked"
    CLAIMED = "claimed"


@dataclass(frozen=True, slots=True)
class ScheduleAttempt:
    outcome: ScheduleOutcome
    workload: WorkloadSnapshot | None = None
    blocked_by: LeaseOwner | None = None


@dataclass(slots=True)
class ReconcileReport:
    inspected: int = 0
    scheduled: int = 0
    blocked: int = 0
    agent_errors: int = 0
    terminal_leases_reaped: int = 0


class StateReconciler:
    """将数据库期望状态收敛到 node-agent 实际状态。

    调度队列跨推理、训练和评测统一按 queued_at 严格 FIFO。队首请求的任一指定 GPU
    被占用时，本轮不会跳过它进行回填；已有推理租约也绝不会因训练排队而被抢占。
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        agent: NodeAgentGateway,
        lease_manager: GPULeaseManager,
        *,
        interval_seconds: float = 2.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._agent = agent
        self._leases = lease_manager
        self._interval_seconds = interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run_once(self) -> ReconcileReport:
        report = ReconcileReport()
        active = await self._load_active_workloads()
        for workload in active:
            report.inspected += 1
            action = self._action_for_active(workload)
            try:
                response = await self._agent.execute(self._command(workload, action))
            except NodeAgentError as exc:
                # agent 暂时失联时保留租约；即便 TTL 过期也先隔离 GPU，避免双重占用。
                report.agent_errors += 1
                await self._record_agent_error(workload, str(exc))
                continue
            await self._apply_observation(workload, response)

        report.terminal_leases_reaped = await self._reap_terminal_expired_leases()
        while True:
            attempt = await self._claim_fifo_head()
            if attempt.outcome == ScheduleOutcome.EMPTY:
                break
            if attempt.outcome == ScheduleOutcome.BLOCKED:
                report.blocked += 1
                break
            workload = attempt.workload
            if workload is None:  # pragma: no cover - 防御性检查，枚举已保证此分支不可达。
                break
            report.scheduled += 1
            try:
                response = await self._agent.execute(self._command(workload, AgentAction.START))
            except NodeAgentError as exc:
                report.agent_errors += 1
                await self._fail_start(workload, str(exc))
                continue
            await self._apply_observation(workload, response)
        return report

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """以固定间隔运行；单轮异常会记录但不会杀死后台协调任务。"""

        while not stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                # 后台守护循环必须在未知单轮错误后继续运行，具体堆栈进入服务日志。
                logger.exception("状态协调器单轮执行失败")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval_seconds)

    @staticmethod
    def _action_for_active(workload: WorkloadSnapshot) -> AgentAction:
        if workload.owner.type == LeaseOwnerType.DEPLOYMENT:
            return (
                AgentAction.STOP
                if workload.desired_state == DesiredServiceState.STOPPED
                else AgentAction.STATUS
            )
        return (
            AgentAction.STOP if workload.desired_state == DesiredJobState.TERMINATED else AgentAction.STATUS
        )

    @staticmethod
    def _command(workload: WorkloadSnapshot, action: AgentAction) -> AgentCommand:
        return AgentCommand(
            action=action,
            owner=AgentOwner(
                type=workload.owner.type,
                id=workload.owner.id,
                name=workload.owner.name,
                generation=workload.owner.generation,
            ),
            resources=AgentResourceRequest(gpu_ids=list(workload.gpu_ids)),
            execution=workload.execution if action == AgentAction.START else {},
        )

    async def _load_active_workloads(self) -> list[WorkloadSnapshot]:
        async with self._session_factory() as session:
            deployments = list(
                await session.scalars(
                    select(Deployment).where(Deployment.actual_state.in_(ACTIVE_DEPLOYMENT_STATES))
                )
            )
            training_jobs = list(
                await session.scalars(
                    select(TrainingJob).where(TrainingJob.actual_state.in_(ACTIVE_JOB_STATES))
                )
            )
            evaluation_runs = list(
                await session.scalars(
                    select(EvaluationRun).where(EvaluationRun.actual_state.in_(ACTIVE_JOB_STATES))
                )
            )
            return (
                [self._snapshot(row, LeaseOwnerType.DEPLOYMENT) for row in deployments]
                + [self._snapshot(row, LeaseOwnerType.TRAINING) for row in training_jobs]
                + [self._snapshot(row, LeaseOwnerType.EVALUATION) for row in evaluation_runs]
            )

    @staticmethod
    def _snapshot(
        row: Deployment | TrainingJob | EvaluationRun,
        owner_type: LeaseOwnerType,
        *,
        execution: dict[str, Any] | None = None,
    ) -> WorkloadSnapshot:
        runtime_generation = row.runtime_generation or row.state_version
        return WorkloadSnapshot(
            owner=LeaseOwner(
                type=owner_type,
                id=row.id,
                name=row.name,
                generation=runtime_generation,
            ),
            gpu_ids=tuple(sorted(row.gpu_ids)),
            desired_state=row.desired_state,
            actual_state=row.actual_state,
            execution=execution or {},
        )

    @staticmethod
    def _queue_sort_key(
        owner_type: LeaseOwnerType,
        row: Deployment | TrainingJob | EvaluationRun,
    ) -> tuple[float, str, str]:
        queued_at = row.queued_at or row.created_at
        if queued_at.tzinfo is None:
            queued_at = queued_at.replace(tzinfo=UTC)
        # 类型只作为同一微秒下的稳定 tie-breaker，不构成隐式优先级。
        return (queued_at.timestamp(), owner_type.value, str(row.id))

    async def _claim_fifo_head(self) -> ScheduleAttempt:
        async with (
            self._session_factory() as session,
            session.begin(),
            self._leases.scheduler_lock(session),
        ):
            deployments = list(
                await session.scalars(
                    select(Deployment)
                    .where(
                        Deployment.actual_state == DeploymentState.QUEUED,
                        Deployment.desired_state == DesiredServiceState.RUNNING,
                    )
                    .with_for_update()
                )
            )
            training_jobs = list(
                await session.scalars(
                    select(TrainingJob)
                    .where(
                        TrainingJob.actual_state == JobState.QUEUED,
                        TrainingJob.desired_state == DesiredJobState.RUNNING,
                    )
                    .with_for_update()
                )
            )
            evaluation_runs = list(
                await session.scalars(
                    select(EvaluationRun)
                    .where(
                        EvaluationRun.actual_state == JobState.QUEUED,
                        EvaluationRun.desired_state == DesiredJobState.RUNNING,
                    )
                    .with_for_update()
                )
            )
            candidates: list[tuple[LeaseOwnerType, Deployment | TrainingJob | EvaluationRun]] = (
                [(LeaseOwnerType.DEPLOYMENT, row) for row in deployments]
                + [(LeaseOwnerType.TRAINING, row) for row in training_jobs]
                + [(LeaseOwnerType.EVALUATION, row) for row in evaluation_runs]
            )
            if not candidates:
                return ScheduleAttempt(ScheduleOutcome.EMPTY)
            owner_type, row = min(
                candidates,
                key=lambda item: self._queue_sort_key(item[0], item[1]),
            )
            # 新一轮实际运行沿用当前期望版本；后续 stop 只改变 state_version，
            # runtime_generation 保持不变，确保迟到的 start 响应无法影响下一代实例。
            row.runtime_generation = row.state_version
            owner = LeaseOwner(
                type=owner_type,
                id=row.id,
                name=row.name,
                generation=row.runtime_generation,
            )
            acquisition = await self._leases.try_acquire_locked(
                session,
                owner,
                row.gpu_ids,
                now=self._clock(),
            )
            if not acquisition.acquired:
                row.runtime_generation = 0
                return ScheduleAttempt(
                    ScheduleOutcome.BLOCKED,
                    blocked_by=acquisition.blocking_owner,
                )
            row.actual_state = (
                DeploymentState.STARTING if owner_type == LeaseOwnerType.DEPLOYMENT else JobState.STARTING
            )
            row.queued_at = None
            row.error_message = None
            execution = await self._build_execution(session, owner_type, row)
            return ScheduleAttempt(
                ScheduleOutcome.CLAIMED,
                workload=self._snapshot(row, owner_type, execution=execution),
            )

    @staticmethod
    async def _build_execution(
        session: AsyncSession,
        owner_type: LeaseOwnerType,
        row: Deployment | TrainingJob | EvaluationRun,
    ) -> dict[str, Any]:
        if owner_type == LeaseOwnerType.DEPLOYMENT:
            assert isinstance(row, Deployment)
            asset = await session.get(ModelAsset, row.model_asset_id)
            if asset is None:
                raise RuntimeError("部署引用的模型资产不存在")
            return {
                "runner": "vllm",
                "service_type": row.task_type.value,
                "model_path": asset.local_path,
                "served_model_name": row.served_model_name,
                "port": row.port,
                "tensor_parallel_size": row.tensor_parallel_size,
                "simplified_config": row.simplified_config,
                "vllm_args": row.vllm_args,
            }
        if owner_type == LeaseOwnerType.TRAINING:
            assert isinstance(row, TrainingJob)
            asset = await session.get(ModelAsset, row.model_asset_id)
            dataset = await session.get(Dataset, row.dataset_id)
            if asset is None or dataset is None:
                raise RuntimeError("训练任务引用的模型资产或数据集不存在")
            return {
                "runner": "llamafactory",
                "model_path": asset.local_path,
                "dataset_path": dataset.local_path,
                "stage": row.stage.value,
                "algorithm": row.algorithm.value,
                "training_config": row.training_config,
                "output_dir": row.output_dir,
            }
        assert isinstance(row, EvaluationRun)
        base = await session.get(ModelAsset, row.base_model_asset_id)
        candidate = await session.get(ModelAsset, row.candidate_model_asset_id)
        dataset = await session.get(Dataset, row.custom_dataset_id) if row.custom_dataset_id else None
        if base is None or candidate is None:
            raise RuntimeError("评测任务引用的模型资产不存在")
        return {
            "runner": "evaluation",
            "base_model_path": base.local_path,
            "candidate_model_path": candidate.local_path,
            "custom_dataset_path": dataset.local_path if dataset else None,
            "builtin_datasets": row.builtin_datasets,
        }

    async def _apply_observation(
        self,
        workload: WorkloadSnapshot,
        response: AgentCommandResponse,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            row = await self._get_row(session, workload.owner, for_update=True)
            if row is None or row.runtime_generation != workload.owner.generation:
                return
            observed = response.observed_state
            if workload.owner.type == LeaseOwnerType.DEPLOYMENT:
                assert isinstance(row, Deployment)
                await self._apply_deployment_observation(session, row, workload.owner, response)
            else:
                assert isinstance(row, (TrainingJob, EvaluationRun))
                await self._apply_job_observation(session, row, workload.owner, response)
            if observed in {
                AgentWorkloadState.STARTING,
                AgentWorkloadState.RUNNING,
                AgentWorkloadState.STOPPING,
            }:
                await self._leases.heartbeat(session, workload.owner, now=self._clock())

    async def _apply_deployment_observation(
        self,
        session: AsyncSession,
        row: Deployment,
        owner: LeaseOwner,
        response: AgentCommandResponse,
    ) -> None:
        observed = response.observed_state
        if not response.accepted and observed == AgentWorkloadState.ABSENT:
            row.actual_state = DeploymentState.FAILED
            row.error_message = response.message or "node-agent 拒绝部署启动"
            await self._leases.release(session, owner)
            return
        if observed == AgentWorkloadState.ABSENT:
            await self._leases.release(session, owner)
            if row.desired_state == DesiredServiceState.RUNNING:
                # 推理服务允许自动恢复，但仍重新进入 FIFO，不越过更早任务。
                row.actual_state = DeploymentState.QUEUED
                row.queued_at = self._clock()
                row.runtime_generation = 0
                row.error_message = "node-agent 未发现运行实例，已重新排队"
            else:
                row.actual_state = DeploymentState.STOPPED
                row.internal_url = None
                row.error_message = None
            return
        if observed in {AgentWorkloadState.FAILED, AgentWorkloadState.SUCCEEDED}:
            row.actual_state = DeploymentState.FAILED
            row.error_message = response.message or "推理实例异常退出"
            row.internal_url = None
            await self._leases.release(session, owner)
            return
        if row.desired_state == DesiredServiceState.STOPPED:
            row.actual_state = DeploymentState.STOPPING
            return
        row.actual_state = {
            AgentWorkloadState.STARTING: DeploymentState.STARTING,
            AgentWorkloadState.RUNNING: DeploymentState.RUNNING,
            AgentWorkloadState.STOPPING: DeploymentState.STOPPING,
        }[observed]
        endpoint = response.metadata.get("endpoint")
        if isinstance(endpoint, str):
            row.internal_url = endpoint
        port = response.metadata.get("port")
        if isinstance(port, int):
            row.port = port
        row.error_message = None if response.accepted else response.message

    async def _apply_job_observation(
        self,
        session: AsyncSession,
        row: TrainingJob | EvaluationRun,
        owner: LeaseOwner,
        response: AgentCommandResponse,
    ) -> None:
        observed = response.observed_state
        if observed == AgentWorkloadState.ABSENT:
            await self._leases.release(session, owner)
            if row.desired_state == DesiredJobState.TERMINATED:
                row.actual_state = JobState.CANCELED
                message = None
            else:
                # 训练和评测不自动恢复，避免重复写 checkpoint 或产生不可比较的结果。
                row.actual_state = JobState.FAILED
                message = response.message or "运行实例意外消失，请人工重新创建任务"
            row.error_message = message
            if isinstance(row, TrainingJob):
                row.finished_at = self._clock()
            return
        if observed in {AgentWorkloadState.SUCCEEDED, AgentWorkloadState.FAILED}:
            row.actual_state = (
                JobState.SUCCEEDED if observed == AgentWorkloadState.SUCCEEDED else JobState.FAILED
            )
            row.error_message = None if observed == AgentWorkloadState.SUCCEEDED else response.message
            if isinstance(row, TrainingJob):
                row.finished_at = self._clock()
                self._copy_training_outputs(row, response.metadata)
            elif observed == AgentWorkloadState.SUCCEEDED:
                metrics = response.metadata.get("metrics")
                comparison = response.metadata.get("comparison")
                if isinstance(metrics, dict):
                    row.metrics = metrics
                if isinstance(comparison, dict):
                    row.comparison = comparison
            await self._leases.release(session, owner)
            return
        if row.desired_state == DesiredJobState.TERMINATED:
            row.actual_state = JobState.CANCELING
        else:
            row.actual_state = {
                AgentWorkloadState.STARTING: JobState.STARTING,
                AgentWorkloadState.RUNNING: JobState.RUNNING,
                AgentWorkloadState.STOPPING: JobState.CANCELING,
            }[observed]
        if isinstance(row, TrainingJob):
            if observed == AgentWorkloadState.RUNNING and row.started_at is None:
                row.started_at = self._clock()
            self._copy_training_progress(row, response.metadata)
        row.error_message = None if response.accepted else response.message

    @staticmethod
    def _copy_training_progress(row: TrainingJob, metadata: dict[str, Any]) -> None:
        progress = metadata.get("progress")
        if isinstance(progress, int | float):
            row.progress = min(100.0, max(0.0, float(progress)))
        current_step = metadata.get("current_step")
        total_steps = metadata.get("total_steps")
        metrics = metadata.get("metrics")
        if isinstance(current_step, int):
            row.current_step = current_step
        if isinstance(total_steps, int):
            row.total_steps = total_steps
        if isinstance(metrics, dict):
            row.metrics = metrics

    @staticmethod
    def _copy_training_outputs(row: TrainingJob, metadata: dict[str, Any]) -> None:
        for key in ("checkpoint_path", "adapter_path", "merged_model_path"):
            value = metadata.get(key)
            if isinstance(value, str):
                setattr(row, key, value)

    async def _record_agent_error(self, workload: WorkloadSnapshot, message: str) -> None:
        async with self._session_factory() as session, session.begin():
            row = await self._get_row(session, workload.owner, for_update=True)
            if row is not None and row.runtime_generation == workload.owner.generation:
                row.error_message = message

    async def _fail_start(self, workload: WorkloadSnapshot, message: str) -> None:
        async with self._session_factory() as session, session.begin():
            row = await self._get_row(session, workload.owner, for_update=True)
            if row is None or row.runtime_generation != workload.owner.generation:
                return
            row.actual_state = (
                DeploymentState.FAILED
                if workload.owner.type == LeaseOwnerType.DEPLOYMENT
                else JobState.FAILED
            )
            row.error_message = message
            if isinstance(row, TrainingJob):
                row.finished_at = self._clock()
            await self._leases.release(session, workload.owner)

    async def _reap_terminal_expired_leases(self) -> int:
        now = self._clock()
        async with (
            self._session_factory() as session,
            session.begin(),
            self._leases.scheduler_lock(session),
        ):
            expired = list(
                await session.scalars(select(GPULease).where(GPULease.expires_at <= now).with_for_update())
            )
            candidate_keys = {(lease.owner_type, lease.owner_id, lease.generation) for lease in expired}
            confirmed: set[tuple[LeaseOwnerType, Any, int]] = set()
            for owner_type, owner_id, generation in candidate_keys:
                owner = LeaseOwner(owner_type, owner_id, "expired", generation)
                row = await self._get_row(session, owner, for_update=True)
                if row is None or self._is_terminal(row):
                    confirmed.add((owner_type, owner_id, generation))
            reaped = await self._leases.reap_expired_locked(
                session,
                confirmed,
                now=now,
            )
            return len(reaped)

    @staticmethod
    def _is_terminal(row: Deployment | TrainingJob | EvaluationRun) -> bool:
        if isinstance(row, Deployment):
            return row.actual_state in TERMINAL_DEPLOYMENT_STATES
        return row.actual_state in TERMINAL_JOB_STATES

    @staticmethod
    async def _get_row(
        session: AsyncSession,
        owner: LeaseOwner,
        *,
        for_update: bool,
    ) -> Deployment | TrainingJob | EvaluationRun | None:
        model = {
            LeaseOwnerType.DEPLOYMENT: Deployment,
            LeaseOwnerType.TRAINING: TrainingJob,
            LeaseOwnerType.EVALUATION: EvaluationRun,
        }[owner.type]
        return await session.get(model, owner.id, with_for_update=for_update)
