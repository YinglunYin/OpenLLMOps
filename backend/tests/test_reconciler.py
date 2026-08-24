import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.config import get_settings
from app.models import Dataset, Deployment, GPULease, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DeploymentState,
    DeploymentTaskType,
    DesiredServiceState,
    JobState,
    LeaseOwnerType,
    ModelKind,
    ModelSourceType,
    TrainingAlgorithm,
    TrainingStage,
)
from app.schemas.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentWorkloadState,
)
from app.services.gpu_scheduler import GPULeaseManager
from app.services.node_agent import NodeAgentError
from app.services.reconciler import StateReconciler


class FakeAgent:
    def __init__(self) -> None:
        self.commands: list[AgentCommand] = []
        self.states: dict[tuple[LeaseOwnerType, object, int], AgentWorkloadState] = {}

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        key = (command.owner.type, command.owner.id, command.owner.generation)
        if command.action == AgentAction.START:
            self.states[key] = AgentWorkloadState.RUNNING
        elif command.action == AgentAction.STOP:
            self.states[key] = AgentWorkloadState.ABSENT
        state = self.states.get(key, AgentWorkloadState.ABSENT)
        metadata = {}
        if command.owner.type == LeaseOwnerType.DEPLOYMENT and state == AgentWorkloadState.RUNNING:
            metadata = {
                "endpoint": "http://127.0.0.1:18000/v1",
                "port": 18000,
                "service_type": command.execution.get("service_type", "generate"),
            }
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=state,
            observed_at=datetime.now(UTC),
            metadata=metadata,
        )


class UncertainStartAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.start_attempted = False

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        key = (command.owner.type, command.owner.id, command.owner.generation)
        if command.action == AgentAction.START:
            self.start_attempted = True
            # 模拟节点已启动容器，但响应在返回控制面前丢失。
            self.states[key] = AgentWorkloadState.RUNNING
            raise NodeAgentError("node-agent 调用失败：响应超时")
        state = self.states.get(key, AgentWorkloadState.ABSENT)
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=state,
            observed_at=datetime.now(UTC),
            metadata={"endpoint": "http://127.0.0.1:18000/v1", "port": 18000}
            if command.owner.type == LeaseOwnerType.DEPLOYMENT
            else {},
        )


class RejectStartAgent(FakeAgent):
    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=False,
            observed_state=AgentWorkloadState.FAILED,
            observed_at=datetime.now(UTC),
            message="GPU 与节点已有工作负载冲突",
            error_code="gpu_conflict",
        )


async def _seed_deployment_and_training(factory):  # type: ignore[no-untyped-def]
    queued_at = datetime(2026, 8, 24, tzinfo=UTC)
    settings = get_settings()
    model_path = settings.model_root / f"reconciler-{uuid.uuid4()}"
    dataset_path = settings.dataset_root / f"reconciler-{uuid.uuid4()}.jsonl"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    (model_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
    dataset_body = '{"instruction":"Q","output":"A"}\n'
    dataset_path.write_text(dataset_body, encoding="utf-8")
    training_id = uuid.uuid4()
    async with factory() as session, session.begin():
        asset = ModelAsset(
            name="base",
            source_type=ModelSourceType.MANUAL,
            local_path=str(model_path),
            model_kind=ModelKind.BASE,
            status=AssetStatus.READY,
        )
        dataset = Dataset(
            name="sft",
            dataset_type=DatasetType.SFT,
            status=DatasetStatus.READY,
            file_name="sft.jsonl",
            local_path=str(dataset_path),
            record_count=1,
            size_bytes=dataset_path.stat().st_size,
            sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            schema_summary={"format": "jsonl", "record_format": "sft_alpaca"},
        )
        session.add_all([asset, dataset])
        await session.flush()
        deployment = Deployment(
            name="chat",
            served_model_name="chat-model",
            model_asset_id=asset.id,
            task_type=DeploymentTaskType.GENERATE,
            desired_state=DesiredServiceState.RUNNING,
            actual_state=DeploymentState.QUEUED,
            gpu_ids=[0],
            tensor_parallel_size=1,
            queued_at=queued_at,
        )
        training = TrainingJob(
            id=training_id,
            name="domain-sft",
            model_asset_id=asset.id,
            dataset_id=dataset.id,
            stage=TrainingStage.SFT,
            algorithm=TrainingAlgorithm.LORA,
            actual_state=JobState.QUEUED,
            gpu_ids=[0],
            training_config={"template": "qwen"},
            output_dir=str(settings.checkpoint_root / str(training_id)),
            queued_at=queued_at + timedelta(seconds=1),
        )
        session.add_all([deployment, training])
        await session.flush()
        return deployment.id, training.id


async def test_fifo_and_training_never_preempts_inference(isolated_session_factory) -> None:
    deployment_id, training_id = await _seed_deployment_and_training(isolated_session_factory)
    agent = FakeAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
    )

    first = await reconciler.run_once()
    assert first.scheduled == 1
    assert first.blocked == 1
    assert [command.action for command in agent.commands] == [AgentAction.START]

    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        training = await session.get(TrainingJob, training_id)
        leases = list(await session.scalars(select(GPULease)))
        assert deployment is not None and deployment.actual_state == DeploymentState.RUNNING
        assert deployment.internal_url == "http://127.0.0.1:18000/v1"
        assert training is not None and training.actual_state == JobState.QUEUED
        assert len(leases) == 1 and leases[0].owner_type == LeaseOwnerType.DEPLOYMENT

    # 只有管理员明确改变推理期望状态后，训练才可获得释放出的整卡。
    async with isolated_session_factory() as session, session.begin():
        deployment = await session.get(Deployment, deployment_id)
        assert deployment is not None
        deployment.desired_state = DesiredServiceState.STOPPED
        deployment.actual_state = DeploymentState.STOPPING
        deployment.state_version += 1

    command_offset = len(agent.commands)
    second = await reconciler.run_once()
    assert second.scheduled == 1
    assert [command.action for command in agent.commands[command_offset:]] == [
        AgentAction.STOP,
        AgentAction.START,
    ]

    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        training = await session.get(TrainingJob, training_id)
        leases = list(await session.scalars(select(GPULease)))
        assert deployment is not None and deployment.actual_state == DeploymentState.STOPPED
        assert training is not None and training.actual_state == JobState.RUNNING
        assert len(leases) == 1 and leases[0].owner_type == LeaseOwnerType.TRAINING

    # 同样的“实例消失”对训练采取保守失败，不自动重跑并重复写 checkpoint。
    agent.states.clear()
    command_offset = len(agent.commands)
    training_missing = await reconciler.run_once()
    assert training_missing.scheduled == 0
    assert [command.action for command in agent.commands[command_offset:]] == [AgentAction.STATUS]
    async with isolated_session_factory() as session:
        training = await session.get(TrainingJob, training_id)
        assert training is not None and training.actual_state == JobState.FAILED
        assert list(await session.scalars(select(GPULease))) == []


async def test_missing_inference_is_requeued_and_restored(isolated_session_factory) -> None:
    deployment_id, training_id = await _seed_deployment_and_training(isolated_session_factory)
    async with isolated_session_factory() as session, session.begin():
        training = await session.get(TrainingJob, training_id)
        assert training is not None
        training.actual_state = JobState.CANCELED

    agent = FakeAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
    )
    await reconciler.run_once()

    # node-agent 丢失实例后，推理会释放旧代租约并按 desired=running 自动重建。
    agent.states.clear()
    command_offset = len(agent.commands)
    recovery = await reconciler.run_once()
    assert recovery.scheduled == 1
    assert [command.action for command in agent.commands[command_offset:]] == [
        AgentAction.STATUS,
        AgentAction.START,
    ]
    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        assert deployment is not None and deployment.actual_state == DeploymentState.RUNNING
        assert deployment.runtime_generation == deployment.state_version


async def test_uncertain_start_keeps_lease_and_converges_by_status(
    isolated_session_factory,
) -> None:
    deployment_id, _ = await _seed_deployment_and_training(isolated_session_factory)
    agent = UncertainStartAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
    )

    first = await reconciler.run_once()
    assert first.scheduled == 1 and first.agent_errors == 1 and first.blocked == 1
    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        leases = list(await session.scalars(select(GPULease)))
        assert deployment is not None and deployment.actual_state == DeploymentState.STARTING
        assert "响应超时" in (deployment.error_message or "")
        assert len(leases) == 1 and leases[0].owner_id == deployment_id

    second = await reconciler.run_once()
    assert second.scheduled == 0 and second.blocked == 1
    assert [command.action for command in agent.commands] == [
        AgentAction.START,
        AgentAction.STATUS,
    ]
    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        leases = list(await session.scalars(select(GPULease)))
        assert deployment is not None and deployment.actual_state == DeploymentState.RUNNING
        assert deployment.error_message is None
        assert len(leases) == 1 and leases[0].owner_id == deployment_id


async def test_explicit_start_rejection_fails_and_releases_lease(
    isolated_session_factory,
) -> None:
    deployment_id, training_id = await _seed_deployment_and_training(isolated_session_factory)
    async with isolated_session_factory() as session, session.begin():
        training = await session.get(TrainingJob, training_id)
        assert training is not None
        training.actual_state = JobState.CANCELED
    agent = RejectStartAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
    )

    report = await reconciler.run_once()
    assert report.scheduled == 1 and report.agent_errors == 0
    async with isolated_session_factory() as session:
        deployment = await session.get(Deployment, deployment_id)
        assert deployment is not None and deployment.actual_state == DeploymentState.FAILED
        assert "GPU" in (deployment.error_message or "")
        assert list(await session.scalars(select(GPULease))) == []
