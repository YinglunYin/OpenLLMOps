from datetime import UTC, datetime, timedelta

from sqlalchemy import select

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


async def _seed_deployment_and_training(factory):  # type: ignore[no-untyped-def]
    queued_at = datetime(2026, 8, 24, tzinfo=UTC)
    async with factory() as session, session.begin():
        asset = ModelAsset(
            name="base",
            source_type=ModelSourceType.MANUAL,
            local_path="/models/base",
            model_kind=ModelKind.BASE,
            status=AssetStatus.READY,
        )
        dataset = Dataset(
            name="sft",
            dataset_type=DatasetType.SFT,
            status=DatasetStatus.READY,
            file_name="sft.jsonl",
            local_path="/datasets/sft.jsonl",
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
            name="domain-sft",
            model_asset_id=asset.id,
            dataset_id=dataset.id,
            stage=TrainingStage.SFT,
            algorithm=TrainingAlgorithm.LORA,
            actual_state=JobState.QUEUED,
            gpu_ids=[0],
            output_dir="/checkpoints/domain-sft",
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
