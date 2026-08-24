import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.routes.deployments import delete_deployment
from app.api.routes.evaluations import delete_evaluation_run
from app.api.routes.model_assets import delete_model_asset
from app.api.routes.training_jobs import delete_training_job
from app.models import Dataset, Deployment, EvaluationRun, GPULease, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DeploymentState,
    DeploymentTaskType,
    DesiredJobState,
    DesiredServiceState,
    EvaluationTemplate,
    JobState,
    LeaseOwnerType,
    ModelKind,
    ModelSourceType,
    TrainingAlgorithm,
    TrainingStage,
)
from app.schemas.agent_contract import AgentCommand, AgentCommandResponse, AgentWorkloadState
from app.services.node_agent import NodeAgentError
from app.services.workload_cleanup import CLEANUP_EXECUTION


class FakeCleanupAgent:
    def __init__(
        self,
        state: AgentWorkloadState = AgentWorkloadState.ABSENT,
        *,
        accepted: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.state = state
        self.accepted = accepted
        self.error = error
        self.commands: list[AgentCommand] = []

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=self.accepted,
            observed_state=self.state,
            observed_at=datetime.now(UTC),
            message=None if self.accepted else "容器仍在运行",
            error_code=None if self.accepted else "workload_conflict",
        )


async def _common_records(session: AsyncSession, root: Path) -> tuple[ModelAsset, Dataset]:
    suffix = uuid.uuid4().hex
    model_path = root / f"model-{suffix}"
    dataset_path = root / f"dataset-{suffix}.jsonl"
    model_path.mkdir()
    dataset_path.write_text('{"text":"demo"}\n', encoding="utf-8")
    asset = ModelAsset(
        name=f"model-{suffix}",
        source_type=ModelSourceType.MANUAL,
        local_path=str(model_path),
        model_kind=ModelKind.INSTRUCT,
        status=AssetStatus.READY,
    )
    dataset = Dataset(
        name=f"dataset-{suffix}",
        dataset_type=DatasetType.SFT,
        status=DatasetStatus.READY,
        file_name=dataset_path.name,
        local_path=str(dataset_path),
    )
    session.add_all([asset, dataset])
    await session.flush()
    return asset, dataset


def _lease(
    owner_type: LeaseOwnerType,
    owner_id: uuid.UUID,
    owner_name: str,
    generation: int,
    gpu_index: int,
) -> GPULease:
    now = datetime.now(UTC)
    return GPULease(
        gpu_index=gpu_index,
        lease_group_id=uuid.uuid4(),
        owner_type=owner_type,
        owner_id=owner_id,
        owner_name=owner_name,
        generation=generation,
        acquired_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(seconds=30),
    )


@pytest.mark.asyncio
async def test_terminal_delete_confirms_absent_for_every_workload_and_keeps_outputs(
    isolated_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    async with isolated_session_factory() as session:
        asset, dataset = await _common_records(session, tmp_path)
        deployment = Deployment(
            name=f"deployment-{uuid.uuid4()}",
            served_model_name=f"served-{uuid.uuid4()}",
            model_asset_id=asset.id,
            task_type=DeploymentTaskType.GENERATE,
            desired_state=DesiredServiceState.STOPPED,
            actual_state=DeploymentState.FAILED,
            gpu_ids=[0],
            tensor_parallel_size=1,
            runtime_generation=2,
        )
        output_dir = tmp_path / f"checkpoint-{uuid.uuid4()}"
        output_dir.mkdir()
        retained_file = output_dir / "trainer-state.json"
        retained_file.write_text("{}", encoding="utf-8")
        evaluation_output_dir = tmp_path / f"evaluation-{uuid.uuid4()}"
        evaluation_output_dir.mkdir()
        retained_report = evaluation_output_dir / "pair-report.json"
        retained_report.write_text('{"comparison":{}}', encoding="utf-8")
        training = TrainingJob(
            name=f"training-{uuid.uuid4()}",
            model_asset_id=asset.id,
            dataset_id=dataset.id,
            stage=TrainingStage.SFT,
            algorithm=TrainingAlgorithm.LORA,
            desired_state=DesiredJobState.RUNNING,
            actual_state=JobState.SUCCEEDED,
            gpu_ids=[1],
            output_dir=str(output_dir),
            runtime_generation=3,
        )
        evaluation = EvaluationRun(
            name=f"evaluation-{uuid.uuid4()}",
            base_model_asset_id=asset.id,
            candidate_model_asset_id=asset.id,
            builtin_datasets=["ceval"],
            base_template=EvaluationTemplate.INSTRUCT,
            candidate_template=EvaluationTemplate.INSTRUCT,
            output_dir=str(evaluation_output_dir),
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            concurrency=1,
            max_tokens=32,
            desired_state=DesiredJobState.RUNNING,
            actual_state=JobState.FAILED,
            gpu_ids=[0],
            runtime_generation=4,
        )
        session.add_all([deployment, training, evaluation])
        await session.flush()
        identities = (
            (LeaseOwnerType.DEPLOYMENT, deployment.id, deployment.name, 2, 0),
            (LeaseOwnerType.TRAINING, training.id, training.name, 3, 1),
            (LeaseOwnerType.EVALUATION, evaluation.id, evaluation.name, 4, 0),
        )
        # 顺序删除，每次成功确认 absent 后该卡租约立即在同一事务释放。
        for owner_type, owner_id, owner_name, generation, gpu_id in identities:
            session.add(_lease(owner_type, owner_id, owner_name, generation, gpu_id))
            await session.commit()
            agent = FakeCleanupAgent()
            if owner_type == LeaseOwnerType.DEPLOYMENT:
                response = await delete_deployment(owner_id, session, agent)
                model = Deployment
            elif owner_type == LeaseOwnerType.TRAINING:
                response = await delete_training_job(owner_id, session, agent)
                model = TrainingJob
            else:
                response = await delete_evaluation_run(owner_id, session, agent)
                model = EvaluationRun

            assert response.status_code == 204
            assert await session.get(model, owner_id) is None
            assert (
                await session.scalar(
                    select(GPULease).where(
                        GPULease.owner_type == owner_type,
                        GPULease.owner_id == owner_id,
                    )
                )
                is None
            )
            command = agent.commands[0]
            assert command.action.value == "stop"
            assert command.execution == CLEANUP_EXECUTION
            assert command.owner.generation == generation

        assert retained_file.read_text(encoding="utf-8") == "{}"
        assert retained_report.read_text(encoding="utf-8") == '{"comparison":{}}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "expected_status"),
    [
        (FakeCleanupAgent(error=NodeAgentError("timeout after remove")), 503),
        (None, 503),
        (FakeCleanupAgent(AgentWorkloadState.FAILED, accepted=False), 409),
        (FakeCleanupAgent(AgentWorkloadState.SUCCEEDED), 409),
    ],
)
async def test_uncertain_or_non_absent_cleanup_preserves_database_and_lease(
    isolated_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
    agent: FakeCleanupAgent | None,
    expected_status: int,
) -> None:
    async with isolated_session_factory() as session:
        asset, _ = await _common_records(session, tmp_path)
        deployment = Deployment(
            name=f"deployment-{uuid.uuid4()}",
            served_model_name=f"served-{uuid.uuid4()}",
            model_asset_id=asset.id,
            task_type=DeploymentTaskType.GENERATE,
            desired_state=DesiredServiceState.STOPPED,
            actual_state=DeploymentState.FAILED,
            gpu_ids=[0],
            tensor_parallel_size=1,
            runtime_generation=7,
        )
        session.add(deployment)
        await session.flush()
        session.add(
            _lease(
                LeaseOwnerType.DEPLOYMENT,
                deployment.id,
                deployment.name,
                7,
                0,
            )
        )
        await session.commit()

        with pytest.raises(HTTPException) as raised:
            await delete_deployment(deployment.id, session, agent)

        assert raised.value.status_code == expected_status
        assert await session.get(Deployment, deployment.id) is not None
        assert (
            await session.scalar(
                select(GPULease).where(
                    GPULease.owner_type == LeaseOwnerType.DEPLOYMENT,
                    GPULease.owner_id == deployment.id,
                )
            )
            is not None
        )


@pytest.mark.asyncio
async def test_published_training_asset_cannot_be_soft_deleted(
    isolated_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    async with isolated_session_factory() as session:
        base, dataset = await _common_records(session, tmp_path)
        published_path = tmp_path / f"published-{uuid.uuid4()}"
        published_path.mkdir()
        published = ModelAsset(
            name=f"published-{uuid.uuid4()}",
            source_type=ModelSourceType.TRAINED,
            local_path=str(published_path),
            model_kind=ModelKind.INSTRUCT,
            status=AssetStatus.READY,
        )
        session.add(published)
        await session.flush()
        job = TrainingJob(
            name=f"publisher-{uuid.uuid4()}",
            model_asset_id=base.id,
            dataset_id=dataset.id,
            stage=TrainingStage.SFT,
            algorithm=TrainingAlgorithm.LORA,
            actual_state=JobState.SUCCEEDED,
            gpu_ids=[0],
            output_dir=str(tmp_path / "publisher-output"),
            published_model_asset_id=published.id,
        )
        session.add(job)
        await session.commit()

        with pytest.raises(HTTPException) as raised:
            await delete_model_asset(published.id, session)

        assert raised.value.status_code == 409
        assert (await session.get(ModelAsset, published.id)).deleted_at is None  # type: ignore[union-attr]
