from __future__ import annotations

import hashlib
import io
import json
import tarfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.routes import training_jobs as training_routes
from app.core.config import Settings, get_settings
from app.models import Dataset, GPULease, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
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
from app.services.training_control import (
    TrainingControlError,
    build_training_archive,
    inspect_training_artifact,
    list_training_artifacts,
    publish_training_model_files,
)


def _settings(tmp_path: Path, **overrides) -> Settings:  # type: ignore[no-untyped-def]
    roots = {
        "model_root": tmp_path / "models",
        "dataset_root": tmp_path / "datasets",
        "checkpoint_root": tmp_path / "checkpoints",
        "evaluation_dataset_root": tmp_path / "evaluation-datasets",
        "evaluation_output_root": tmp_path / "evaluation-output",
        "node_agent_runtime_root": tmp_path / "runtime",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return Settings(_env_file=None, environment="test", gpu_count=2, **roots, **overrides)


def _write_safetensors(path: Path, tensor_name: str = "weight") -> None:
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00")


def _write_deployable_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(path / "model.safetensors")


def _write_training_outputs(output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    (output / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (output / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(output / "adapter_model.safetensors", "adapter")
    checkpoint = output / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step":100}', encoding="utf-8")
    merged = output / "merged"
    _write_deployable_model(merged)
    return checkpoint, merged


def _job(
    settings: Settings,
    *,
    state: JobState = JobState.SUCCEEDED,
    algorithm: TrainingAlgorithm = TrainingAlgorithm.LORA,
) -> TrainingJob:
    job_id = uuid.uuid4()
    output = settings.checkpoint_root / str(job_id)
    checkpoint, merged = _write_training_outputs(output)
    return TrainingJob(
        id=job_id,
        name=f"training-{job_id}",
        model_asset_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        stage=TrainingStage.SFT,
        algorithm=algorithm,
        desired_state=DesiredJobState.RUNNING,
        actual_state=state,
        gpu_ids=[0],
        training_config={"template": "qwen"},
        output_dir=str(output),
        checkpoint_path=str(checkpoint),
        adapter_path=str(output) if algorithm != TrainingAlgorithm.FREEZE else None,
        merged_model_path=str(merged if algorithm != TrainingAlgorithm.FREEZE else output),
    )


def test_training_create_uses_strict_server_derived_contract(client: TestClient) -> None:
    settings = get_settings()
    model_path = settings.model_root / f"training-contract-{uuid.uuid4()}"
    _write_deployable_model(model_path)
    model = client.post(
        "/api/v1/model-assets",
        json={
            "name": f"training-contract-{uuid.uuid4()}",
            "source_type": "manual",
            "local_path": str(model_path),
            "model_kind": "base",
            "status": "ready",
        },
    )
    assert model.status_code == 201, model.text
    dataset = client.post(
        "/api/v1/datasets/upload",
        data={"name": f"training-contract-{uuid.uuid4()}", "dataset_type": "sft"},
        files={
            "file": (
                "sft.jsonl",
                io.BytesIO(b'{"instruction":"Q","output":"A"}\n'),
                "application/jsonl",
            )
        },
    )
    assert dataset.status_code == 201, dataset.text
    payload = {
        "name": f"training-contract-{uuid.uuid4()}",
        "model_asset_id": model.json()["id"],
        "dataset_id": dataset.json()["id"],
        "stage": "sft",
        "algorithm": "lora",
        "gpu_ids": [1, 0],
        "training_config": {"template": "qwen", "learning_rate": 0.0001},
    }
    response = client.post("/api/v1/training-jobs", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["output_dir"] == str(settings.checkpoint_root / body["id"])
    assert not Path(body["output_dir"]).exists()
    assert body["training_config"]["num_train_epochs"] == 3.0
    assert body["training_config"]["learning_rate"] == 0.0001
    assert body["published_model_asset_id"] is None

    injected = client.post(
        "/api/v1/training-jobs",
        json={**payload, "name": f"injected-{uuid.uuid4()}", "output_dir": "/tmp/escape"},
    )
    assert injected.status_code == 422 and "extra_forbidden" in injected.text
    missing_template = client.post(
        "/api/v1/training-jobs",
        json={**payload, "name": f"missing-template-{uuid.uuid4()}", "training_config": {}},
    )
    assert missing_template.status_code == 422 and "template" in missing_template.text
    unknown = client.post(
        "/api/v1/training-jobs",
        json={
            **payload,
            "name": f"unknown-{uuid.uuid4()}",
            "training_config": {"template": "qwen", "deepspeed": "/tmp/config.json"},
        },
    )
    assert unknown.status_code == 422 and "extra_forbidden" in unknown.text
    wrong_type = client.post(
        "/api/v1/training-jobs",
        json={
            **payload,
            "name": f"wrong-type-{uuid.uuid4()}",
            "training_config": {"template": "qwen", "cutoff_len": "4096"},
        },
    )
    assert wrong_type.status_code == 422 and "int_type" in wrong_type.text

    embedding_path = settings.model_root / f"training-embedding-{uuid.uuid4()}"
    _write_deployable_model(embedding_path)
    embedding = client.post(
        "/api/v1/model-assets",
        json={
            "name": f"training-embedding-{uuid.uuid4()}",
            "source_type": "manual",
            "local_path": str(embedding_path),
            "model_kind": "embedding",
            "status": "ready",
        },
    )
    embedding_job = client.post(
        "/api/v1/training-jobs",
        json={**payload, "name": f"embedding-job-{uuid.uuid4()}", "model_asset_id": embedding.json()["id"]},
    )
    assert embedding_job.status_code == 422 and "Embedding" in embedding_job.text

    schema = client.get("/openapi.json").json()
    create_schema = schema["components"]["schemas"]["TrainingJobCreate"]
    assert create_schema["additionalProperties"] is False
    assert "output_dir" not in create_schema["properties"]
    assert "/api/v1/training-jobs/{job_id}/artifacts" in schema["paths"]
    assert "/api/v1/training-jobs/{job_id}/artifacts/{kind}/download" in schema["paths"]
    assert "/api/v1/training-jobs/{job_id}/publish-model" in schema["paths"]


def test_artifact_kinds_are_independent_and_archives_are_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job(settings)

    artifacts = list_training_artifacts(job, settings)
    assert [item.kind for item in artifacts] == ["checkpoint", "adapter", "merged", "full"]
    adapter = next(item for item in artifacts if item.kind == "adapter")
    full = next(item for item in artifacts if item.kind == "full")
    assert adapter.file_count == 4
    assert full.file_count > adapter.file_count

    first, descriptor = build_training_archive(job, "adapter", settings)
    second, _ = build_training_archive(job, "adapter", settings)
    try:
        assert first.read_bytes() == second.read_bytes()
        assert descriptor.archive_filename == f"training-{job.id}-adapter.tar.gz"
        with tarfile.open(first, "r:gz") as archive:
            names = archive.getnames()
        assert "artifact/adapter_model.safetensors" in names
        assert not any("merged" in name or "checkpoint-" in name for name in names)
    finally:
        first.unlink()
        second.unlink()


def test_artifact_scan_rejects_symlink_and_limits(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job(settings)
    output = Path(job.output_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (output / "escape").symlink_to(outside)
    with pytest.raises(TrainingControlError, match="软链接"):
        inspect_training_artifact(job, "full", settings)

    (output / "escape").unlink()
    limited = Settings(**{**settings.model_dump(), "training_artifact_max_files": 2})
    with pytest.raises(TrainingControlError, match="数量"):
        inspect_training_artifact(job, "full", limited)


@pytest.mark.parametrize("state", [JobState.CANCELED, JobState.FAILED])
def test_interrupted_artifacts_never_export_raw_full_or_pickle_state(
    tmp_path: Path,
    state: JobState,
) -> None:
    settings = _settings(tmp_path)
    job = _job(settings, state=state)
    checkpoint = Path(job.checkpoint_path or "")
    (checkpoint / "optimizer.pt").write_bytes(b"pickle")
    (checkpoint / "rng_state.pth").write_bytes(b"pickle")
    nested = checkpoint / "state"
    nested.mkdir()
    (nested / "scheduler.pkl").write_bytes(b"pickle")
    (nested / "safe-state.json").write_text("{}", encoding="utf-8")
    # raw output 根中即使还有更多未清理状态，取消/失败任务也不能暴露 full。
    (Path(job.output_dir) / "training_args.bin").write_bytes(b"pickle")

    artifacts = list_training_artifacts(job, settings)
    assert [item.kind for item in artifacts] == ["checkpoint", "adapter", "merged"]
    checkpoint_artifact = artifacts[0]
    assert checkpoint_artifact.file_count == 2
    with pytest.raises(TrainingControlError, match="没有可下载的 full"):
        build_training_archive(job, "full", settings)

    archive_path, _ = build_training_archive(job, "checkpoint", settings)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
        assert "artifact/trainer_state.json" in names
        assert "artifact/state/safe-state.json" in names
        assert not any(
            name.endswith((".bin", ".ckpt", ".joblib", ".pkl", ".pickle", ".pt", ".pth")) for name in names
        )
    finally:
        archive_path.unlink()


def test_publish_files_prefers_merged_and_never_references_checkpoint_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job(settings)

    first = publish_training_model_files(job, settings)
    second = publish_training_model_files(job, settings)
    assert first.created is True and second.created is False
    assert first.artifact_kind == second.artifact_kind == "merged"
    assert first.path == settings.model_root / f"trained-{job.id}"
    assert not first.path.is_relative_to(settings.checkpoint_root)
    assert first.checksum == second.checksum
    assert (first.path / "model.safetensors").is_file()
    assert not any(path.name.startswith(f".publish-{job.id}-") for path in settings.model_root.iterdir())


def test_publish_rejects_unsafe_or_incomplete_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = _job(settings)
    merged = Path(job.merged_model_path or "")
    (merged / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(TrainingControlError, match="不安全"):
        publish_training_model_files(job, settings)
    (merged / "pytorch_model.bin").unlink()
    (merged / "model.safetensors").write_bytes(b"invalid")
    with pytest.raises(TrainingControlError, match="safetensors"):
        publish_training_model_files(job, settings)


class _TrainingAgent:
    def __init__(self, terminal_metadata: dict | None = None) -> None:
        self.commands: list[AgentCommand] = []
        self.terminal_metadata = terminal_metadata

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        observed = (
            AgentWorkloadState.SUCCEEDED if self.terminal_metadata is not None else AgentWorkloadState.RUNNING
        )
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=observed,
            observed_at=datetime.now(UTC),
            metadata=self.terminal_metadata or {},
        )


class _TerminateTrainingAgent:
    def __init__(
        self,
        *,
        status_state: AgentWorkloadState,
        status_metadata: dict,
        stop_state: AgentWorkloadState = AgentWorkloadState.ABSENT,
        stop_metadata: dict | None = None,
    ) -> None:
        self.commands: list[AgentCommand] = []
        self.status_state = status_state
        self.status_metadata = status_metadata
        self.stop_state = stop_state
        self.stop_metadata = stop_metadata or {}

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        state = self.status_state if command.action == AgentAction.STATUS else self.stop_state
        metadata = self.status_metadata if command.action == AgentAction.STATUS else self.stop_metadata
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=state,
            observed_at=datetime.now(UTC),
            metadata=metadata,
        )


async def _seed_training(
    factory,  # type: ignore[no-untyped-def]
    settings: Settings,
    *,
    state: JobState,
    with_outputs: bool,
) -> tuple[uuid.UUID, Path]:
    model_path = settings.model_root / f"base-{uuid.uuid4()}"
    _write_deployable_model(model_path)
    dataset_path = settings.dataset_root / f"sft-{uuid.uuid4()}.jsonl"
    dataset_body = b'{"instruction":"Q","output":"A"}\n'
    dataset_path.write_bytes(dataset_body)
    job_id = uuid.uuid4()
    output = settings.checkpoint_root / str(job_id)
    checkpoint: Path | None = None
    merged: Path | None = None
    if with_outputs:
        checkpoint, merged = _write_training_outputs(output)
    async with factory() as session, session.begin():
        asset = ModelAsset(
            name=f"base-{uuid.uuid4()}",
            source_type=ModelSourceType.MANUAL,
            local_path=str(model_path),
            model_kind=ModelKind.BASE,
            status=AssetStatus.READY,
        )
        dataset = Dataset(
            name=f"sft-{uuid.uuid4()}",
            dataset_type=DatasetType.SFT,
            status=DatasetStatus.READY,
            file_name=dataset_path.name,
            local_path=str(dataset_path),
            record_count=1,
            size_bytes=len(dataset_body),
            sha256=hashlib.sha256(dataset_body).hexdigest(),
            schema_summary={"format": "jsonl", "record_format": "sft_alpaca"},
        )
        session.add_all([asset, dataset])
        await session.flush()
        job = TrainingJob(
            id=job_id,
            name=f"training-{uuid.uuid4()}",
            model_asset_id=asset.id,
            dataset_id=dataset.id,
            stage=TrainingStage.SFT,
            algorithm=TrainingAlgorithm.LORA,
            desired_state=DesiredJobState.RUNNING,
            actual_state=state,
            gpu_ids=[1, 0],
            training_config={"template": "qwen", "num_train_epochs": 1.0},
            output_dir=str(output),
            queued_at=datetime(2026, 8, 25, tzinfo=UTC) if state == JobState.QUEUED else None,
            runtime_generation=1 if state == JobState.RUNNING else 0,
        )
        if checkpoint is not None and merged is not None:
            job.checkpoint_path = str(checkpoint)
            job.adapter_path = str(output)
            job.merged_model_path = str(merged)
        session.add(job)
        if state == JobState.RUNNING:
            now = datetime.now(UTC)
            for gpu_id in job.gpu_ids:
                session.add(
                    GPULease(
                        gpu_index=gpu_id,
                        lease_group_id=job.id,
                        owner_type=LeaseOwnerType.TRAINING,
                        owner_id=job.id,
                        owner_name=job.name,
                        generation=1,
                        acquired_at=now,
                        heartbeat_at=now,
                        expires_at=now + timedelta(seconds=30),
                    )
                )
    return job_id, output


async def test_reconciler_sends_exact_training_execution(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.QUEUED,
        with_outputs=False,
    )
    agent = _TrainingAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    report = await reconciler.run_once()
    assert report.scheduled == 1 and report.preflight_failed == 0
    command = agent.commands[0]
    assert command.resources.gpu_ids == [0, 1]
    assert set(command.execution) == {
        "runner",
        "model_path",
        "dataset_path",
        "stage",
        "algorithm",
        "training_config",
        "output_dir",
    }
    assert command.execution["output_dir"] == str(output)
    assert command.execution["training_config"] == {
        "template": "qwen",
        "num_train_epochs": 1.0,
        "learning_rate": 2e-4,
        "cutoff_len": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "logging_steps": 10,
        "save_steps": 100,
        "warmup_ratio": 0.03,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "freeze_trainable_layers": 2,
        "seed": 42,
    }
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.RUNNING


async def test_reconciler_validates_success_artifacts_and_final_progress(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    metadata = {
        "progress": 99.0,
        "current_step": 100,
        "total_steps": 100,
        "metrics": {"loss": 0.1},
        "checkpoint_path": str(output / "checkpoint-100"),
        "adapter_path": str(output),
        "merged_model_path": str(output / "merged"),
    }
    reconciler = StateReconciler(
        isolated_session_factory,
        _TrainingAgent(metadata),
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    await reconciler.run_once()
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.SUCCEEDED
        assert job.progress == 100.0 and job.current_step == job.total_steps == 100
        assert job.metrics == {"loss": 0.1}
        assert job.merged_model_path == str(output / "merged")
        assert job.finished_at is not None and job.error_message is None
        assert list(await session.scalars(select(GPULease))) == []


async def test_terminate_probes_status_and_preserves_last_complete_checkpoint(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.desired_state = DesiredJobState.TERMINATED
        job.actual_state = JobState.CANCELING
        job.state_version += 1
        # 确认 metadata 是本轮 STATUS 保存，而不是 seed 遗留值。
        job.checkpoint_path = None
        job.adapter_path = None
        job.merged_model_path = None

    agent = _TerminateTrainingAgent(
        status_state=AgentWorkloadState.RUNNING,
        status_metadata={
            "progress": 50.0,
            "current_step": 50,
            "total_steps": 100,
            "checkpoint_path": str(output / "checkpoint-100"),
        },
    )
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    await reconciler.run_once()

    assert [command.action for command in agent.commands] == [
        AgentAction.STATUS,
        AgentAction.STOP,
    ]
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.CANCELED
        assert job.checkpoint_path == str(output / "checkpoint-100")
        assert job.current_step == 50 and job.total_steps == 100
        assert job.finished_at is not None
        assert list(await session.scalars(select(GPULease))) == []


async def test_terminate_race_keeps_naturally_completed_training_succeeded(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.desired_state = DesiredJobState.TERMINATED
        job.actual_state = JobState.CANCELING
        job.state_version += 1

    terminal_metadata = {
        "progress": 100.0,
        "current_step": 100,
        "total_steps": 100,
        "metrics": {"loss": 0.1},
        "checkpoint_path": str(output / "checkpoint-100"),
        "adapter_path": str(output),
        "merged_model_path": str(output / "merged"),
    }
    agent = _TerminateTrainingAgent(
        status_state=AgentWorkloadState.SUCCEEDED,
        status_metadata=terminal_metadata,
    )
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    await reconciler.run_once()

    # STATUS 已证明 code 0 成功后不能再发送 STOP，也不能降级为 CANCELED。
    assert [command.action for command in agent.commands] == [AgentAction.STATUS]
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.SUCCEEDED
        assert job.progress == 100.0 and job.merged_model_path == str(output / "merged")
        assert job.error_message is None
        assert list(await session.scalars(select(GPULease))) == []


async def test_stop_success_response_wins_over_terminate_intent(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.desired_state = DesiredJobState.TERMINATED
        job.actual_state = JobState.CANCELING
        job.state_version += 1

    terminal_metadata = {
        "checkpoint_path": str(output / "checkpoint-100"),
        "adapter_path": str(output),
        "merged_model_path": str(output / "merged"),
    }
    agent = _TerminateTrainingAgent(
        status_state=AgentWorkloadState.RUNNING,
        status_metadata={"checkpoint_path": str(output / "checkpoint-100")},
        stop_state=AgentWorkloadState.SUCCEEDED,
        stop_metadata=terminal_metadata,
    )
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    await reconciler.run_once()

    assert [command.action for command in agent.commands] == [
        AgentAction.STATUS,
        AgentAction.STOP,
    ]
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.SUCCEEDED
        assert job.checkpoint_path == str(output / "checkpoint-100")
        assert list(await session.scalars(select(GPULease))) == []


async def test_terminate_endpoint_is_idempotent_and_never_overwrites_terminal_state(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, _ = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=False,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.actual_state = JobState.SUCCEEDED
        job.finished_at = datetime.now(UTC)
        initial_version = job.state_version

    async with isolated_session_factory() as session:
        response = await training_routes.terminate_training_job(job_id, session)
        assert response.actual_state == JobState.SUCCEEDED.value
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.SUCCEEDED
        assert job.state_version == initial_version

    second_job_id, _ = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.QUEUED,
        with_outputs=False,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, second_job_id)
        assert job is not None
        job.desired_state = DesiredJobState.TERMINATED
        job.actual_state = JobState.CANCELING
        version = job.state_version
    async with isolated_session_factory() as session:
        response = await training_routes.terminate_training_job(second_job_id, session)
        assert response.actual_state == JobState.CANCELING.value
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, second_job_id)
        assert job is not None and job.state_version == version


async def test_reconciler_rejects_dataset_changed_after_upload(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, _ = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.QUEUED,
        with_outputs=False,
    )
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        dataset = await session.get(Dataset, job.dataset_id)
        assert dataset is not None
        with Path(dataset.local_path).open("ab") as output:
            output.write(b'{"instruction":"tampered","output":"data"}\n')
    agent = _TrainingAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    report = await reconciler.run_once()
    assert report.preflight_failed == 1 and report.scheduled == 0
    assert agent.commands == []
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.FAILED
        assert "SHA-256" in (job.error_message or "")
        assert list(await session.scalars(select(GPULease))) == []


async def test_reconciler_rejects_success_without_required_merged_artifact(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    metadata = {"adapter_path": str(output)}
    reconciler = StateReconciler(
        isolated_session_factory,
        _TrainingAgent(metadata),
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    await reconciler.run_once()
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.actual_state == JobState.FAILED
        assert "merged" in (job.error_message or "")
        assert list(await session.scalars(select(GPULease))) == []


async def test_publish_endpoint_is_database_idempotent(
    isolated_session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    job_id, output = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.RUNNING,
        with_outputs=True,
    )
    async with isolated_session_factory() as session, session.begin():
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.actual_state = JobState.SUCCEEDED
        job.checkpoint_path = str(output / "checkpoint-100")
        job.adapter_path = str(output)
        job.merged_model_path = str(output / "merged")
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)

    async with isolated_session_factory() as session:
        manifest = await training_routes.get_training_artifacts(job_id, session)
        assert [item.kind for item in manifest.artifacts] == [
            "checkpoint",
            "adapter",
            "merged",
            "full",
        ]
        download = await training_routes.download_training_artifact(job_id, "merged", session)
        assert download.headers["content-disposition"] == (
            f'attachment; filename="training-{job_id}-merged.tar.gz"'
        )
        archive_path = Path(download.path)
        assert archive_path.is_file()
        archive_path.unlink()

    async with isolated_session_factory() as session:
        first = await training_routes.publish_training_model(job_id, session)
    async with isolated_session_factory() as session:
        second = await training_routes.publish_training_model(job_id, session)
    assert first.id == second.id
    assert first.source_type == ModelSourceType.TRAINED
    assert first.model_kind == ModelKind.INSTRUCT
    assert Path(first.local_path).is_relative_to(settings.model_root)
    assert not Path(first.local_path).is_relative_to(settings.checkpoint_root)
    async with isolated_session_factory() as session:
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.published_model_asset_id == first.id
        assert await session.scalar(select(func.count()).select_from(ModelAsset)) == 2


async def test_publish_endpoint_rejects_nonterminal_job(
    isolated_session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    job_id, _ = await _seed_training(
        isolated_session_factory,
        settings,
        state=JobState.QUEUED,
        with_outputs=False,
    )
    monkeypatch.setattr(training_routes, "get_settings", lambda: settings)
    async with isolated_session_factory() as session:
        with pytest.raises(HTTPException) as caught:
            await training_routes.publish_training_model(job_id, session)
    assert caught.value.status_code == 409
