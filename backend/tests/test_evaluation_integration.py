from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.models import Dataset, EvaluationRun, GPULease, ModelAsset
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DesiredJobState,
    EvaluationTemplate,
    JobState,
    LeaseOwnerType,
    ModelKind,
    ModelSourceType,
)
from app.schemas.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentWorkloadState,
)
from app.services.gpu_scheduler import GPULeaseManager
from app.services.reconciler import StateReconciler


def _evaluation_row(sample_id: str = "sample-1", category: str = "general") -> bytes:
    return (
        json.dumps(
            {
                "id": sample_id,
                "task_type": "multiple_choice",
                "category": category,
                "question": "中国的首都是？",
                "choices": {"A": "北京", "B": "上海"},
                "answer": "A",
            },
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )


def _prepare_builtin(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = directory / f"{name}.jsonl"
    body = _evaluation_row(sample_id=f"{name}-1", category="general")
    dataset_path.write_bytes(body)
    (directory / f"{name}.manifest.json").write_text(
        json.dumps(
            {
                "output": {
                    "path": dataset_path.name,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "record_count": 1,
                },
                "benchmark": name,
                "conversion": {
                    "format": "openllmops-eval-jsonl-v1",
                    "partial": False,
                },
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    return dataset_path


def _create_model(
    seed_model_asset: Callable[..., dict[str, str]],
    kind: ModelKind,
    suffix: str,
) -> dict[str, str]:
    settings = get_settings()
    model_path = settings.model_root / f"evaluation-{suffix}-{uuid.uuid4()}"
    model_path.mkdir(parents=True)
    return seed_model_asset(
        model_path,
        kind=kind,
        name=f"evaluation-{suffix}-{uuid.uuid4()}",
    )


def _upload_custom_dataset(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": f"evaluation-custom-{uuid.uuid4()}", "dataset_type": "evaluation"},
        files={
            "file": (
                "custom.jsonl",
                io.BytesIO(_evaluation_row(sample_id="custom-1", category="domain")),
                "application/jsonl",
            )
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_evaluation_api_derives_execution_and_rejects_client_paths(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
) -> None:
    settings = get_settings()
    _prepare_builtin(settings.evaluation_dataset_root, "ceval")
    _prepare_builtin(settings.evaluation_dataset_root, "cmmlu")
    base = _create_model(seed_model_asset, ModelKind.BASE, "base")
    candidate = _create_model(seed_model_asset, ModelKind.INSTRUCT, "candidate")
    custom = _upload_custom_dataset(client)
    payload = {
        "name": f"evaluation-{uuid.uuid4()}",
        "base_model_asset_id": base["id"],
        "candidate_model_asset_id": candidate["id"],
        "custom_dataset_id": custom["id"],
        "builtin_datasets": ["cmmlu", "ceval"],
        "gpu_ids": [1, 0],
    }
    response = client.post("/api/v1/evaluation-runs", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["output_dir"] == str(settings.evaluation_output_root / body["id"])
    assert body["base_template"] == "base"
    assert body["candidate_template"] == "instruct"
    assert body["tensor_parallel_size"] == 2
    assert body["gpu_memory_utilization"] == settings.evaluation_gpu_memory_utilization
    assert body["concurrency"] == settings.evaluation_concurrency
    assert body["max_tokens"] == settings.evaluation_max_tokens
    assert body["metrics"] == {} and body["comparison"] == {}
    assert body["result_path"] is None and body["dataset_manifest_path"] is None

    injected = client.post(
        "/api/v1/evaluation-runs",
        json={**payload, "name": f"injected-{uuid.uuid4()}", "output_dir": "/tmp/escape"},
    )
    assert injected.status_code == 422
    assert "extra_forbidden" in injected.text


class RecordingEvaluationAgent:
    def __init__(self, terminal_metadata: dict | None = None) -> None:
        self.commands: list[AgentCommand] = []
        self.terminal_metadata = terminal_metadata

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        self.commands.append(command)
        if command.action == AgentAction.START:
            observed = AgentWorkloadState.RUNNING
            metadata = {}
        elif self.terminal_metadata is not None:
            observed = AgentWorkloadState.SUCCEEDED
            metadata = self.terminal_metadata
        else:
            observed = AgentWorkloadState.RUNNING
            metadata = {}
        return AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=observed,
            observed_at=datetime.now(UTC),
            metadata=metadata,
        )


def _isolated_settings(tmp_path: Path) -> Settings:
    roots = {
        "model_root": tmp_path / "models",
        "dataset_root": tmp_path / "datasets",
        "evaluation_dataset_root": tmp_path / "evaluation-datasets",
        "evaluation_output_root": tmp_path / "evaluation-output",
        "node_agent_runtime_root": tmp_path / "runtime",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return Settings(
        _env_file=None,
        environment="test",
        gpu_count=2,
        **roots,
    )


async def _seed_evaluation(
    factory,  # type: ignore[no-untyped-def]
    settings: Settings,
    *,
    builtin_datasets: list[str],
    with_custom: bool,
    actual_state: JobState = JobState.QUEUED,
    runtime_generation: int = 0,
) -> uuid.UUID:
    base_path = settings.model_root / f"base-{uuid.uuid4()}"
    candidate_path = settings.model_root / f"candidate-{uuid.uuid4()}"
    base_path.mkdir()
    candidate_path.mkdir()
    custom_path = settings.dataset_root / f"custom-{uuid.uuid4()}.jsonl"
    custom_path.write_bytes(_evaluation_row(sample_id="custom-1", category="domain"))
    run_id = uuid.uuid4()
    async with factory() as session, session.begin():
        base = ModelAsset(
            name=f"base-{uuid.uuid4()}",
            source_type=ModelSourceType.MANUAL,
            local_path=str(base_path),
            model_kind=ModelKind.BASE,
            status=AssetStatus.READY,
        )
        candidate = ModelAsset(
            name=f"candidate-{uuid.uuid4()}",
            source_type=ModelSourceType.MANUAL,
            local_path=str(candidate_path),
            model_kind=ModelKind.INSTRUCT,
            status=AssetStatus.READY,
        )
        dataset = Dataset(
            name=f"custom-{uuid.uuid4()}",
            dataset_type=DatasetType.EVALUATION,
            status=DatasetStatus.READY,
            file_name=custom_path.name,
            local_path=str(custom_path),
            record_count=1,
            size_bytes=custom_path.stat().st_size,
            sha256=hashlib.sha256(custom_path.read_bytes()).hexdigest(),
        )
        session.add_all([base, candidate, dataset])
        await session.flush()
        run = EvaluationRun(
            id=run_id,
            name=f"evaluation-{uuid.uuid4()}",
            base_model_asset_id=base.id,
            candidate_model_asset_id=candidate.id,
            custom_dataset_id=dataset.id if with_custom else None,
            builtin_datasets=builtin_datasets,
            base_template=EvaluationTemplate.BASE,
            candidate_template=EvaluationTemplate.INSTRUCT,
            output_dir=str(settings.evaluation_output_root / str(run_id)),
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            concurrency=3,
            max_tokens=64,
            desired_state=DesiredJobState.RUNNING,
            actual_state=actual_state,
            gpu_ids=[1, 0],
            queued_at=datetime(2026, 8, 25, tzinfo=UTC) if actual_state == JobState.QUEUED else None,
            runtime_generation=runtime_generation,
        )
        session.add(run)
    return run_id


async def test_reconciler_builds_exact_multi_dataset_execution(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _isolated_settings(tmp_path)
    ceval_path = _prepare_builtin(settings.evaluation_dataset_root, "ceval")
    cmmlu_path = _prepare_builtin(settings.evaluation_dataset_root, "cmmlu")
    run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=["cmmlu", "ceval"],
        with_custom=True,
    )
    agent = RecordingEvaluationAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )

    report = await reconciler.run_once()
    assert report.scheduled == 1 and report.preflight_failed == 0
    start = agent.commands[0]
    assert start.action == AgentAction.START
    assert start.resources.gpu_ids == [0, 1]
    execution = start.execution
    assert set(execution) == {
        "runner",
        "base_model_path",
        "candidate_model_path",
        "base_template",
        "candidate_template",
        "datasets",
        "output_dir",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "concurrency",
        "max_tokens",
    }
    assert execution["base_template"] == "base"
    assert execution["candidate_template"] == "instruct"
    assert execution["output_dir"] == str(settings.evaluation_output_root / str(run_id))
    assert execution["tensor_parallel_size"] == len(start.resources.gpu_ids) == 2
    assert execution["gpu_memory_utilization"] == 0.85
    assert execution["concurrency"] == 3 and execution["max_tokens"] == 64
    datasets = execution["datasets"]
    assert datasets[:2] == [
        {"name": "ceval", "path": str(ceval_path)},
        {"name": "cmmlu", "path": str(cmmlu_path)},
    ]
    assert datasets[2]["name"].startswith("custom-")
    assert Path(datasets[2]["path"]).is_relative_to(settings.dataset_root)


@pytest.mark.parametrize(
    ("failure_mode", "expected_error"),
    [
        ("missing", "内置评测数据集 ceval"),
        ("jsonl_tampered", "SHA-256 不一致"),
        ("manifest_tampered", "SHA-256 不一致"),
        ("partial", "partial"),
    ],
)
async def test_invalid_builtin_fails_visibly_without_agent_or_lease(
    isolated_session_factory,
    tmp_path: Path,
    failure_mode: str,
    expected_error: str,
) -> None:
    settings = _isolated_settings(tmp_path)
    if failure_mode != "missing":
        dataset_path = _prepare_builtin(settings.evaluation_dataset_root, "ceval")
        manifest_path = dataset_path.with_suffix(".manifest.json")
        if failure_mode == "jsonl_tampered":
            with dataset_path.open("ab") as output:
                output.write(_evaluation_row(sample_id="ceval-tampered"))
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if failure_mode == "manifest_tampered":
                manifest["output"]["sha256"] = "b" * 64
            else:
                manifest["conversion"]["partial"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=["ceval"],
        with_custom=False,
    )
    agent = RecordingEvaluationAgent()
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
        run = await session.get(EvaluationRun, run_id)
        assert run is not None and run.actual_state == JobState.FAILED
        assert expected_error in (run.error_message or "")
        assert run.finished_at is not None
        assert list(await session.scalars(select(GPULease))) == []


async def test_preflight_failure_does_not_block_later_fifo_item(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _isolated_settings(tmp_path)
    bad_run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=["ceval"],
        with_custom=False,
    )
    good_run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=[],
        with_custom=True,
    )
    async with isolated_session_factory() as session, session.begin():
        good = await session.get(EvaluationRun, good_run_id)
        assert good is not None
        good.queued_at = datetime(2026, 8, 25, tzinfo=UTC) + timedelta(seconds=1)

    agent = RecordingEvaluationAgent()
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )
    report = await reconciler.run_once()

    assert report.preflight_failed == 1 and report.scheduled == 1
    assert len(agent.commands) == 1 and agent.commands[0].owner.id == good_run_id
    async with isolated_session_factory() as session:
        bad = await session.get(EvaluationRun, bad_run_id)
        good = await session.get(EvaluationRun, good_run_id)
        assert bad is not None and bad.actual_state == JobState.FAILED
        assert good is not None and good.actual_state == JobState.RUNNING
        leases = list(await session.scalars(select(GPULease).order_by(GPULease.gpu_index)))
        assert [lease.owner_id for lease in leases] == [good_run_id, good_run_id]


def _success_metadata(settings: Settings, run_id: uuid.UUID, generation: int) -> dict:
    output_dir = settings.evaluation_output_root / str(run_id)
    output_dir.mkdir()
    result_path = output_dir / "pair-report.json"
    result_path.write_text("{}", encoding="utf-8")
    manifest_path = (
        settings.node_agent_runtime_root
        / "contract"
        / "evaluation"
        / str(run_id)
        / str(generation)
        / "dataset-manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    digest = "a" * 64
    baseline = {
        "dataset_sha256": digest,
        "model_name": "baseline",
        "template": "base",
        "total": 2,
        "correct": 1,
        "invalid": 0,
        "accuracy_percent": 50.0,
        "average_latency_ms": 10.0,
        "categories": [
            {
                "category": "ceval/general",
                "total": 2,
                "correct": 1,
                "invalid": 0,
                "accuracy_percent": 50.0,
            }
        ],
    }
    candidate = {
        **baseline,
        "model_name": "candidate",
        "template": "instruct",
        "correct": 2,
        "accuracy_percent": 100.0,
        "categories": [
            {
                "category": "ceval/general",
                "total": 2,
                "correct": 2,
                "invalid": 0,
                "accuracy_percent": 100.0,
            }
        ],
    }
    comparison = {
        "dataset_sha256": digest,
        "baseline_model": "baseline",
        "candidate_model": "candidate",
        "baseline_percent": 50.0,
        "candidate_percent": 100.0,
        "percentage_point_change": 50.0,
        "relative_change_percent": 100.0,
        "comparable": True,
        "reason": None,
        "category_changes": [
            {
                "category": "ceval/general",
                "baseline_percent": 50.0,
                "candidate_percent": 100.0,
                "percentage_point_change": 50.0,
            }
        ],
    }
    return {
        "metrics": {"baseline": baseline, "candidate": candidate},
        "comparison": comparison,
        "result_path": str(result_path),
        "dataset_manifest_path": str(manifest_path),
    }


async def _lease_running_evaluation(factory, run_id: uuid.UUID) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    async with factory() as session, session.begin():
        run = await session.get(EvaluationRun, run_id)
        assert run is not None
        run.actual_state = JobState.RUNNING
        run.runtime_generation = 1
        run.queued_at = None
        for gpu_id in run.gpu_ids:
            session.add(
                GPULease(
                    gpu_index=gpu_id,
                    lease_group_id=run.id,
                    owner_type=LeaseOwnerType.EVALUATION,
                    owner_id=run.id,
                    owner_name=run.name,
                    generation=1,
                    acquired_at=now,
                    heartbeat_at=now,
                    expires_at=now + timedelta(seconds=30),
                )
            )


async def test_success_metadata_is_strictly_validated_and_persisted(
    isolated_session_factory,
    tmp_path: Path,
) -> None:
    settings = _isolated_settings(tmp_path)
    run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=[],
        with_custom=True,
    )
    await _lease_running_evaluation(isolated_session_factory, run_id)
    metadata = _success_metadata(settings, run_id, 1)
    agent = RecordingEvaluationAgent(metadata)
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )

    await reconciler.run_once()
    async with isolated_session_factory() as session:
        run = await session.get(EvaluationRun, run_id)
        assert run is not None and run.actual_state == JobState.SUCCEEDED
        assert run.metrics["baseline"]["accuracy_percent"] == 50.0
        assert run.metrics["candidate"]["accuracy_percent"] == 100.0
        assert run.comparison["percentage_point_change"] == 50.0
        assert run.result_path == metadata["result_path"]
        assert run.dataset_manifest_path == metadata["dataset_manifest_path"]
        assert run.finished_at is not None and run.error_message is None
        assert list(await session.scalars(select(GPULease))) == []


@pytest.mark.parametrize("invalid_kind", ["path", "type", "counts", "model_name"])
async def test_malicious_or_invalid_success_metadata_marks_run_failed(
    isolated_session_factory,
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    settings = _isolated_settings(tmp_path)
    run_id = await _seed_evaluation(
        isolated_session_factory,
        settings,
        builtin_datasets=[],
        with_custom=True,
    )
    await _lease_running_evaluation(isolated_session_factory, run_id)
    metadata = _success_metadata(settings, run_id, 1)
    if invalid_kind == "path":
        metadata["result_path"] = "/tmp/attacker/pair-report.json"
    elif invalid_kind == "type":
        metadata["metrics"]["baseline"]["total"] = "2"
    elif invalid_kind == "counts":
        metadata["metrics"]["baseline"]["correct"] = 2
        metadata["metrics"]["baseline"]["invalid"] = 1
        metadata["metrics"]["baseline"]["accuracy_percent"] = 100.0
    else:
        metadata["metrics"]["baseline"]["model_name"] = "attacker-controlled"
        metadata["comparison"]["baseline_model"] = "attacker-controlled"
    agent = RecordingEvaluationAgent(metadata)
    reconciler = StateReconciler(
        isolated_session_factory,
        agent,
        GPULeaseManager(ttl_seconds=30),
        settings=settings,
    )

    await reconciler.run_once()
    async with isolated_session_factory() as session:
        run = await session.get(EvaluationRun, run_id)
        assert run is not None and run.actual_state == JobState.FAILED
        expected_error = "result_path" if invalid_kind == "path" else "结构无效"
        assert expected_error in (run.error_message or "")
        assert run.metrics == {} and run.result_path is None
        assert list(await session.scalars(select(GPULease))) == []
