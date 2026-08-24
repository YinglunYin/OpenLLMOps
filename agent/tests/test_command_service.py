import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from openllmops_agent.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
)
from openllmops_agent.command_service import CommandProcessor, CommandStateStore
from openllmops_agent.config import Settings
from openllmops_agent.docker_runner import InvalidWorkload, WorkloadConflict, WorkloadNotFound
from openllmops_agent.schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
    TrainingLaunchRequest,
    WorkloadInfo,
)


class FakeRunner:
    def __init__(self) -> None:
        self.workloads: dict[tuple[str, UUID], WorkloadInfo] = {}
        self.inference_requests: list[InferenceLaunchRequest] = []
        self.training_requests: list[TrainingLaunchRequest] = []
        self.evaluation_requests: list[EvaluationLaunchRequest] = []
        self.stop_calls = 0
        self.cleanup_calls: list[tuple[str, UUID, int]] = []
        self.quiesce_calls: list[tuple[UUID, int, int]] = []
        self.quiesce_result = True
        self.training_metadata_completed_calls: list[bool] = []

    def prepare_contract_start(
        self, owner_type: str, workload_id: UUID, generation: int
    ) -> WorkloadInfo | None:
        existing = self.workloads.get((owner_type, workload_id))
        if existing is None:
            return None
        if existing.generation > generation:
            raise WorkloadConflict("stale")
        if existing.generation == generation:
            return existing
        if existing.status == "running":
            raise WorkloadConflict("旧 generation 仍在运行")
        return None

    def launch_inference(self, request: InferenceLaunchRequest) -> WorkloadInfo:
        self.inference_requests.append(request)
        info = WorkloadInfo(
            name=f"openllmops-inference-{request.deployment_id}",
            workload_id=request.deployment_id,
            kind="inference",
            image=request.image,
            status="running",
            gpu_ids=request.gpu_ids,
            service_type=request.service_type,
            endpoint=f"http://runtime:{request.port}",
            port=request.port,
            generation=request.generation,
            health_status="healthy",
        )
        self.workloads[("deployment", request.deployment_id)] = info
        return info

    def launch_training(self, request: TrainingLaunchRequest) -> WorkloadInfo:
        self.training_requests.append(request)
        info = WorkloadInfo(
            name=f"openllmops-training-{request.job_id}",
            workload_id=request.job_id,
            kind="training",
            image=request.image,
            status="running",
            gpu_ids=request.gpu_ids,
            generation=request.generation,
        )
        self.workloads[("training", request.job_id)] = info
        return info

    def launch_evaluation(self, request: EvaluationLaunchRequest) -> WorkloadInfo:
        self.evaluation_requests.append(request)
        info = WorkloadInfo(
            name=f"openllmops-evaluation-{request.run_id}",
            workload_id=request.run_id,
            kind="evaluation",
            image=request.image,
            status="running",
            gpu_ids=request.gpu_ids,
            generation=request.generation,
        )
        self.workloads[("evaluation", request.run_id)] = info
        return info

    def get_contract_workload(self, owner_type: str, workload_id: UUID) -> WorkloadInfo:
        try:
            return self.workloads[(owner_type, workload_id)]
        except KeyError as exc:
            raise WorkloadNotFound("absent") from exc

    def stop_contract_workload(
        self, owner_type: str, workload_id: UUID, generation: int
    ) -> WorkloadInfo | None:
        info = self.get_contract_workload(owner_type, workload_id)
        if info.generation != generation:
            raise WorkloadConflict("generation mismatch")
        self.stop_calls += 1
        if owner_type in {"training", "evaluation"} and info.status == "exited" and info.exit_code == 0:
            return info
        del self.workloads[(owner_type, workload_id)]
        return None

    def cleanup_contract_workload(self, owner_type: str, workload_id: UUID, generation: int) -> None:
        self.cleanup_calls.append((owner_type, workload_id, generation))
        info = self.workloads.get((owner_type, workload_id))
        if info is None:
            return
        if info.generation != generation:
            raise WorkloadConflict("generation mismatch")
        if info.status not in {"exited", "dead"}:
            raise WorkloadConflict("not terminal")
        del self.workloads[(owner_type, workload_id)]

    def quiesce_failed_inference(
        self,
        workload_id: UUID,
        generation: int,
        *,
        timeout_seconds: int,
    ) -> bool:
        self.quiesce_calls.append((workload_id, generation, timeout_seconds))
        if self.quiesce_result:
            key = ("deployment", workload_id)
            if key in self.workloads:
                self.workloads[key] = self.workloads[key].model_copy(
                    update={"status": "exited", "exit_code": 1}
                )
        return self.quiesce_result

    def training_metadata(self, workload_id: UUID, *, completed: bool = False) -> dict:
        self.training_metadata_completed_calls.append(completed)
        return {"progress": 25.0, "current_step": 1, "total_steps": 4}

    def evaluation_metadata(self, workload_id: UUID) -> dict:
        return {
            "metrics": {"baseline": {"accuracy_percent": 50.0}},
            "comparison": {"percentage_point_change": 25.0},
            "result_path": f"/evaluations/{workload_id}/pair-report.json",
            "dataset_manifest_path": f"/runtime/{workload_id}/dataset-manifest.json",
        }


def settings_for(tmp_path: Path, **overrides) -> Settings:  # type: ignore[no-untyped-def]
    settings = Settings(
        node_agent_token="a" * 32,
        model_root=tmp_path / "models",
        dataset_root=tmp_path / "datasets",
        evaluation_dataset_root=tmp_path / "evaluation-datasets",
        evaluation_output_root=tmp_path / "evaluation-output",
        checkpoint_root=tmp_path / "checkpoints",
        training_config_root=tmp_path / "configs",
        runtime_root=tmp_path / "runtime",
        **overrides,
    )
    settings.ensure_layout()
    return settings


def deployment_command(
    owner_id: UUID,
    *,
    request_id: UUID | None = None,
    generation: int = 2,
    max_model_len: int = 4096,
) -> AgentCommand:
    return AgentCommand(
        request_id=request_id or uuid4(),
        action=AgentAction.START,
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=generation),
        resources=AgentResourceRequest(gpu_ids=[0, 1]),
        execution={
            "runner": "vllm",
            "service_type": "generate",
            "model_path": "/srv/openllmops/models/demo",
            "served_model_name": "demo",
            "port": 8123,
            "tensor_parallel_size": 2,
            "simplified_config": {"max_model_len": max_model_len},
            "vllm_args": {"gpu_memory_utilization": 0.9},
        },
    )


def cleanup_command(
    owner_type: str,
    owner_id: UUID,
    *,
    generation: int = 1,
    execution: dict | None = None,
) -> AgentCommand:
    return AgentCommand(
        action=AgentAction.STOP,
        owner=AgentOwner(type=owner_type, id=owner_id, name="cleanup", generation=generation),  # type: ignore[arg-type]
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={"cleanup_terminal": True} if execution is None else execution,
    )


@pytest.mark.parametrize("exit_code", [0, 1])
def test_terminal_cleanup_removes_success_or_failure_and_reports_absent(
    tmp_path: Path,
    exit_code: int,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    runner.workloads[("training", owner_id)] = WorkloadInfo(
        name=f"openllmops-training-{owner_id}",
        workload_id=owner_id,
        kind="training",
        image="training:test",
        status="exited",
        exit_code=exit_code,
        gpu_ids=[0],
        generation=3,
    )

    result = processor.execute(cleanup_command("training", owner_id, generation=3))

    assert result.status_code == 200
    assert result.response.accepted is True
    assert result.response.observed_state == AgentWorkloadState.ABSENT
    assert ("training", owner_id) not in runner.workloads


def test_terminal_cleanup_is_idempotent_when_target_is_already_absent(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()

    first = processor.execute(cleanup_command("evaluation", owner_id, generation=2))
    repeated = processor.execute(cleanup_command("evaluation", owner_id, generation=2))

    assert first.response.accepted is True
    assert first.response.observed_state == AgentWorkloadState.ABSENT
    assert repeated.response.accepted is True
    assert repeated.response.observed_state == AgentWorkloadState.ABSENT
    assert runner.cleanup_calls == [
        ("evaluation", owner_id, 2),
        ("evaluation", owner_id, 2),
    ]


@pytest.mark.parametrize(
    "execution",
    [
        {"cleanup_terminal": False},
        {"cleanup_terminal": 1},
        {"cleanup_terminal": True, "force": True},
        {"force": True},
    ],
)
def test_terminal_cleanup_marker_is_exact_and_strict(
    tmp_path: Path,
    execution: dict,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]

    result = processor.execute(cleanup_command("training", uuid4(), execution=execution))

    assert result.status_code == 422
    assert result.response.accepted is False
    assert result.response.error_code == "invalid_execution"
    assert runner.cleanup_calls == []


@pytest.mark.parametrize("status", ["created", "running", "restarting", "paused", "removing"])
def test_terminal_cleanup_rejects_nonterminal_container_without_removing_it(
    tmp_path: Path,
    status: str,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    key = ("deployment", owner_id)
    runner.workloads[key] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm:test",
        status=status,
        gpu_ids=[0],
        generation=1,
    )

    result = processor.execute(cleanup_command("deployment", owner_id))

    assert result.status_code == 409
    assert result.response.accepted is False
    assert result.response.observed_state != AgentWorkloadState.ABSENT
    assert key in runner.workloads


def test_terminal_cleanup_rejects_generation_mismatch_and_preserves_container(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    key = ("training", owner_id)
    runner.workloads[key] = WorkloadInfo(
        name=f"openllmops-training-{owner_id}",
        workload_id=owner_id,
        kind="training",
        image="training:test",
        status="exited",
        exit_code=0,
        gpu_ids=[0],
        generation=2,
    )

    result = processor.execute(cleanup_command("training", owner_id, generation=3))

    assert result.status_code == 409
    assert result.response.accepted is False
    assert result.response.observed_state != AgentWorkloadState.ABSENT
    assert key in runner.workloads


def test_request_and_generation_are_idempotent_and_stale_is_rejected(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    command = deployment_command(owner_id)

    first = processor.execute(command)
    repeated = processor.execute(command)
    same_generation = processor.execute(deployment_command(owner_id))

    assert first.response.observed_state == AgentWorkloadState.RUNNING
    assert repeated.response == first.response
    assert same_generation.response.accepted
    assert len(runner.inference_requests) == 1
    assert runner.inference_requests[0].generation == 2
    assert runner.inference_requests[0].vllm_args["max_model_len"] == 4096

    changed = processor.execute(deployment_command(owner_id, max_model_len=8192))
    assert changed.status_code == 409
    assert changed.response.error_code == "generation_reused"

    newer = processor.execute(deployment_command(owner_id, generation=3))
    assert newer.status_code == 409  # 非抢占策略拒绝仍在运行的第 2 代容器。
    assert processor.execute(command).response.error_code == "stale_generation"

    stale = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0, 1]),
        execution={},
    )
    stale_result = processor.execute(stale)
    assert stale_result.status_code == 409
    assert stale_result.response.error_code == "stale_generation"

    # generation 水位持久化，构造新的处理器后仍拒绝旧命令。
    restarted = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    assert restarted.execute(stale).response.error_code == "stale_generation"


@pytest.mark.parametrize(
    ("status", "health_status", "expected"),
    [
        ("running", None, AgentWorkloadState.STARTING),
        ("running", "starting", AgentWorkloadState.STARTING),
        ("running", "unhealthy", AgentWorkloadState.STARTING),
        ("restarting", "unhealthy", AgentWorkloadState.STARTING),
        ("paused", "healthy", AgentWorkloadState.STARTING),
        ("running", "healthy", AgentWorkloadState.RUNNING),
        ("exited", "unhealthy", AgentWorkloadState.FAILED),
    ],
)
def test_inference_requires_healthy_container_before_reporting_running(
    tmp_path: Path,
    status: str,
    health_status: str | None,
    expected: AgentWorkloadState,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    runner.workloads[("deployment", owner_id)] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm/vllm-openai:v0.27.1",
        status=status,
        health_status=health_status,  # type: ignore[arg-type]
        gpu_ids=[0],
        service_type="generate",
        endpoint=f"http://openllmops-inference-{owner_id}:8000",
        port=8000,
        generation=1,
        exit_code=1 if status == "exited" else None,
        health_failing_streak=1 if health_status == "unhealthy" else 0,
        created_at=datetime.now(UTC),
    )
    command = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )

    result = processor.execute(command)

    assert result.response.observed_state == expected
    if expected == AgentWorkloadState.RUNNING:
        assert result.response.metadata["endpoint"].endswith(":8000")
    else:
        assert "endpoint" not in result.response.metadata


def test_inference_health_can_recover_without_restarting_or_releasing_gpu(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    key = ("deployment", owner_id)
    runner.workloads[key] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm/vllm-openai:v0.27.1",
        status="running",
        health_status="unhealthy",
        health_failing_streak=1,
        created_at=datetime.now(UTC),
        gpu_ids=[0],
        service_type="generate",
        endpoint=f"http://openllmops-inference-{owner_id}:8000",
        port=8000,
        generation=1,
    )

    def status_command() -> AgentCommand:
        return AgentCommand(
            action="status",
            owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )

    waiting = processor.execute(status_command())
    runner.workloads[key] = runner.workloads[key].model_copy(update={"health_status": "healthy"})
    ready = processor.execute(status_command())

    assert waiting.response.observed_state == AgentWorkloadState.STARTING
    assert "健康检查未通过" in (waiting.response.message or "")
    assert ready.response.observed_state == AgentWorkloadState.RUNNING
    assert len(runner.inference_requests) == 0


@pytest.mark.parametrize(
    ("health_status", "health_failing_streak", "message"),
    [
        ("starting", 0, "缺少可信启动时间"),
        ("unhealthy", 0, "缺少可信连续失败计数"),
    ],
)
def test_malformed_inference_probe_state_is_stopped_instead_of_waiting_forever(
    tmp_path: Path,
    health_status: str,
    health_failing_streak: int,
    message: str,
) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    runner.workloads[("deployment", owner_id)] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm/vllm-openai:v0.27.1",
        status="running",
        health_status=health_status,  # type: ignore[arg-type]
        health_failing_streak=health_failing_streak,
        gpu_ids=[0],
        generation=1,
    )
    command = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )

    result = processor.execute(command)

    assert runner.quiesce_calls == [(owner_id, 1, 30)]
    assert result.response.observed_state == AgentWorkloadState.FAILED
    assert message in (result.response.message or "")


@pytest.mark.parametrize(
    ("status", "health_status", "failing_streak", "restart_count", "age_seconds", "message"),
    [
        ("running", "unhealthy", 3, 0, 5, "连续健康检查失败"),
        ("running", "starting", 0, 0, 61, "启动超过"),
        ("restarting", "starting", 0, 1, 61, "启动超过"),
        ("restarting", "starting", 0, 3, 1, "重启次数"),
    ],
)
def test_bounded_inference_failure_stops_before_reporting_failed(
    tmp_path: Path,
    status: str,
    health_status: str,
    failing_streak: int,
    restart_count: int,
    age_seconds: int,
    message: str,
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    settings = settings_for(
        tmp_path,
        inference_startup_timeout_seconds=60,
        inference_unhealthy_timeout_seconds=15,
        inference_failure_stop_timeout_seconds=7,
    )
    runner = FakeRunner()
    processor = CommandProcessor(settings, runner, clock=lambda: now)  # type: ignore[arg-type]
    owner_id = uuid4()
    runner.workloads[("deployment", owner_id)] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm/vllm-openai:v0.27.1",
        status=status,
        health_status=health_status,  # type: ignore[arg-type]
        health_failing_streak=failing_streak,
        restart_count=restart_count,
        started_at=now - timedelta(seconds=age_seconds),
        finished_at=now - timedelta(seconds=age_seconds),
        gpu_ids=[0],
        service_type="generate",
        endpoint=f"http://openllmops-inference-{owner_id}:8000",
        port=8000,
        generation=2,
    )
    command = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=2),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )

    result = processor.execute(command)

    assert runner.quiesce_calls == [(owner_id, 2, 7)]
    assert result.response.observed_state == AgentWorkloadState.FAILED
    assert message in (result.response.message or "")
    assert "endpoint" not in result.response.metadata


def test_failed_inference_stop_uncertainty_keeps_starting_and_lease_semantics(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    settings = settings_for(
        tmp_path,
        inference_startup_timeout_seconds=60,
    )
    runner = FakeRunner()
    runner.quiesce_result = False
    processor = CommandProcessor(settings, runner, clock=lambda: now)  # type: ignore[arg-type]
    owner_id = uuid4()
    runner.workloads[("deployment", owner_id)] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm/vllm-openai:v0.27.1",
        status="running",
        health_status="starting",
        started_at=now - timedelta(seconds=61),
        gpu_ids=[0],
        generation=1,
    )
    command = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )

    result = processor.execute(command)

    assert runner.quiesce_calls == [(owner_id, 1, 30)]
    assert result.response.observed_state == AgentWorkloadState.STARTING
    assert "停止结果尚未确认" in (result.response.message or "")


def test_quiesced_failed_inference_can_be_cleaned_up_by_exact_generation(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    runner = FakeRunner()
    processor = CommandProcessor(
        settings_for(tmp_path, inference_startup_timeout_seconds=60),
        runner,
        clock=lambda: now,
    )  # type: ignore[arg-type]
    owner_id = uuid4()
    key = ("deployment", owner_id)
    runner.workloads[key] = WorkloadInfo(
        name=f"openllmops-inference-{owner_id}",
        workload_id=owner_id,
        kind="inference",
        image="vllm:test",
        status="running",
        health_status="starting",
        started_at=now - timedelta(seconds=61),
        gpu_ids=[0],
        generation=4,
    )
    status = AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=owner_id, name="chat", generation=4),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )

    failed = processor.execute(status)
    cleaned = processor.execute(cleanup_command("deployment", owner_id, generation=4))

    assert failed.response.observed_state == AgentWorkloadState.FAILED
    assert runner.quiesce_calls == [(owner_id, 4, 30)]
    assert cleaned.response.accepted is True
    assert cleaned.response.observed_state == AgentWorkloadState.ABSENT
    assert key not in runner.workloads


def test_request_id_cannot_be_rebound(tmp_path: Path) -> None:
    runner = FakeRunner()
    processor = CommandProcessor(settings_for(tmp_path), runner)  # type: ignore[arg-type]
    owner_id = uuid4()
    request_id = uuid4()
    processor.execute(deployment_command(owner_id, request_id=request_id))

    rebound = deployment_command(owner_id, request_id=request_id, max_model_len=16384)
    result = processor.execute(rebound)
    assert result.status_code == 409
    assert result.response.error_code == "request_id_reused"
    assert len(runner.inference_requests) == 1


def test_start_binding_and_generation_watermark_persist_atomically(
    tmp_path: Path,
) -> None:
    owner_id = uuid4()
    store = CommandStateStore(tmp_path)

    store.bind_start("deployment", owner_id, 7, "fingerprint")

    assert store.generation("deployment", owner_id) == 7
    restarted = CommandStateStore(tmp_path)
    assert restarted.generation("deployment", owner_id) == 7
    # 同代同参数可以安全恢复，不会产生第二套绑定。
    restarted.bind_start("deployment", owner_id, 7, "fingerprint")


def test_training_execution_materializes_controlled_config_and_dataset_info(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    model_path = settings.model_root / "demo"
    model_path.mkdir()
    dataset_path = settings.dataset_root / "sft.jsonl"
    dataset_path.write_text(
        '{"instruction":"问候","input":"","output":"你好"}\n',
        encoding="utf-8",
    )
    job_id = uuid4()
    output_path = settings.checkpoint_root / str(job_id)
    runner = FakeRunner()
    processor = CommandProcessor(settings, runner)  # type: ignore[arg-type]
    command = AgentCommand(
        action="start",
        owner=AgentOwner(type="training", id=job_id, name="sft", generation=3),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={
            "runner": "llamafactory",
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "stage": "sft",
            "algorithm": "qlora",
            "training_config": {"num_train_epochs": 1.0, "template": "qwen"},
            "output_dir": str(output_path),
        },
    )

    result = processor.execute(command)

    assert result.response.accepted
    request = runner.training_requests[0]
    config = json.loads(request.config_path.read_text(encoding="utf-8"))
    dataset_info = json.loads((request.dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert config["stage"] == "sft"
    assert config["finetuning_type"] == "lora"
    assert config["quantization_bit"] == 4
    assert config["trust_remote_code"] is False
    assert dataset_info["openllmops_dataset"]["file_name"] == "/workspace/data/training.jsonl"
    assert dataset_info["openllmops_dataset"]["columns"]["query"] == "input"
    assert request.stage == "sft"
    assert request.algorithm == "qlora"
    assert request.dataset_format == "alpaca"


def test_training_execution_rejects_unknown_parameter_and_non_uuid_output(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    model_path = settings.model_root / "demo"
    model_path.mkdir()
    dataset_path = settings.dataset_root / "sft.jsonl"
    dataset_path.write_text('{"instruction":"hi","output":"hello"}\n', encoding="utf-8")
    runner = FakeRunner()
    processor = CommandProcessor(settings, runner)  # type: ignore[arg-type]

    def command(job_id: UUID, training_config: dict, output_dir: Path) -> AgentCommand:
        return AgentCommand(
            action="start",
            owner=AgentOwner(type="training", id=job_id, name="sft", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={
                "runner": "llamafactory",
                "model_path": str(model_path),
                "dataset_path": str(dataset_path),
                "stage": "sft",
                "algorithm": "lora",
                "training_config": training_config,
                "output_dir": str(output_dir),
            },
        )

    unknown_job = uuid4()
    unknown = processor.execute(
        command(
            unknown_job,
            {"template": "qwen", "trust_remote_code": True},
            settings.checkpoint_root / str(unknown_job),
        )
    )
    assert unknown.status_code == 422
    assert unknown.response.error_code == "invalid_execution"

    wrong_output_job = uuid4()
    wrong_output = processor.execute(
        command(
            wrong_output_job,
            {"template": "qwen"},
            settings.checkpoint_root / "operator-selected-name",
        )
    )
    assert wrong_output.status_code == 422
    assert wrong_output.response.error_code == "invalid_workload"
    assert runner.training_requests == []


def test_alpaca_dataset_without_input_does_not_declare_query(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(
        '{"instruction":"问候","output":"你好"}\n',
        encoding="utf-8",
    )

    info, dataset_format = CommandProcessor._dataset_info(dataset_path, "sft")

    assert info["columns"] == {"prompt": "instruction", "response": "output"}
    assert dataset_format.value == "alpaca"


def test_training_dataset_probe_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(
        '{"instruction":"hi","instruction":"changed","output":"hello"}\n',
        encoding="utf-8",
    )

    with pytest.raises(InvalidWorkload, match="重复字段"):
        CommandProcessor._dataset_info(dataset_path, "sft")


def test_training_stop_preserves_naturally_succeeded_container_and_is_repeatable(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    runner = FakeRunner()
    job_id = uuid4()
    runner.workloads[("training", job_id)] = WorkloadInfo(
        name=f"openllmops-training-{job_id}",
        workload_id=job_id,
        kind="training",
        image="safe-training-image",
        status="exited",
        exit_code=0,
        gpu_ids=[0],
        generation=2,
    )
    processor = CommandProcessor(settings, runner)  # type: ignore[arg-type]

    def stop_command() -> AgentCommand:
        return AgentCommand(
            action="stop",
            owner=AgentOwner(type="training", id=job_id, name="sft", generation=2),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )

    first = processor.execute(stop_command())
    repeated = processor.execute(stop_command())

    assert first.response.observed_state == AgentWorkloadState.SUCCEEDED
    assert repeated.response.observed_state == AgentWorkloadState.SUCCEEDED
    assert first.response.metadata["progress"] == 25.0
    assert ("training", job_id) in runner.workloads
    assert runner.stop_calls == 2
    assert runner.training_metadata_completed_calls == [True, True]


def test_training_stop_returns_absent_after_real_cancellation(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    runner = FakeRunner()
    job_id = uuid4()
    runner.workloads[("training", job_id)] = WorkloadInfo(
        name=f"openllmops-training-{job_id}",
        workload_id=job_id,
        kind="training",
        image="safe-training-image",
        status="running",
        gpu_ids=[0],
        generation=1,
    )
    processor = CommandProcessor(settings, runner)  # type: ignore[arg-type]

    result = processor.execute(
        AgentCommand(
            action="stop",
            owner=AgentOwner(type="training", id=job_id, name="sft", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )
    )

    assert result.response.observed_state == AgentWorkloadState.ABSENT
    assert ("training", job_id) not in runner.workloads
    assert runner.training_metadata_completed_calls == []


def test_evaluation_start_status_and_stop_use_real_runner(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    base_model = settings.model_root / "base"
    candidate_model = settings.model_root / "candidate"
    base_model.mkdir()
    candidate_model.mkdir()
    dataset = settings.dataset_root / "domain.jsonl"
    dataset.write_text(
        '{"id":"q1","question":"1+1?","choices":{"A":"1","B":"2"},"answer":"B"}\n',
        encoding="utf-8",
    )
    runner = FakeRunner()
    processor = CommandProcessor(settings, runner)  # type: ignore[arg-type]
    run_id = uuid4()
    command = AgentCommand(
        action="start",
        owner=AgentOwner(type="evaluation", id=run_id, name="eval", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={
            "runner": "evaluation",
            "base_model_path": str(base_model),
            "candidate_model_path": str(candidate_model),
            "base_template": "base",
            "candidate_template": "instruct",
            "datasets": [{"name": "domain", "path": str(dataset)}],
            "output_dir": str(settings.evaluation_output_root / str(run_id)),
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "concurrency": 2,
            "max_tokens": 64,
        },
    )

    result = processor.execute(command)

    assert result.status_code == 200
    assert result.response.observed_state == AgentWorkloadState.RUNNING
    request = runner.evaluation_requests[0]
    assert request.dataset_path.read_text(encoding="utf-8").startswith(
        '{"answer":"B","category":"domain/default","choices"'
    )
    assert request.dataset_manifest_path.is_file()
    assert request.output_path == settings.evaluation_output_root / str(run_id)

    runner.workloads[("evaluation", run_id)] = runner.workloads[("evaluation", run_id)].model_copy(
        update={"status": "exited", "exit_code": 0}
    )
    status = processor.execute(
        AgentCommand(
            action="status",
            owner=AgentOwner(type="evaluation", id=run_id, name="eval", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )
    )
    assert status.response.observed_state == AgentWorkloadState.SUCCEEDED
    assert status.response.metadata["comparison"]["percentage_point_change"] == 25.0

    stopped = processor.execute(
        AgentCommand(
            action="stop",
            owner=AgentOwner(type="evaluation", id=run_id, name="eval", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )
    )
    assert stopped.response.observed_state == AgentWorkloadState.SUCCEEDED
    assert stopped.response.metadata["result_path"].endswith("pair-report.json")
    assert runner.stop_calls == 1

    repeated = processor.execute(
        AgentCommand(
            action="stop",
            owner=AgentOwner(type="evaluation", id=run_id, name="eval", generation=1),
            resources=AgentResourceRequest(gpu_ids=[0]),
            execution={},
        )
    )
    assert repeated.response.observed_state == AgentWorkloadState.SUCCEEDED
    assert repeated.response.metadata["result_path"].endswith("pair-report.json")
    assert runner.stop_calls == 2
