import json
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


def settings_for(tmp_path: Path) -> Settings:
    settings = Settings(
        node_agent_token="a" * 32,
        model_root=tmp_path / "models",
        dataset_root=tmp_path / "datasets",
        evaluation_dataset_root=tmp_path / "evaluation-datasets",
        evaluation_output_root=tmp_path / "evaluation-output",
        checkpoint_root=tmp_path / "checkpoints",
        training_config_root=tmp_path / "configs",
        runtime_root=tmp_path / "runtime",
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
