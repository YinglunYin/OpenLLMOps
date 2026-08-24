import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from docker.errors import APIError, NotFound
from openllmops_training_config import (
    Algorithm,
    DatasetFormat,
    Stage,
    TrainingRequest,
    build_training_config,
)
from openllmops_training_runtime import (
    WORKSPACE_CONFIG,
    WORKSPACE_DATA_FILE,
    WORKSPACE_DATASET,
    WORKSPACE_MODEL,
    WORKSPACE_OUTPUT,
)

from openllmops_agent.config import Settings
from openllmops_agent.docker_runner import (
    GENERATION_LABEL,
    GPU_LABEL,
    ID_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    TRAINING_ALGORITHM_LABEL,
    DockerRunner,
    InvalidWorkload,
    WorkloadConflict,
)
from openllmops_agent.evaluation_image_policy import EXPECTED_EVALUATION_LABELS
from openllmops_agent.image_policy import EXPECTED_SECURITY_LABELS
from openllmops_agent.schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
    TrainingLaunchRequest,
    WorkloadInfo,
)


@pytest.fixture
def runner(tmp_path: Path) -> DockerRunner:
    settings = Settings(
        node_agent_token="a" * 32,
        model_root=tmp_path / "models",
        dataset_root=tmp_path / "datasets",
        evaluation_dataset_root=tmp_path / "evaluation-datasets",
        evaluation_output_root=tmp_path / "evaluation-output",
        checkpoint_root=tmp_path / "checkpoints",
        training_config_root=tmp_path / "configs",
        runtime_root=tmp_path / "runtime",
        enforce_nvml_process_check=False,
    )
    settings.ensure_layout()
    instance = DockerRunner(settings, client=MagicMock())
    yield instance
    instance.close()


def test_vllm_command_never_accepts_trust_remote_code(runner: DockerRunner) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="demo",
        vllm_args={"trust_remote_code": True},
    )

    with pytest.raises(InvalidWorkload, match="不允许的 vLLM 参数"):
        runner._vllm_command(request, [0])


def test_vllm_command_owns_tensor_parallel_size(runner: DockerRunner) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0, 1],
        model_path="/unused",
        served_model_name="demo",
        vllm_args={"max_model_len": 8192},
    )

    command = runner._vllm_command(request, [0, 1])

    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--load-format") + 1] == "safetensors"
    assert "--trust-remote-code" not in command


def test_vllm_rejects_unsafe_gpu_memory_ratio(runner: DockerRunner) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="demo",
        vllm_args={"gpu_memory_utilization": 1.0},
    )

    with pytest.raises(InvalidWorkload, match=r"0\.1\.\.0\.98"):
        runner._vllm_command(request, [0])


def test_embedding_deployment_selects_embed_task(runner: DockerRunner) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="embedding-demo",
        service_type="embedding",
    )

    command = runner._vllm_command(request, [0])

    assert command[command.index("--runner") + 1] == "pooling"
    assert command[command.index("--convert") + 1] == "embed"


def test_vllm_027_command_keeps_supported_model_and_safetensors_flags(
    runner: DockerRunner,
) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="demo",
    )

    command = runner._vllm_command(request, [0])

    assert command[command.index("--model") + 1] == "/workspace/model"
    assert command[command.index("--runner") + 1] == "generate"
    assert command[command.index("--load-format") + 1] == "safetensors"


def test_inference_installs_local_vllm_readiness_and_liveness_healthcheck(
    runner: DockerRunner,
) -> None:
    deployment_id = uuid4()
    model_path = runner.settings.model_root / "inference-demo"
    model_path.mkdir()
    image = MagicMock()
    image.id = "sha256:" + "b" * 64
    runner.client.images.get.return_value = image
    runner.client.containers.run.return_value = MagicMock()
    expected = WorkloadInfo(
        name=f"openllmops-inference-{deployment_id}",
        workload_id=deployment_id,
        kind="inference",
        image=image.id,
        status="running",
        health_status="starting",
        gpu_ids=[0],
        service_type="generate",
        endpoint=f"http://openllmops-inference-{deployment_id}:8123",
        port=8123,
    )
    request = InferenceLaunchRequest(
        deployment_id=deployment_id,
        image=runner.settings.vllm_runtime_image,
        gpu_ids=[0],
        model_path=model_path,
        served_model_name="demo",
        port=8123,
    )

    with (
        patch.object(runner, "_assert_name_available"),
        patch.object(runner, "_assert_gpus_available"),
        patch.object(runner, "_to_info", return_value=expected),
    ):
        assert runner.launch_inference(request) == expected

    arguments = runner.client.containers.run.call_args.kwargs
    healthcheck = arguments["healthcheck"]
    assert healthcheck["test"][:2] == ["CMD", "python"]
    assert healthcheck["test"][2] == "-c"
    assert "http://127.0.0.1:8123/health" in healthcheck["test"][3]
    assert healthcheck["interval"] == 5 * 1_000_000_000
    assert healthcheck["start_period"] == (runner.settings.inference_startup_timeout_seconds * 1_000_000_000)
    assert healthcheck["retries"] == 3
    assert arguments["restart_policy"] == {"Name": "on-failure", "MaximumRetryCount": 3}
    assert arguments["log_config"] == {
        "type": "local",
        "config": {"max-size": "20m", "max-file": "5"},
    }


@pytest.mark.parametrize(
    ("raw_health", "expected"),
    [
        ({"Status": "starting"}, "starting"),
        ({"Status": "healthy"}, "healthy"),
        ({"Status": "unhealthy"}, "unhealthy"),
        ({"Status": "unexpected"}, None),
        (None, None),
    ],
)
def test_workload_info_reads_only_known_docker_health_states(
    runner: DockerRunner,
    raw_health: dict[str, str] | None,
    expected: str | None,
) -> None:
    deployment_id = uuid4()
    container = MagicMock()
    container.name = f"openllmops-inference-{deployment_id}"
    container.status = "running"
    container.labels = {
        MANAGED_LABEL: "true",
        ID_LABEL: str(deployment_id),
        KIND_LABEL: "inference",
        GPU_LABEL: "0",
        GENERATION_LABEL: "1",
    }
    container.image.tags = [runner.settings.vllm_runtime_image]
    state: dict[str, object] = {
        "ExitCode": 0,
        "StartedAt": "2026-08-25T01:02:03.123456Z",
        "FinishedAt": "2026-08-25T01:03:04.123456Z",
    }
    if raw_health is not None:
        state["Health"] = {**raw_health, "FailingStreak": 7}
    container.attrs = {
        "Created": "2026-08-25T01:00:00Z",
        "RestartCount": 2,
        "State": state,
    }

    info = runner._to_info(container)

    assert info.health_status == expected
    assert info.restart_count == 2
    assert info.health_failing_streak == (7 if raw_health is not None else 0)
    assert info.started_at is not None and info.finished_at is not None and info.created_at is not None


def _inference_container(runner: DockerRunner, *, status: str = "running") -> MagicMock:
    workload_id = uuid4()
    container = MagicMock()
    container.name = f"openllmops-inference-{workload_id}"
    container.status = status
    container.labels = {
        MANAGED_LABEL: "true",
        ID_LABEL: str(workload_id),
        KIND_LABEL: "inference",
        GPU_LABEL: "0",
        GENERATION_LABEL: "4",
    }
    container.image.tags = [runner.settings.vllm_runtime_image]
    container.attrs = {"State": {"ExitCode": 0}}
    runner.client.containers.get.return_value = container
    return container


def test_quiesce_failed_inference_stops_then_confirms_non_active(
    runner: DockerRunner,
) -> None:
    container = _inference_container(runner)

    def stopped(*, timeout: int) -> None:
        assert timeout == 9
        container.status = "exited"
        container.attrs = {"State": {"ExitCode": 137}}

    container.stop.side_effect = stopped

    confirmed = runner.quiesce_failed_inference(
        UUID(container.labels[ID_LABEL]),
        4,
        timeout_seconds=9,
    )

    assert confirmed is True
    container.stop.assert_called_once_with(timeout=9)
    assert container.reload.call_count >= 2
    container.remove.assert_not_called()


def test_quiesce_failed_inference_keeps_uncertain_active_container(
    runner: DockerRunner,
) -> None:
    container = _inference_container(runner)
    container.stop.side_effect = APIError("stop response lost")

    confirmed = runner.quiesce_failed_inference(
        UUID(container.labels[ID_LABEL]),
        4,
        timeout_seconds=9,
    )

    assert confirmed is False
    assert container.status == "running"
    container.remove.assert_not_called()


@pytest.mark.parametrize(
    "removed_argument",
    ["disable_log_requests", "guided_decoding_backend", "rope_scaling", "swap_space"],
)
def test_vllm_027_removed_arguments_are_rejected(
    runner: DockerRunner,
    removed_argument: str,
) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="demo",
        vllm_args={removed_argument: "unused"},
    )

    with pytest.raises(InvalidWorkload, match="不允许的 vLLM 参数"):
        runner._vllm_command(request, [0])


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("block_size", 64),
        ("max_model_len", 131_073),
        ("max_num_batched_tokens", 65_537),
        ("max_num_seqs", 1_025),
        ("max_logprobs", 101),
        ("cpu_offload_gb", 16.1),
    ],
)
def test_vllm_resource_amplification_arguments_are_bounded(
    runner: DockerRunner,
    argument: str,
    value: int | float,
) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.27.1",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="demo",
        vllm_args={argument: value},
    )

    with pytest.raises(InvalidWorkload, match=r"安全上限|仅允许"):
        runner._vllm_command(request, [0])


def test_training_uses_fixed_wrapper_paths_offline_network_and_world_size(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    model_path = runner.settings.model_root / "demo"
    model_path.mkdir()
    dataset_path = runner.settings.dataset_root / "sft.jsonl"
    dataset_path.write_text('{"instruction":"hi","output":"hello"}\n', encoding="utf-8")
    dataset_dir = runner.settings.runtime_root / "contract" / "training" / str(job_id) / "1"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    output_path = runner.settings.checkpoint_root / str(job_id)
    config_path = runner.settings.training_config_root / f"{job_id}.json"
    config_path.write_text(
        json.dumps(
            build_training_config(
                TrainingRequest(
                    stage=Stage.SFT,
                    algorithm=Algorithm.LORA,
                    model_path=WORKSPACE_MODEL,
                    dataset_dir=WORKSPACE_DATASET,
                    output_dir=WORKSPACE_OUTPUT,
                    dataset_format=DatasetFormat.ALPACA,
                    template="qwen",
                )
            )
        ),
        encoding="utf-8",
    )
    image = MagicMock()
    image.labels = EXPECTED_SECURITY_LABELS
    image.id = "sha256:" + "c" * 64
    runner.client.images.get.return_value = image
    runner.client.containers.run.return_value = MagicMock()
    expected = WorkloadInfo(
        name=f"openllmops-training-{job_id}",
        workload_id=job_id,
        kind="training",
        image=image.id,
        status="running",
        gpu_ids=[0, 1],
    )
    request = TrainingLaunchRequest(
        job_id=job_id,
        image=runner.settings.llamafactory_runtime_image,
        gpu_ids=[0, 1],
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_dir=dataset_dir,
        config_path=config_path,
        output_path=output_path,
        stage="sft",
        algorithm="lora",
        dataset_format="alpaca",
    )

    with (
        patch.object(runner, "_assert_name_available"),
        patch.object(runner, "_assert_gpus_available"),
        patch.object(runner, "_to_info", return_value=expected),
    ):
        assert runner.launch_training(request) == expected

    arguments = runner.client.containers.run.call_args.kwargs
    assert arguments["command"][0] == "run"
    assert arguments["command"][arguments["command"].index("--config") + 1] == str(WORKSPACE_CONFIG)
    assert arguments["volumes"][str(model_path)]["bind"] == str(WORKSPACE_MODEL)
    assert arguments["volumes"][str(dataset_path)]["bind"] == str(WORKSPACE_DATA_FILE)
    assert arguments["volumes"][str(dataset_dir)]["bind"] == str(WORKSPACE_DATASET)
    assert arguments["volumes"][str(output_path)]["bind"] == str(WORKSPACE_OUTPUT)
    assert arguments["environment"]["FORCE_TORCHRUN"] == "1"
    assert arguments["environment"]["NPROC_PER_NODE"] == "2"
    assert arguments["environment"]["NCCL_P2P_DISABLE"] == "1"
    assert arguments["network_mode"] == "none"
    assert arguments["labels"][TRAINING_ALGORITHM_LABEL] == "lora"
    assert arguments["log_config"] == {
        "type": "local",
        "config": {"max-size": "20m", "max-file": "5"},
    }


def test_training_image_is_resolved_to_verified_immutable_id(
    runner: DockerRunner,
) -> None:
    image = MagicMock()
    image.labels = EXPECTED_SECURITY_LABELS
    image.id = "sha256:" + "c" * 64
    runner.client.images.get.return_value = image

    assert runner._verified_training_image_id("openllmops/llamafactory-secure:test") == image.id


def test_vllm_image_is_resolved_to_verified_immutable_id(runner: DockerRunner) -> None:
    image = MagicMock()
    image.id = "sha256:" + "b" * 64
    runner.client.images.get.return_value = image

    assert runner._verified_vllm_image_id("vllm/vllm-openai:v0.27.1") == image.id


def test_training_image_rejects_forged_security_labels(runner: DockerRunner) -> None:
    image = MagicMock()
    image.labels = {"com.openllmops.security.ghsa-mwc7-mf87-v3mf": "mitigated"}
    image.id = "sha256:" + "d" * 64
    runner.client.images.get.return_value = image

    with pytest.raises(InvalidWorkload, match="安全构建标签"):
        runner._verified_training_image_id("registry/secure@sha256:" + "d" * 64)


def _full_training_model(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe")


def _training_container(runner: DockerRunner, job_id, output: Path, algorithm: str) -> MagicMock:
    container = MagicMock()
    container.name = f"openllmops-training-{job_id}"
    container.labels = {
        MANAGED_LABEL: "true",
        ID_LABEL: str(job_id),
        KIND_LABEL: "training",
        GPU_LABEL: "0",
        GENERATION_LABEL: "1",
        "com.openllmops.output-path": str(output),
        TRAINING_ALGORITHM_LABEL: algorithm,
    }
    runner.client.containers.get.return_value = container
    return container


def test_completed_freeze_reports_validated_output_as_deployable_path(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    output = runner.settings.checkpoint_root / str(job_id)
    _full_training_model(output)
    _training_container(runner, job_id, output, "freeze")

    metadata = runner.training_metadata(job_id, completed=True)

    assert metadata["merged_model_path"] == str(output.resolve())
    assert "adapter_path" not in metadata


def test_completed_lora_requires_adapter_and_merged_safetensors(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    output = runner.settings.checkpoint_root / str(job_id)
    output.mkdir()
    (output / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (output / "adapter_model.safetensors").write_bytes(b"safe adapter")
    _full_training_model(output / "merged")
    _training_container(runner, job_id, output, "lora")

    metadata = runner.training_metadata(job_id, completed=True)

    assert metadata["adapter_path"] == str(output.resolve())
    assert metadata["merged_model_path"] == str((output / "merged").resolve())

    (output / "merged" / "model.safetensors").unlink()
    with pytest.raises(InvalidWorkload, match="产物校验失败"):
        runner.training_metadata(job_id, completed=True)


def test_transient_trainer_state_is_ignored_while_running_but_strict_on_success(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    output = runner.settings.checkpoint_root / str(job_id)
    output.mkdir()
    (output / "trainer_state.json").write_text('{"global_step":', encoding="utf-8")
    _training_container(runner, job_id, output, "lora")

    assert runner.training_metadata(job_id, completed=False) == {}
    with pytest.raises(InvalidWorkload, match="训练状态"):
        runner.training_metadata(job_id, completed=True)


def test_running_progress_falls_back_to_latest_numeric_checkpoint(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    output = runner.settings.checkpoint_root / str(job_id)
    checkpoint = output / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 20,
                "max_steps": 40,
                "log_history": [{"loss": 1.25, "step": 20}],
            }
        ),
        encoding="utf-8",
    )
    _training_container(runner, job_id, output, "lora")

    metadata = runner.training_metadata(job_id, completed=False)

    assert metadata["current_step"] == 20
    assert metadata["total_steps"] == 40
    assert metadata["progress"] == 50.0
    assert metadata["metrics"]["loss"] == 1.25
    assert "checkpoint_path" not in metadata


def test_failed_or_running_training_never_reports_partial_artifact_paths(
    runner: DockerRunner,
) -> None:
    job_id = uuid4()
    output = runner.settings.checkpoint_root / str(job_id)
    output.mkdir()
    (output / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (output / "adapter_model.safetensors").write_bytes(b"safe adapter")
    _full_training_model(output / "merged")
    _training_container(runner, job_id, output, "lora")

    assert runner.training_metadata(job_id, completed=False) == {}


def test_evaluation_launch_uses_verified_image_fixed_command_and_no_network(
    runner: DockerRunner,
) -> None:
    run_id = uuid4()
    baseline = runner.settings.model_root / "baseline"
    candidate = runner.settings.model_root / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    workspace = runner.settings.runtime_root / "contract" / "evaluation" / str(run_id) / "1"
    workspace.mkdir(parents=True)
    dataset = workspace / "evaluation.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    manifest = workspace / "dataset-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output = runner.settings.evaluation_output_root / str(run_id)
    output.mkdir()
    image = MagicMock()
    image.labels = EXPECTED_EVALUATION_LABELS
    image.id = "sha256:" + "e" * 64
    runner.client.images.get.return_value = image
    container = MagicMock()
    runner.client.containers.run.return_value = container
    expected = WorkloadInfo(
        name=f"openllmops-evaluation-{run_id}",
        workload_id=run_id,
        kind="evaluation",
        image=image.id,
        status="running",
        gpu_ids=[0],
    )
    request = EvaluationLaunchRequest(
        run_id=run_id,
        image=runner.settings.evaluation_runtime_image,
        gpu_ids=[0],
        baseline_model_path=baseline,
        candidate_model_path=candidate,
        dataset_path=dataset,
        dataset_manifest_path=manifest,
        output_path=output,
        base_template="base",
        candidate_template="instruct",
        tensor_parallel_size=1,
        concurrency=2,
        max_tokens=64,
    )

    with (
        patch.object(runner, "_assert_name_available"),
        patch.object(runner, "_assert_gpus_available"),
        patch.object(runner, "_to_info", return_value=expected),
    ):
        assert runner.launch_evaluation(request) == expected

    arguments = runner.client.containers.run.call_args.kwargs
    assert arguments["image"] == image.id
    assert arguments["command"][0] == "run-pair"
    assert arguments["command"][arguments["command"].index("--baseline-path") + 1] == (
        "/workspace/models/baseline"
    )
    assert arguments["network_mode"] == "none"
    assert "network" not in arguments
    assert arguments["read_only"] is True
    assert arguments["security_opt"] == ["no-new-privileges:true"]
    assert arguments["volumes"][str(dataset.parent)]["mode"] == "ro"
    assert arguments["volumes"][str(output)]["mode"] == "rw"
    assert arguments["log_config"] == {
        "type": "local",
        "config": {"max-size": "20m", "max-file": "5"},
    }


def test_stop_reloads_once_before_and_once_after_transition(runner: DockerRunner) -> None:
    workload_id = uuid4()
    container = MagicMock()
    container.name = f"openllmops-evaluation-{workload_id}"
    container.status = "running"
    container.labels = {
        MANAGED_LABEL: "true",
        ID_LABEL: str(workload_id),
        KIND_LABEL: "evaluation",
        GPU_LABEL: "0",
        GENERATION_LABEL: "1",
    }
    container.image.tags = [runner.settings.evaluation_runtime_image]
    container.attrs = {"State": {"ExitCode": 0}}

    def mark_stopped(*, timeout: int) -> None:
        assert timeout == 30
        container.status = "exited"

    container.stop.side_effect = mark_stopped
    runner.client.containers.get.return_value = container

    info = runner.stop(container.name, 30)

    assert info.status == "exited"
    assert container.reload.call_count == 2
    container.stop.assert_called_once_with(timeout=30)


def _contract_terminal_container(
    runner: DockerRunner,
    workload_id,
    *,
    kind: str = "training",
    status: str,
    exit_code: int,
) -> MagicMock:
    container = MagicMock()
    container.name = f"openllmops-{kind}-{workload_id}"
    container.status = status
    container.labels = {
        MANAGED_LABEL: "true",
        ID_LABEL: str(workload_id),
        KIND_LABEL: kind,
        GPU_LABEL: "0",
        GENERATION_LABEL: "1",
        "com.openllmops.owner-type": "evaluation" if kind == "evaluation" else "training",
    }
    container.attrs = {"State": {"ExitCode": exit_code}}
    container.image.tags = [runner.settings.llamafactory_runtime_image]
    runner.client.containers.get.return_value = container
    return container


@pytest.mark.parametrize(
    ("kind", "owner_type", "status", "exit_code"),
    [
        ("training", "training", "exited", 0),
        ("training", "training", "exited", 1),
        ("evaluation", "evaluation", "dead", 137),
        ("inference", "deployment", "exited", 1),
    ],
)
def test_contract_cleanup_removes_only_confirmed_terminal_container_and_keeps_bind_output(
    runner: DockerRunner,
    kind: str,
    owner_type: str,
    status: str,
    exit_code: int,
) -> None:
    workload_id = uuid4()
    output = runner.settings.runtime_root / "preserved" / str(workload_id)
    output.mkdir(parents=True)
    artifact = output / "artifact.bin"
    artifact.write_bytes(b"preserve me")
    container = _contract_terminal_container(
        runner,
        workload_id,
        kind=kind,
        status=status,
        exit_code=exit_code,
    )
    container.labels["com.openllmops.owner-type"] = owner_type

    runner.cleanup_contract_workload(owner_type, workload_id, generation=1)

    container.remove.assert_called_once_with(force=False, v=False)
    assert artifact.read_bytes() == b"preserve me"


def test_contract_cleanup_is_idempotent_when_container_is_absent(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    runner.client.containers.get.side_effect = NotFound("gone")

    runner.cleanup_contract_workload("training", workload_id, generation=4)
    runner.cleanup_contract_workload("training", workload_id, generation=4)

    assert runner.client.containers.get.call_count == 2


@pytest.mark.parametrize("status", ["created", "running", "restarting", "paused", "removing", "unknown"])
def test_contract_cleanup_rejects_nonterminal_or_uncertain_status_without_removing(
    runner: DockerRunner,
    status: str,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status=status,
        exit_code=0,
    )

    with pytest.raises(WorkloadConflict, match="cleanup"):
        runner.cleanup_contract_workload("training", workload_id, generation=1)

    container.remove.assert_not_called()


def test_contract_cleanup_rejects_generation_mismatch_without_removing(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="exited",
        exit_code=0,
    )

    with pytest.raises(WorkloadConflict, match="generation"):
        runner.cleanup_contract_workload("training", workload_id, generation=2)

    container.remove.assert_not_called()


@pytest.mark.parametrize(
    ("label", "value", "message"),
    [
        (MANAGED_LABEL, "false", "owner"),
        (KIND_LABEL, "evaluation", "owner"),
        ("com.openllmops.owner-type", "evaluation", "owner"),
        (GENERATION_LABEL, "01", "generation"),
    ],
)
def test_contract_cleanup_rejects_untrusted_identity_labels_without_removing(
    runner: DockerRunner,
    label: str,
    value: str,
    message: str,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="exited",
        exit_code=0,
    )
    container.labels[label] = value

    with pytest.raises(WorkloadConflict, match=message):
        runner.cleanup_contract_workload("training", workload_id, generation=1)

    container.remove.assert_not_called()


def test_contract_cleanup_preserves_container_when_inspect_is_uncertain(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="exited",
        exit_code=0,
    )
    container.reload.side_effect = APIError("inspect unavailable")

    with pytest.raises(APIError, match="inspect unavailable"):
        runner.cleanup_contract_workload("training", workload_id, generation=1)

    container.remove.assert_not_called()


def test_contract_cleanup_does_not_claim_success_when_remove_is_uncertain(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="exited",
        exit_code=1,
    )
    container.remove.side_effect = APIError("remove response unavailable")

    with pytest.raises(APIError, match="remove response unavailable"):
        runner.cleanup_contract_workload("training", workload_id, generation=1)

    container.remove.assert_called_once_with(force=False, v=False)


def test_contract_stop_preserves_preexisting_success_and_repeated_stop(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="exited",
        exit_code=0,
    )

    first = runner.stop_contract_workload("training", workload_id, generation=1)
    repeated = runner.stop_contract_workload("training", workload_id, generation=1)

    assert first is not None and first.status == "exited" and first.exit_code == 0
    assert repeated is not None and repeated.status == "exited" and repeated.exit_code == 0
    container.stop.assert_not_called()
    container.remove.assert_not_called()


def test_contract_stop_removes_active_training_that_exits_nonzero(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="running",
        exit_code=0,
    )

    def cancelled(*, timeout: int) -> None:
        assert timeout == 30
        container.status = "exited"
        container.attrs = {"State": {"ExitCode": 143}}

    container.stop.side_effect = cancelled

    assert runner.stop_contract_workload("training", workload_id, generation=1) is None
    container.remove.assert_called_once_with(force=False, v=True)


def test_contract_stop_window_preserves_training_that_naturally_exits_zero(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        status="running",
        exit_code=0,
    )

    def naturally_completed(*, timeout: int) -> None:
        assert timeout == 30
        container.status = "exited"
        container.attrs = {"State": {"ExitCode": 0}}

    container.stop.side_effect = naturally_completed

    info = runner.stop_contract_workload("training", workload_id, generation=1)

    assert info is not None and info.status == "exited" and info.exit_code == 0
    container.remove.assert_not_called()


def test_contract_stop_preserves_successful_evaluation_container(
    runner: DockerRunner,
) -> None:
    workload_id = uuid4()
    container = _contract_terminal_container(
        runner,
        workload_id,
        kind="evaluation",
        status="exited",
        exit_code=0,
    )

    info = runner.stop_contract_workload("evaluation", workload_id, generation=1)

    assert info is not None and info.status == "exited" and info.exit_code == 0
    container.remove.assert_not_called()
