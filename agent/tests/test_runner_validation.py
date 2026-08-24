from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from openllmops_agent.config import Settings
from openllmops_agent.docker_runner import (
    GENERATION_LABEL,
    GPU_LABEL,
    ID_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    DockerRunner,
    InvalidWorkload,
)
from openllmops_agent.evaluation_image_policy import EXPECTED_EVALUATION_LABELS
from openllmops_agent.image_policy import EXPECTED_SECURITY_LABELS
from openllmops_agent.schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
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


def test_cpt_rejects_freeze(runner: DockerRunner, tmp_path: Path) -> None:
    model_path = runner.settings.model_root / "demo"
    dataset_path = runner.settings.dataset_root / "cpt"
    output_path = runner.settings.checkpoint_root / "job"
    for path in (model_path, dataset_path, output_path):
        path.mkdir(parents=True)
    config_path = runner.settings.training_config_root / "job.yaml"
    config_path.write_text(
        "\n".join(
            (
                "stage: pt",
                "finetuning_type: freeze",
                f"model_name_or_path: {model_path}",
                f"dataset_dir: {dataset_path}",
                f"output_dir: {output_path}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidWorkload, match="继续预训练固定使用 LoRA"):
        runner._validate_training_config(config_path, model_path, dataset_path, output_path)


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
