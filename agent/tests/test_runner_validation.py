from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from openllmops_agent.config import Settings
from openllmops_agent.docker_runner import DockerRunner, InvalidWorkload
from openllmops_agent.image_policy import EXPECTED_SECURITY_LABELS
from openllmops_agent.schemas import InferenceLaunchRequest


@pytest.fixture
def runner(tmp_path: Path) -> DockerRunner:
    settings = Settings(
        node_agent_token="a" * 32,
        model_root=tmp_path / "models",
        dataset_root=tmp_path / "datasets",
        checkpoint_root=tmp_path / "checkpoints",
        training_config_root=tmp_path / "configs",
        runtime_root=tmp_path / "runtime",
    )
    settings.ensure_layout()
    instance = DockerRunner(settings, client=MagicMock())
    yield instance
    instance.close()


def test_vllm_command_never_accepts_trust_remote_code(runner: DockerRunner) -> None:
    request = InferenceLaunchRequest(
        deployment_id=uuid4(),
        image="vllm/vllm-openai:v0.10.2",
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
        image="vllm/vllm-openai:v0.10.2",
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
        image="vllm/vllm-openai:v0.10.2",
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
        image="vllm/vllm-openai:v0.10.2",
        gpu_ids=[0],
        model_path="/unused",
        served_model_name="embedding-demo",
        service_type="embedding",
    )

    command = runner._vllm_command(request, [0])

    assert command[command.index("--runner") + 1] == "pooling"
    assert command[command.index("--convert") + 1] == "embed"


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


def test_training_image_rejects_forged_security_labels(runner: DockerRunner) -> None:
    image = MagicMock()
    image.labels = {"com.openllmops.security.ghsa-mwc7-mf87-v3mf": "mitigated"}
    image.id = "sha256:" + "d" * 64
    runner.client.images.get.return_value = image

    with pytest.raises(InvalidWorkload, match="安全构建标签"):
        runner._verified_training_image_id("registry/secure@sha256:" + "d" * 64)
