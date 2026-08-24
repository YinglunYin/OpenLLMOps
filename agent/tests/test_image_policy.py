import pytest

from openllmops_agent.config import Settings
from openllmops_agent.evaluation_image_policy import (
    EVALUATION_RUNTIME_IMAGE,
    EXPECTED_EVALUATION_LABELS,
    UnsafeEvaluationImage,
    validate_evaluation_image_labels,
    validate_evaluation_image_reference,
)
from openllmops_agent.image_policy import (
    EXPECTED_SECURITY_LABELS,
    HARDENED_LLAMAFACTORY_IMAGE,
    UnsafeTrainingImage,
    validate_hardening_labels,
    validate_training_image_reference,
)
from openllmops_agent.vllm_image_policy import (
    VLLM_AMD64_DIGEST,
    VLLM_CU129_RUNTIME_IMAGE,
    VLLM_RUNTIME_IMAGE,
    UnsafeVLLMImage,
    validate_vllm_image_reference,
)


@pytest.mark.parametrize(
    "reference",
    [
        "hiyouga/llamafactory:0.9.5",
        "hiyouga/llamafactory:latest",
        "docker.io/hiyouga/llamafactory@sha256:" + "a" * 64,
        "registry.internal/llamafactory:latest",
        "registry.internal/llamafactory:0.9.6",
        "registry.internal/unsafe*@sha256:" + "a" * 64,
    ],
)
def test_training_image_policy_rejects_mutable_or_upstream_images(
    reference: str,
) -> None:
    with pytest.raises(UnsafeTrainingImage):
        validate_training_image_reference(reference)


def test_training_image_policy_accepts_hardened_dev_image() -> None:
    assert validate_training_image_reference(HARDENED_LLAMAFACTORY_IMAGE) == HARDENED_LLAMAFACTORY_IMAGE


def test_training_image_policy_accepts_production_digest() -> None:
    reference = "registry.internal/openllmops/llamafactory-secure@sha256:" + "b" * 64
    assert validate_training_image_reference(reference) == reference


def test_training_image_requires_verified_build_labels() -> None:
    validate_hardening_labels(EXPECTED_SECURITY_LABELS)
    with pytest.raises(UnsafeTrainingImage, match="安全构建标签"):
        validate_hardening_labels({})


@pytest.mark.parametrize(
    "reference",
    [
        "openllmops/evaluation:latest",
        "openllmops/evaluation:0.1.0",
        "registry.internal/openllmops/evaluation:v1",
        "registry.internal/evaluation@sha256:short",
    ],
)
def test_evaluation_image_policy_rejects_latest_and_unreviewed_mutable_tags(
    reference: str,
) -> None:
    with pytest.raises(UnsafeEvaluationImage):
        validate_evaluation_image_reference(reference)


def test_evaluation_image_policy_accepts_local_build_or_production_digest() -> None:
    digest = "registry.internal/openllmops/evaluation@sha256:" + "e" * 64
    assert validate_evaluation_image_reference(EVALUATION_RUNTIME_IMAGE) == EVALUATION_RUNTIME_IMAGE
    assert validate_evaluation_image_reference(digest) == digest
    validate_evaluation_image_labels(EXPECTED_EVALUATION_LABELS)
    with pytest.raises(UnsafeEvaluationImage, match="安全构建标签"):
        validate_evaluation_image_labels({})


def test_settings_rejects_vulnerable_training_image(tmp_path) -> None:
    with pytest.raises(ValueError, match="禁止直接使用上游"):
        Settings(
            node_agent_token="a" * 32,
            llamafactory_allowed_images="hiyouga/llamafactory:0.9.5",
            model_root=tmp_path / "models",
            dataset_root=tmp_path / "datasets",
            checkpoint_root=tmp_path / "checkpoints",
            training_config_root=tmp_path / "configs",
            runtime_root=tmp_path / "runtime",
        )


@pytest.mark.parametrize(
    "reference",
    [
        "vllm/vllm-openai:latest",
        "vllm/vllm-openai:v0.10.2",
        "registry.internal/vllm:v0.27.1",
        "vllm/vllm-openai@sha256:short",
        "registry.internal/vllm@sha256:" + "f" * 64,
    ],
)
def test_vllm_policy_rejects_latest_old_or_mutable_internal_tags(reference: str) -> None:
    with pytest.raises(UnsafeVLLMImage):
        validate_vllm_image_reference(reference)


def test_vllm_policy_accepts_027_variants_and_production_digest() -> None:
    digest = f"registry.internal/openllmops/vllm@{VLLM_AMD64_DIGEST}"
    assert validate_vllm_image_reference(VLLM_RUNTIME_IMAGE) == VLLM_RUNTIME_IMAGE
    assert validate_vllm_image_reference(VLLM_CU129_RUNTIME_IMAGE) == VLLM_CU129_RUNTIME_IMAGE
    assert validate_vllm_image_reference(digest) == digest
