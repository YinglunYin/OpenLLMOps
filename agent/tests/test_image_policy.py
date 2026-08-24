import pytest

from openllmops_agent.config import Settings
from openllmops_agent.image_policy import (
    EXPECTED_SECURITY_LABELS,
    HARDENED_LLAMAFACTORY_IMAGE,
    UnsafeTrainingImage,
    validate_hardening_labels,
    validate_training_image_reference,
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
