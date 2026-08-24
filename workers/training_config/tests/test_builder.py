from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openllmops_training_config import (
    Algorithm,
    DatasetFormat,
    Stage,
    TrainingConfigError,
    TrainingHyperparameters,
    TrainingRequest,
    build_dataset_info,
    build_training_config,
)


def request(**overrides: object) -> TrainingRequest:
    values: dict[str, object] = {
        "stage": Stage.SFT,
        "algorithm": Algorithm.LORA,
        "model_path": Path("/srv/openllmops/models/model-1"),
        "dataset_dir": Path("/srv/openllmops/datasets/dataset-1"),
        "output_dir": Path("/srv/openllmops/checkpoints/job-1"),
        "dataset_format": DatasetFormat.ALPACA,
        "template": "qwen",
    }
    values.update(overrides)
    return TrainingRequest(**values)  # type: ignore[arg-type]


def test_cpt_only_accepts_lora() -> None:
    with pytest.raises(TrainingConfigError, match="仅支持 LoRA"):
        build_training_config(
            request(
                stage=Stage.CPT,
                algorithm=Algorithm.FREEZE,
                dataset_format=DatasetFormat.CPT_TEXT,
                template=None,
            )
        )


def test_qlora_uses_lora_with_four_bit_quantization() -> None:
    config = build_training_config(request(algorithm=Algorithm.QLORA))

    assert config["finetuning_type"] == "lora"
    assert config["quantization_bit"] == 4
    assert config["quantization_method"] == "bitsandbytes"
    assert config["trust_remote_code"] is False
    assert config["save_safetensors"] is True


def test_freeze_configuration_does_not_include_lora_fields() -> None:
    config = build_training_config(request(algorithm=Algorithm.FREEZE))

    assert config["finetuning_type"] == "freeze"
    assert config["freeze_trainable_layers"] == 2
    assert "lora_target" not in config
    assert "quantization_bit" not in config


def test_sft_requires_explicit_template() -> None:
    with pytest.raises(TrainingConfigError, match="显式选择"):
        build_training_config(request(template=None))


def test_dataset_info_rejects_parent_path() -> None:
    with pytest.raises(TrainingConfigError, match="当前目录"):
        build_dataset_info("../escape.jsonl", DatasetFormat.ALPACA)


def test_messages_dataset_uses_sharegpt_mapping() -> None:
    info = build_dataset_info("data.jsonl", DatasetFormat.MESSAGES)

    assert info["openllmops_dataset"]["formatting"] == "sharegpt"
    assert info["openllmops_dataset"]["columns"]["messages"] == "messages"


@pytest.mark.parametrize(
    "payload",
    [
        {"trust_remote_code": True},
        {"learning_rate": "0.0002"},
        {"cutoff_len": 65_537},
        {"warmup_ratio": float("nan")},
        {"template": "../custom.py"},
        {"max_samples": 10_000_001},
    ],
)
def test_hyperparameters_reject_extra_coerced_or_out_of_range_values(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TrainingHyperparameters.model_validate(payload)


def test_hyperparameters_expose_only_frozen_control_plane_fields() -> None:
    fields = set(TrainingHyperparameters.model_fields)

    assert fields == {
        "template",
        "num_train_epochs",
        "learning_rate",
        "cutoff_len",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "warmup_ratio",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "freeze_trainable_layers",
        "max_samples",
        "seed",
    }
