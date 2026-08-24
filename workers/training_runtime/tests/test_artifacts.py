from __future__ import annotations

import json
from pathlib import Path

import pytest

from openllmops_training_runtime import (
    TrainingArtifactError,
    validate_adapter_directory,
    validate_full_model_directory,
)


def adapter(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "auto_mapping": None}),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(b"safe adapter")


def full_model(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe model")


def test_full_model_requires_tokenizer_config_and_payload(tmp_path: Path) -> None:
    root = tmp_path / "model"
    full_model(root)
    (root / "tokenizer.json").unlink()

    with pytest.raises(TrainingArtifactError, match="真实 tokenizer"):
        validate_full_model_directory(root)


def test_adapter_requires_safetensors_and_rejects_links(tmp_path: Path) -> None:
    adapter(tmp_path / "adapter")
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "adapter" / "escape").symlink_to(outside)

    with pytest.raises(TrainingArtifactError, match="软链接"):
        validate_adapter_directory(tmp_path / "adapter")


def test_adapter_rejects_nested_pickle_state(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    adapter(root)
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"pickle")

    with pytest.raises(TrainingArtifactError, match="反序列化"):
        validate_adapter_directory(root)


def test_full_model_rejects_pickle_and_remote_code(tmp_path: Path) -> None:
    root = tmp_path / "model"
    full_model(root)
    (root / "training_args.bin").write_bytes(b"pickle")

    with pytest.raises(TrainingArtifactError, match="非 Safetensors"):
        validate_full_model_directory(root)

    (root / "training_args.bin").unlink()
    (root / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_custom.Model"}}),
        encoding="utf-8",
    )
    with pytest.raises(TrainingArtifactError, match="auto_map"):
        validate_full_model_directory(root)


def test_full_model_validates_safetensors_index(tmp_path: Path) -> None:
    root = tmp_path / "model"
    full_model(root)
    (root / "model.safetensors").unlink()
    (root / "model-00001-of-00001.safetensors").write_bytes(b"safe shard")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    assert validate_full_model_directory(root) == root.resolve()
