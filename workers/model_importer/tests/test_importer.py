from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from openllmops_model_importer import (
    ImportRequest,
    ModelImporter,
    ModelSource,
    ModelValidationError,
    validate_model_directory,
)


def create_model(root: Path, *, config: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config or {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"safe bytes for structural test")


def test_validate_and_atomically_import_controlled_directory(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    staging = tmp_path / "staging"
    store = tmp_path / "store"
    for directory in (inbox, staging, store):
        directory.mkdir()
    source = inbox / "qwen"
    create_model(source)
    importer = ModelImporter(inbox_root=inbox, staging_root=staging, store_root=store)
    import_id = uuid.uuid4()

    final, manifest = importer.run(
        ImportRequest(
            import_id=import_id,
            source=ModelSource.CONTROLLED_DIRECTORY,
            source_directory=source,
        )
    )

    assert final == store / str(import_id)
    assert not (staging / str(import_id)).exists()
    assert manifest.model_type == "qwen2"
    assert (final / "openllmops-manifest.json").is_file()


def test_rejects_pickle_weights(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")

    with pytest.raises(ModelValidationError, match="非 Safetensors"):
        validate_model_directory(tmp_path)


def test_rejects_remote_code_declaration(tmp_path: Path) -> None:
    create_model(tmp_path, config={"model_type": "custom", "auto_map": {"AutoModel": "model.X"}})

    with pytest.raises(ModelValidationError, match="auto_map"):
        validate_model_directory(tmp_path)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "model"
    create_model(root)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (root / "escaped.safetensors").symlink_to(outside)

    with pytest.raises(ModelValidationError, match="软链接"):
        validate_model_directory(root)


def test_cancellation_cleans_only_staging_directory(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    staging = tmp_path / "staging"
    store = tmp_path / "store"
    for directory in (inbox, staging, store):
        directory.mkdir()
    source = inbox / "model"
    create_model(source)
    importer = ModelImporter(inbox_root=inbox, staging_root=staging, store_root=store)
    import_id = uuid.uuid4()

    with pytest.raises(RuntimeError, match="取消"):
        importer.run(
            ImportRequest(
                import_id=import_id,
                source=ModelSource.CONTROLLED_DIRECTORY,
                source_directory=source,
            ),
            cancelled=lambda: True,
        )

    assert not (staging / str(import_id)).exists()
    assert source.exists()
