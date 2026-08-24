from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import openllmops_model_importer.importer as importer_module
from openllmops_model_importer import (
    ImportRequest,
    ModelImporter,
    ModelSource,
    ModelValidationError,
    validate_model_directory,
)
from openllmops_model_importer.downloaders import DownloadResult
from openllmops_model_importer.importer import ImportCancelledError


def safetensors_bytes(
    *,
    data: bytes = b"\0\0\0\0",
    offsets: tuple[int, int] | None = None,
    dtype: str = "F32",
    shape: list[int] | None = None,
) -> bytes:
    tensor_offsets = offsets or (0, len(data))
    header = json.dumps(
        {
            "weight": {
                "dtype": dtype,
                "shape": [1] if shape is None else shape,
                "data_offsets": list(tensor_offsets),
            }
        },
        separators=(",", ":"),
    ).encode()
    return len(header).to_bytes(8, "little") + header + data


def safetensors_with_tensors(*tensor_names: str) -> bytes:
    header: dict[str, dict[str, object]] = {}
    data = bytearray()
    for tensor_name in tensor_names:
        start = len(data)
        data.extend(b"\0\0\0\0")
        header[tensor_name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [start, len(data)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    return len(raw_header).to_bytes(8, "little") + raw_header + bytes(data)


def create_model(root: Path, *, config: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config or {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(safetensors_bytes())


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


def test_rejects_garbage_safetensors(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not-a-safetensors-file")

    with pytest.raises(ModelValidationError, match="safetensors"):
        validate_model_directory(tmp_path)


def test_rejects_truncated_safetensors_header(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes((128).to_bytes(8, "little") + b"{}")

    with pytest.raises(ModelValidationError, match="截断|越界"):
        validate_model_directory(tmp_path)


def test_rejects_out_of_bounds_safetensors_offsets(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0\0\0\0", offsets=(0, 8))
    )

    with pytest.raises(ModelValidationError, match="偏移"):
        validate_model_directory(tmp_path)


def test_rejects_safetensors_data_not_covered_by_tensor_index(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0\0\0\0extra", offsets=(0, 4))
    )

    with pytest.raises(ModelValidationError, match="未完整覆盖"):
        validate_model_directory(tmp_path)


def test_accepts_dtype_shape_with_matching_storage_bytes(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0" * 12, dtype="BF16", shape=[2, 3])
    )

    manifest = validate_model_directory(tmp_path)

    assert manifest.model_type == "qwen2"
    assert manifest.parameter_count == 6
    assert manifest.weight_dtypes == ("BF16",)
    assert manifest.checksum is not None and len(manifest.checksum) == 64


def test_accepts_byte_aligned_packed_safetensors_dtype(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0", dtype="F4", shape=[2])
    )

    assert validate_model_directory(tmp_path).file_count == 4


def test_rejects_unknown_safetensors_dtype(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0\0\0\0", dtype="MADE_UP", shape=[1])
    )

    with pytest.raises(ModelValidationError, match="dtype 不受支持"):
        validate_model_directory(tmp_path)


def test_rejects_shape_that_does_not_match_tensor_byte_range(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0\0\0\0", dtype="F32", shape=[2])
    )

    with pytest.raises(ModelValidationError, match="dtype/shape"):
        validate_model_directory(tmp_path)


def test_rejects_missing_tokenizer_payload(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "tokenizer.json").unlink()

    with pytest.raises(ModelValidationError, match="词表载荷"):
        validate_model_directory(tmp_path)


def test_rejects_source_that_shadows_generated_manifest(tmp_path: Path) -> None:
    create_model(tmp_path)
    (tmp_path / "openllmops-manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelValidationError, match="保留文件名"):
        validate_model_directory(tmp_path)


def _replace_with_sharded_model(
    root: Path,
    *,
    referenced: list[str],
    actual: list[str],
) -> None:
    (root / "model.safetensors").unlink()
    tensor_by_shard = {name: f"layer.{index}" for index, name in enumerate(referenced)}
    for name in actual:
        tensor_name = tensor_by_shard.get(name, f"extra.{name}")
        (root / name).write_bytes(safetensors_with_tensors(tensor_name))
    weight_map = {tensor_name: name for name, tensor_name in tensor_by_shard.items()}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def test_accepts_complete_sharded_safetensors_model(tmp_path: Path) -> None:
    create_model(tmp_path)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _replace_with_sharded_model(tmp_path, referenced=shards, actual=shards)

    manifest = validate_model_directory(tmp_path)

    assert {item.path for item in manifest.files}.issuperset(shards)


def test_rejects_missing_safetensors_shard_referenced_by_index(tmp_path: Path) -> None:
    create_model(tmp_path)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _replace_with_sharded_model(tmp_path, referenced=shards, actual=shards[:1])

    with pytest.raises(ModelValidationError, match="缺失"):
        validate_model_directory(tmp_path)


def test_rejects_extra_safetensors_shard_not_referenced_by_index(tmp_path: Path) -> None:
    create_model(tmp_path)
    referenced = ["model-00001-of-00001.safetensors"]
    actual = [*referenced, "model-00002-of-00002.safetensors"]
    _replace_with_sharded_model(tmp_path, referenced=referenced, actual=actual)

    with pytest.raises(ModelValidationError, match="多余"):
        validate_model_directory(tmp_path)


def test_rejects_incomplete_shard_number_sequence(tmp_path: Path) -> None:
    create_model(tmp_path)
    shards = ["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    _replace_with_sharded_model(tmp_path, referenced=shards, actual=shards)

    with pytest.raises(ModelValidationError, match="分片序号不完整"):
        validate_model_directory(tmp_path)


def test_rejects_tensor_mapped_to_wrong_shard(tmp_path: Path) -> None:
    create_model(tmp_path)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _replace_with_sharded_model(tmp_path, referenced=shards, actual=shards)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": shards[1],
                    "layer.1": shards[0],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelValidationError, match="错误分片"):
        validate_model_directory(tmp_path)


def test_rejects_shard_tensor_missing_from_weight_map(tmp_path: Path) -> None:
    create_model(tmp_path)
    shard = "model-00001-of-00001.safetensors"
    _replace_with_sharded_model(tmp_path, referenced=[shard], actual=[shard])
    (tmp_path / shard).write_bytes(safetensors_with_tensors("layer.0", "unindexed.weight"))

    with pytest.raises(ModelValidationError, match="未被索引"):
        validate_model_directory(tmp_path)


def test_rejects_weight_map_tensor_missing_from_shard(tmp_path: Path) -> None:
    create_model(tmp_path)
    shard = "model-00001-of-00001.safetensors"
    _replace_with_sharded_model(tmp_path, referenced=[shard], actual=[shard])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": shard,
                    "nonexistent.weight": shard,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelValidationError, match="不存在的 tensor"):
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


def test_online_import_records_requested_and_resolved_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inbox = tmp_path / "inbox"
    staging = tmp_path / "staging"
    store = tmp_path / "store"
    for directory in (inbox, staging, store):
        directory.mkdir()
    resolved = "a" * 40
    observed_progress: list[tuple[str, int, int | None]] = []

    def fake_download(  # type: ignore[no-untyped-def]
        repository,
        revision,
        destination,
        token,
        *,
        progress,
        cancelled,
    ):
        assert repository == "Qwen/Test"
        assert revision == "main"
        assert token == "secret"
        assert cancelled is not None and not cancelled()
        create_model(destination)
        if progress:
            progress("transferring", 4, 4)
        return DownloadResult(resolved_revision=resolved, total_bytes=4)

    monkeypatch.setattr(importer_module, "download_huggingface", fake_download)
    importer = ModelImporter(inbox_root=inbox, staging_root=staging, store_root=store)

    final, manifest = importer.run(
        ImportRequest(
            import_id=uuid.uuid4(),
            source=ModelSource.HUGGINGFACE,
            repository="Qwen/Test",
            revision="main",
            access_token="secret",
        ),
        progress=lambda stage, done, total: observed_progress.append((stage, done, total)),
        cancelled=lambda: False,
    )

    assert manifest.requested_revision == "main"
    assert manifest.resolved_revision == resolved
    stored_manifest = json.loads((final / "openllmops-manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest["requested_revision"] == "main"
    assert stored_manifest["resolved_revision"] == resolved
    validating = [item for item in observed_progress if item[0] == "validating"]
    assert validating[0][1] == 0
    assert validating[0][2] == manifest.total_size_bytes
    assert validating[-1][1] == validating[-1][2] == manifest.total_size_bytes


def test_cancellation_during_hashing_removes_staging(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    staging = tmp_path / "staging"
    store = tmp_path / "store"
    for directory in (inbox, staging, store):
        directory.mkdir()
    source = inbox / "large-model"
    create_model(source)
    data_size = 16 * 1024 * 1024
    (source / "model.safetensors").write_bytes(
        safetensors_bytes(data=b"\0" * data_size, dtype="F32", shape=[data_size // 4])
    )
    importer = ModelImporter(inbox_root=inbox, staging_root=staging, store_root=store)
    import_id = uuid.uuid4()
    state = {"cancel": False}

    def progress(stage: str, completed: int, total: int | None) -> None:
        del total
        if stage == "validating" and completed >= 8 * 1024 * 1024:
            state["cancel"] = True

    with pytest.raises(ImportCancelledError, match="取消"):
        importer.run(
            ImportRequest(
                import_id=import_id,
                source=ModelSource.CONTROLLED_DIRECTORY,
                source_directory=source,
            ),
            progress=progress,
            cancelled=lambda: state["cancel"],
        )

    assert not (staging / str(import_id)).exists()
    assert not (store / str(import_id)).exists()
