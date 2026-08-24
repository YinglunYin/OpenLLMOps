from __future__ import annotations

import json
from pathlib import Path

from openllmops_model_importer import scan_inbox


def _valid_safetensors() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    return len(header).to_bytes(8, "little") + header + b"\0\0\0\0"


def test_scan_inbox_marks_structurally_ready_directory(tmp_path: Path) -> None:
    ready = tmp_path / "ready-model"
    ready.mkdir()
    (ready / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (ready / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (ready / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (ready / "model.safetensors").write_bytes(_valid_safetensors())
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "README.md").write_text("missing weights", encoding="utf-8")

    candidates = {item.name: item for item in scan_inbox(tmp_path)}

    assert candidates["ready-model"].ready_for_import is True
    assert candidates["ready-model"].size_bytes > 0
    assert candidates["incomplete"].ready_for_import is False


def test_scan_inbox_does_not_mark_garbage_weights_ready(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "garbage-model"
    candidate_dir.mkdir()
    (candidate_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (candidate_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (candidate_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (candidate_dir / "model.safetensors").write_bytes(b"garbage")

    candidate = next(item for item in scan_inbox(tmp_path) if item.name == candidate_dir.name)

    assert candidate.ready_for_import is False
    assert candidate.file_count == 4
    assert candidate.size_bytes > 0
    assert "safetensors" in (candidate.reason or "")


def test_scan_inbox_rejects_top_level_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-model"
    link.symlink_to(outside, target_is_directory=True)

    candidate = next(item for item in scan_inbox(tmp_path) if item.name == "linked-model")

    assert candidate.ready_for_import is False
    assert "软链接" in (candidate.reason or "")
