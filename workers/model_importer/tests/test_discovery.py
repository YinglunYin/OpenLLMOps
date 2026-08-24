from __future__ import annotations

from pathlib import Path

from openllmops_model_importer import scan_inbox


def test_scan_inbox_marks_structurally_ready_directory(tmp_path: Path) -> None:
    ready = tmp_path / "ready-model"
    ready.mkdir()
    (ready / "config.json").write_text("{}", encoding="utf-8")
    (ready / "model.safetensors").write_bytes(b"weights")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "README.md").write_text("missing weights", encoding="utf-8")

    candidates = {item.name: item for item in scan_inbox(tmp_path)}

    assert candidates["ready-model"].ready_for_import is True
    assert candidates["ready-model"].size_bytes > 0
    assert candidates["incomplete"].ready_for_import is False


def test_scan_inbox_rejects_top_level_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-model"
    link.symlink_to(outside, target_is_directory=True)

    candidate = next(item for item in scan_inbox(tmp_path) if item.name == "linked-model")

    assert candidate.ready_for_import is False
    assert "软链接" in (candidate.reason or "")
