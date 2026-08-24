from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from openllmops_artifacts import ArtifactPackagingError, create_checkpoint_archive


def test_creates_archive_with_manifest(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-100"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "trainer_state.json").write_text("{}", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    destination = artifact_root / "job-1.tar.gz"

    manifest = create_checkpoint_archive(source, destination, artifact_root=artifact_root)

    assert manifest.file_count == 2
    assert len(manifest.archive_sha256) == 64
    with tarfile.open(destination, "r:gz") as archive:
        assert "checkpoint/adapter_model.safetensors" in archive.getnames()
        assert "openllmops-artifact-manifest.json" in archive.getnames()

    second = artifact_root / "job-1-copy.tar.gz"
    second_manifest = create_checkpoint_archive(source, second, artifact_root=artifact_root)
    assert second_manifest.archive_sha256 == manifest.archive_sha256


def test_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    (source / "link").symlink_to(outside)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    with pytest.raises(ArtifactPackagingError, match="软链接"):
        create_checkpoint_archive(
            source, artifact_root / "unsafe.tar.gz", artifact_root=artifact_root
        )


def test_rejects_output_outside_artifact_root(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "state.json").write_text("{}", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    with pytest.raises(ArtifactPackagingError, match="不在受控"):
        create_checkpoint_archive(
            source, tmp_path / "escaped.tar.gz", artifact_root=artifact_root
        )
