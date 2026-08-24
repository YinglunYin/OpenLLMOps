"""拒绝链接逃逸并生成可校验的 checkpoint 压缩包。"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path


class ArtifactPackagingError(ValueError):
    """Checkpoint 目录或输出路径不满足安全约束。"""


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    file_count: int
    total_size_bytes: int
    files: tuple[ArtifactFile, ...]
    archive_sha256: str


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _files(source: Path) -> list[tuple[Path, Path]]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ArtifactPackagingError("Checkpoint 来源必须是目录")
    files: list[tuple[Path, Path]] = []
    for root, directories, names in os.walk(source, followlinks=False):
        current = Path(root)
        for name in [*directories, *names]:
            candidate = current / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ArtifactPackagingError(f"Checkpoint 不允许软链接: {candidate.relative_to(source)}")
        for name in names:
            candidate = current / name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ArtifactPackagingError(f"Checkpoint 包含特殊文件: {candidate.relative_to(source)}")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(source):
                raise ArtifactPackagingError("Checkpoint 文件逃逸来源目录")
            files.append((candidate, candidate.relative_to(source)))
    if not files:
        raise ArtifactPackagingError("Checkpoint 目录为空")
    return sorted(files, key=lambda item: item[1].as_posix())


def create_checkpoint_archive(
    source: Path,
    destination: Path,
    *,
    artifact_root: Path,
) -> ArtifactManifest:
    """创建确定性 tar.gz；只允许写入受控 artifact_root 且不覆盖已有归档。"""

    root = artifact_root.resolve(strict=True)
    resolved_destination = destination.resolve(strict=False)
    if not resolved_destination.is_relative_to(root):
        raise ArtifactPackagingError("归档输出不在受控产物目录内")
    if resolved_destination.exists():
        raise ArtifactPackagingError("归档文件已存在，禁止覆盖")
    if resolved_destination.suffixes[-2:] != [".tar", ".gz"]:
        raise ArtifactPackagingError("归档扩展名必须是 .tar.gz")
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)

    source_files = _files(source)
    manifest_files = tuple(
        ArtifactFile(
            path=relative.as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_digest(path),
        )
        for path, relative in source_files
    )
    manifest_payload = json.dumps(
        {
            "file_count": len(manifest_files),
            "total_size_bytes": sum(item.size_bytes for item in manifest_files),
            "files": [asdict(item) for item in manifest_files],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode()

    temporary = resolved_destination.with_name(f".{resolved_destination.name}.{os.getpid()}.part")
    try:
        with (
            temporary.open("xb") as raw_output,
            # gzip 头也固定 mtime，否则 tar 内容相同仍会得到不同 SHA-256。
            gzip.GzipFile(
                filename="", fileobj=raw_output, mode="wb", compresslevel=6, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for path, relative in source_files:
                info = archive.gettarinfo(
                    str(path), arcname=f"checkpoint/{relative.as_posix()}"
                )
                # 删除宿主机身份与时间信息，归档在不同机器上仍可稳定审计。
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode &= 0o644
                with path.open("rb") as content:
                    archive.addfile(info, content)
            manifest_info = tarfile.TarInfo("openllmops-artifact-manifest.json")
            manifest_info.size = len(manifest_payload)
            manifest_info.mode = 0o644
            manifest_info.mtime = 0
            archive.addfile(manifest_info, io.BytesIO(manifest_payload))
        os.replace(temporary, resolved_destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return ArtifactManifest(
        file_count=len(manifest_files),
        total_size_bytes=sum(item.size_bytes for item in manifest_files),
        files=manifest_files,
        archive_sha256=_digest(resolved_destination),
    )
