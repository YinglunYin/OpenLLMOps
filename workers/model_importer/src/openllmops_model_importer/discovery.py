"""发现人工/SFTP 已拷入受控 inbox 的候选目录。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import UnsafePathError, iter_regular_files


@dataclass(frozen=True, slots=True)
class InboxCandidate:
    name: str
    path: str
    file_count: int
    size_bytes: int
    ready_for_import: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scan_inbox(inbox_root: Path, *, maximum_candidates: int = 1000) -> list[InboxCandidate]:
    """只扫描一层目录，避免一次 UI 刷新递归遍历整个模型仓库。"""

    root = inbox_root.resolve(strict=True)
    candidates: list[InboxCandidate] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if child.name.startswith("."):
            continue
        if len(candidates) >= maximum_candidates:
            break
        if child.is_symlink() or not child.is_dir():
            candidates.append(
                InboxCandidate(
                    name=child.name,
                    path=str(child),
                    file_count=0,
                    size_bytes=0,
                    ready_for_import=False,
                    reason="候选必须是普通目录且不能是软链接",
                )
            )
            continue
        try:
            files = list(iter_regular_files(child))
            file_count = len(files)
            size_bytes = sum(path.stat().st_size for path, _ in files)
            has_config = any(relative.as_posix() == "config.json" for _, relative in files)
            has_safetensors = any(
                relative.suffix.casefold() == ".safetensors" for _, relative in files
            )
            ready = bool(files) and has_config and has_safetensors
            reason = None if ready else "缺少 config.json、Safetensors 权重或目录为空"
        except (OSError, UnsafePathError) as exc:
            file_count = 0
            size_bytes = 0
            ready = False
            reason = str(exc)
        candidates.append(
            InboxCandidate(
                name=child.name,
                path=str(child),
                file_count=file_count,
                size_bytes=size_bytes,
                ready_for_import=ready,
                reason=reason,
            )
        )
    return candidates
