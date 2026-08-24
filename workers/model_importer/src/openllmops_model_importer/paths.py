"""受控目录与文件类型安全检查。"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path


class UnsafePathError(ValueError):
    """来源路径可能逃逸受控目录或包含特殊文件。"""


def resolve_inside(root: Path, candidate: Path, *, must_exist: bool = True) -> Path:
    """解析路径后再次检查边界，不能只比较未经规范化的字符串前缀。"""

    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=must_exist)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise UnsafePathError("路径不在受控目录内")
    return resolved_candidate


def iter_regular_files(root: Path) -> Iterator[tuple[Path, Path]]:
    """遍历普通文件并拒绝所有软链接、设备文件和嵌套挂载逃逸。"""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise UnsafePathError("模型来源必须是目录")

    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *filenames]:
            child = current_path / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise UnsafePathError(f"模型目录不允许软链接: {child.relative_to(root)}")
        for filename in filenames:
            path = current_path / filename
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise UnsafePathError(f"模型目录包含非普通文件: {path.relative_to(root)}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise UnsafePathError(f"文件越过受控目录: {path.relative_to(root)}")
            yield path, path.relative_to(root)
