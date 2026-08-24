"""模型目录格式、安全策略与内容清单校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import UnsafePathError, iter_regular_files

FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
}
MAX_MANIFEST_FILES = 50_000


class ModelValidationError(ValueError):
    """可向导入任务详情页展示的模型校验错误。"""


@dataclass(frozen=True, slots=True)
class FileDigest:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_type: str
    architecture: str | None
    total_size_bytes: int
    file_count: int
    files: tuple[FileDigest, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"{label} 不是有效的 UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelValidationError(f"{label} 必须是 JSON 对象")
    return value


def _validate_weight_index(root: Path, safetensors: set[str]) -> None:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = _load_json(index_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelValidationError("Safetensors 索引缺少非空 weight_map")
    referenced = {str(value) for value in weight_map.values()}
    missing = sorted(referenced - safetensors)
    if missing:
        raise ModelValidationError(f"Safetensors 索引引用了缺失分片: {missing[:5]}")


def validate_model_directory(root: Path) -> ModelManifest:
    """读取全部文件并生成不可变 SHA-256 清单。

    校验器不会 import 模型目录中的 Python 文件，也不会反序列化权重；这使恶意
    模型只能作为普通字节被检查，不能在控制面执行代码。
    """

    try:
        files = list(iter_regular_files(root))
    except UnsafePathError as exc:
        raise ModelValidationError(str(exc)) from exc
    if not files:
        raise ModelValidationError("模型目录为空")
    if len(files) > MAX_MANIFEST_FILES:
        raise ModelValidationError(f"模型文件数超过 {MAX_MANIFEST_FILES} 限制")

    relative_paths = {relative.as_posix() for _, relative in files}
    if "config.json" not in relative_paths:
        raise ModelValidationError("模型目录缺少 config.json")
    config = _load_json(root / "config.json", "config.json")
    if config.get("auto_map"):
        raise ModelValidationError("模型声明 auto_map，需要远程代码，首版禁止导入")
    if config.get("trust_remote_code") is True:
        raise ModelValidationError("模型请求 trust_remote_code，首版禁止导入")

    forbidden = sorted(
        path for path in relative_paths if Path(path).suffix.casefold() in FORBIDDEN_SUFFIXES
    )
    if forbidden:
        raise ModelValidationError(f"发现非 Safetensors/可执行反序列化权重: {forbidden[:5]}")

    safetensors = {path for path in relative_paths if path.casefold().endswith(".safetensors")}
    if not safetensors:
        raise ModelValidationError("模型目录不包含 .safetensors 权重")
    _validate_weight_index(root, safetensors)

    digests: list[FileDigest] = []
    total_size = 0
    for path, relative in sorted(files, key=lambda item: item[1].as_posix()):
        size = path.stat().st_size
        total_size += size
        digests.append(
            FileDigest(path=relative.as_posix(), size_bytes=size, sha256=_sha256(path))
        )

    architectures = config.get("architectures")
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else None
    )
    return ModelManifest(
        model_type=str(config.get("model_type", "unknown")),
        architecture=architecture,
        total_size_bytes=total_size,
        file_count=len(digests),
        files=tuple(digests),
    )

