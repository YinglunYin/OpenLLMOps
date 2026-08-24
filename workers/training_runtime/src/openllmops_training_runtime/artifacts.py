"""只通过普通文件检查训练产物，不反序列化模型权重。"""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any

MAX_FILES = 50_000
MAX_JSON_BYTES = 8 * 1024 * 1024
FORBIDDEN_DIRECT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".joblib", ".pkl", ".pickle", ".pt", ".pth"}
)


class TrainingArtifactError(ValueError):
    """训练产物不是可安全部署或上报的 Safetensors 目录。"""


def _reject_constant(value: str) -> None:
    raise TrainingArtifactError(f"JSON 不允许非有限数值：{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingArtifactError(f"JSON 包含重复字段：{key}")
        result[key] = value
    return result


def _bounded_json(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TrainingArtifactError(f"{label} 不存在") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise TrainingArtifactError(f"{label} 必须是非软链接普通文件")
    if not 1 <= info.st_size <= MAX_JSON_BYTES:
        raise TrainingArtifactError(f"{label} 大小无效")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            raw = source.read(MAX_JSON_BYTES + 1)
        value = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except TrainingArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TrainingArtifactError(f"{label} 不是有效的有限 UTF-8 JSON") from exc
    if len(raw) > MAX_JSON_BYTES or not isinstance(value, dict):
        raise TrainingArtifactError(f"{label} 必须是有界 JSON 对象")
    _validate_finite_tree(value)
    return value


def _validate_finite_tree(value: Any, *, depth: int = 0, count: list[int] | None = None) -> None:
    if count is None:
        count = [0]
    count[0] += 1
    if depth > 32 or count[0] > 100_000:
        raise TrainingArtifactError("产物 JSON 结构过深或过大")
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingArtifactError("产物 JSON 包含非有限数值")
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_tree(item, depth=depth + 1, count=count)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 512:
                raise TrainingArtifactError("产物 JSON 字段名无效")
            _validate_finite_tree(item, depth=depth + 1, count=count)
        return
    raise TrainingArtifactError("产物 JSON 包含不支持的数据类型")


def _files(root: Path) -> dict[str, Path]:
    try:
        raw = Path(os.path.abspath(root))
        root_info = raw.lstat()
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingArtifactError(f"训练产物目录不存在：{root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise TrainingArtifactError("训练产物必须是非软链接目录")

    result: dict[str, Path] = {}
    for current, directories, names in os.walk(resolved, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *names]:
            candidate = current_path / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                raise TrainingArtifactError("无法检查训练产物") from exc
            if stat.S_ISLNK(mode):
                raise TrainingArtifactError(
                    f"训练产物禁止软链接：{candidate.relative_to(resolved)}"
                )
            if name in directories and not stat.S_ISDIR(mode):
                raise TrainingArtifactError("训练产物包含非普通目录")
            if name in names and not stat.S_ISREG(mode):
                raise TrainingArtifactError("训练产物包含特殊文件")
        for name in names:
            candidate = current_path / name
            relative = candidate.relative_to(resolved).as_posix()
            result[relative] = candidate
            if len(result) > MAX_FILES:
                raise TrainingArtifactError(f"训练产物文件数超过 {MAX_FILES}")
    if not result:
        raise TrainingArtifactError("训练产物目录为空")
    return result


def _validate_safe_config(config: dict[str, Any], label: str) -> None:
    if config.get("auto_map") or config.get("auto_mapping"):
        raise TrainingArtifactError(f"{label} 声明 auto_map/auto_mapping，禁止作为受控产物")
    if config.get("trust_remote_code") is True:
        raise TrainingArtifactError(f"{label} 请求 trust_remote_code")


def validate_adapter_directory(root: Path) -> Path:
    """确认目录包含 PEFT 配置和 Safetensors adapter。"""

    resolved = Path(os.path.abspath(root)).resolve(strict=True)
    files = _files(resolved)
    config = _bounded_json(resolved / "adapter_config.json", "adapter_config.json")
    _validate_safe_config(config, "adapter_config.json")
    weights = {
        name
        for name in files
        if "/" not in name
        and name.startswith("adapter_model")
        and name.casefold().endswith(".safetensors")
    }
    if not weights:
        raise TrainingArtifactError("adapter 目录缺少 adapter_model*.safetensors")
    forbidden = sorted(
        name for name in files if Path(name).suffix.casefold() in FORBIDDEN_DIRECT_SUFFIXES
    )
    if forbidden:
        raise TrainingArtifactError(f"adapter 根目录包含可执行反序列化文件：{forbidden[:5]}")
    return resolved


def _validate_weight_index(root: Path, direct_safetensors: set[str]) -> None:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = _bounded_json(index_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TrainingArtifactError("Safetensors 索引缺少 weight_map")
    referenced = set()
    for value in weight_map.values():
        if not isinstance(value, str) or Path(value).name != value:
            raise TrainingArtifactError("Safetensors 索引包含路径或非字符串值")
        referenced.add(value)
    if not referenced.issubset(direct_safetensors):
        raise TrainingArtifactError("Safetensors 索引引用缺失分片")


def validate_full_model_directory(root: Path) -> Path:
    """确认目录可由禁用远程代码的模型服务安全加载。"""

    resolved = Path(os.path.abspath(root)).resolve(strict=True)
    files = _files(resolved)
    config = _bounded_json(resolved / "config.json", "config.json")
    _validate_safe_config(config, "config.json")
    direct_files = {name for name in files if "/" not in name}
    forbidden = sorted(
        name for name in files if Path(name).suffix.casefold() in FORBIDDEN_DIRECT_SUFFIXES
    )
    if forbidden:
        raise TrainingArtifactError(f"模型根目录包含非 Safetensors 权重：{forbidden[:5]}")
    safetensors = {name for name in direct_files if name.casefold().endswith(".safetensors")}
    if not safetensors:
        raise TrainingArtifactError("完整模型目录缺少 Safetensors 权重")
    _validate_weight_index(resolved, safetensors)
    if "tokenizer_config.json" not in direct_files:
        raise TrainingArtifactError("完整模型目录缺少 tokenizer_config.json")
    tokenizer_payloads = {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"}
    if not tokenizer_payloads & direct_files:
        raise TrainingArtifactError("完整模型目录缺少真实 tokenizer 词表或模型文件")
    tokenizer = _bounded_json(resolved / "tokenizer_config.json", "tokenizer_config.json")
    _validate_safe_config(tokenizer, "tokenizer_config.json")
    return resolved


def validate_checkpoint_directory(root: Path) -> Path:
    """checkpoint 必须至少是一个可验证的 adapter 或完整模型目录。"""

    candidate = Path(os.path.abspath(root))
    if (candidate / "adapter_config.json").exists():
        return validate_adapter_directory(candidate)
    return validate_full_model_directory(candidate)
