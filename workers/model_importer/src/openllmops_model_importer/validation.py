"""模型目录格式、安全策略与内容清单校验。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Sequence
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
MAX_JSON_METADATA_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
MAX_PARAMETER_COUNT = (1 << 63) - 1
GENERATED_MANIFEST_NAME = "openllmops-manifest.json"
# 与 safetensors 0.8 格式枚举一致。以 bit 表示可同时覆盖 F4/F6 等打包 dtype；
# BOOL 在 safetensors 中按 1 byte 存储，而不是按单 bit 存储。
SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}
TOKENIZER_PAYLOADS = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "spiece.model",
        "vocab.json",
    }
)
SHARD_NAME = re.compile(r"^model-([0-9]+)-of-([0-9]+)\.safetensors$")


class ModelValidationError(ValueError):
    """可向导入任务详情页展示的模型校验错误。"""


class ModelValidationCancelled(RuntimeError):
    """校验或哈希计算被管理员取消。"""


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
    parameter_count: int | None = None
    weight_dtypes: tuple[str, ...] = ()
    checksum: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SafetensorsSummary:
    tensor_names: frozenset[str]
    parameter_count: int
    dtypes: frozenset[str]


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise ModelValidationCancelled("模型校验已取消")


def _sha256(
    path: Path,
    *,
    on_chunk: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            _check_cancelled(cancelled)
            digest.update(chunk)
            if on_chunk:
                on_chunk(len(chunk))
            _check_cancelled(cancelled)
    _check_cancelled(cancelled)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 包含非有限数值: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise ModelValidationError(f"{label} 必须是非软链接普通文件")
        if not 1 <= file_stat.st_size <= MAX_JSON_METADATA_BYTES:
            raise ModelValidationError(f"{label} 为空或超过 16 MiB")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            value = json.load(
                source,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except ModelValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ModelValidationError(f"{label} 不是安全有效的 UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelValidationError(f"{label} 必须是 JSON 对象")
    return value


def _validate_safetensors(
    path: Path,
    cancelled: Callable[[], bool] | None = None,
) -> SafetensorsSummary:
    """只解析受限文件头，不加载权重。

    该实现刻意与 backend 的训练产物验证保持相同边界，但复制在 worker 内，避免
    导入执行器依赖控制面包。额外检查数据区必须由不重叠的张量区间完整覆盖。
    """

    _check_cancelled(cancelled)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            file_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size < 10:
                raise ModelValidationError(f"safetensors 文件为空或过小: {path.name}")
            header_size = int.from_bytes(source.read(8), "little", signed=False)
            if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ModelValidationError(f"safetensors header 大小无效: {path.name}")
            if 8 + header_size > file_stat.st_size:
                raise ModelValidationError(f"safetensors header 越界或文件被截断: {path.name}")
            raw_header = source.read(header_size)
            if len(raw_header) != header_size:
                raise ModelValidationError(f"safetensors header 被截断: {path.name}")
            _check_cancelled(cancelled)
            header = json.loads(
                raw_header,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except ModelValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ModelValidationError(f"safetensors 文件头无效: {path.name}") from exc

    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise ModelValidationError(f"safetensors 不含张量索引: {path.name}")

    data_size = file_stat.st_size - 8 - header_size
    ranges: list[tuple[int, int, str]] = []
    parameter_count = 0
    dtypes: set[str] = set()
    for name, tensor in header.items():
        _check_cancelled(cancelled)
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise ModelValidationError(f"safetensors metadata 描述无效: {path.name}")
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            raise ModelValidationError(f"safetensors 张量描述无效: {path.name}")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if (
            not isinstance(shape, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise ModelValidationError(f"safetensors 张量偏移或形状无效: {path.name}")
        if not isinstance(dtype, str) or dtype not in SAFETENSORS_DTYPE_BITS:
            raise ModelValidationError(f"safetensors 张量 dtype 不受支持: {path.name}")
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        parameter_count += element_count
        if parameter_count > MAX_PARAMETER_COUNT:
            raise ModelValidationError(f"模型参数量超出数据库可表示范围: {path.name}")
        dtypes.add(dtype)
        storage_bits = element_count * SAFETENSORS_DTYPE_BITS[dtype]
        if storage_bits % 8 or offsets[1] - offsets[0] != storage_bits // 8:
            raise ModelValidationError(
                f"safetensors 张量 dtype/shape 与偏移字节数不一致: {path.name}"
            )
        ranges.append((offsets[0], offsets[1], name))

    cursor = 0
    for start, end, _ in sorted(ranges):
        if start != cursor:
            raise ModelValidationError(f"safetensors 张量数据区存在空洞或重叠: {path.name}")
        cursor = end
    if cursor != data_size:
        raise ModelValidationError(f"safetensors 张量索引未完整覆盖数据区: {path.name}")
    return SafetensorsSummary(
        tensor_names=frozenset(name for _, _, name in ranges),
        parameter_count=parameter_count,
        dtypes=frozenset(dtypes),
    )


def _validate_shard_sequence(shard_names: set[str]) -> None:
    parsed = [SHARD_NAME.fullmatch(name) for name in shard_names]
    if not all(parsed):
        raise ModelValidationError("Safetensors 分片索引包含不安全文件名")
    matches = [match for match in parsed if match is not None]
    totals = {int(match.group(2)) for match in matches}
    if len(totals) != 1 or 0 in totals:
        raise ModelValidationError("Safetensors 分片文件声明的总分片数不一致")
    total = totals.pop()
    ordinals = {int(match.group(1)) for match in matches}
    if (
        len(matches) != total
        or len(ordinals) != len(matches)
        or ordinals != set(range(1, total + 1))
    ):
        raise ModelValidationError("Safetensors 分片序号不完整或超出声明范围")


def _select_weight_files(
    root: Path,
    top_level_files: dict[str, Path],
    safetensors: set[str],
) -> tuple[list[Path], dict[str, str] | None]:
    single_weight = top_level_files.get("model.safetensors")
    index_path = top_level_files.get("model.safetensors.index.json")
    if single_weight is not None and index_path is not None:
        raise ModelValidationError("模型不能同时包含单文件权重与 Safetensors 分片索引")
    if single_weight is not None:
        unexpected = sorted(safetensors - {"model.safetensors"})
        if unexpected:
            raise ModelValidationError(f"发现多余或非完整模型 Safetensors 权重: {unexpected[:5]}")
        return [single_weight], None
    if index_path is None:
        raise ModelValidationError("模型缺少 model.safetensors 或 Safetensors 分片索引")

    index = _load_json(index_path, "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ModelValidationError("Safetensors 索引缺少非空 weight_map")
    if not all(
        isinstance(tensor_name, str)
        and bool(tensor_name)
        and isinstance(shard_name, str)
        and bool(shard_name)
        for tensor_name, shard_name in weight_map.items()
    ):
        raise ModelValidationError("Safetensors 索引 weight_map 必须映射非空字符串")

    referenced = set(weight_map.values())
    _validate_shard_sequence(referenced)
    actual_shards = {name for name in safetensors if SHARD_NAME.fullmatch(name)}
    unexpected = sorted(safetensors - actual_shards)
    if unexpected:
        raise ModelValidationError(f"发现多余或非完整模型 Safetensors 权重: {unexpected[:5]}")
    missing = sorted(referenced - actual_shards)
    extra = sorted(actual_shards - referenced)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"缺失 {missing[:5]}")
        if extra:
            detail.append(f"多余 {extra[:5]}")
        raise ModelValidationError(f"Safetensors 索引与实际分片不一致: {'; '.join(detail)}")
    normalized_weight_map = {
        tensor_name: shard_name for tensor_name, shard_name in weight_map.items()
    }
    return [root / name for name in sorted(referenced)], normalized_weight_map


def _validate_index_tensor_mapping(
    weight_map: dict[str, str],
    tensors_by_shard: dict[str, set[str]],
) -> None:
    actual_owner: dict[str, str] = {}
    duplicates: list[str] = []
    for shard_name, tensor_names in tensors_by_shard.items():
        for tensor_name in tensor_names:
            previous = actual_owner.setdefault(tensor_name, shard_name)
            if previous != shard_name:
                duplicates.append(tensor_name)
    if duplicates:
        raise ModelValidationError(f"Safetensors 多个分片包含同名 tensor: {sorted(duplicates)[:5]}")

    indexed_names = set(weight_map)
    actual_names = set(actual_owner)
    nonexistent = sorted(indexed_names - actual_names)
    if nonexistent:
        raise ModelValidationError(f"Safetensors 索引包含不存在的 tensor: {nonexistent[:5]}")
    missing = sorted(actual_names - indexed_names)
    if missing:
        raise ModelValidationError(f"Safetensors 分片包含未被索引的 tensor: {missing[:5]}")
    wrong_shard = sorted(
        tensor_name
        for tensor_name, shard_name in weight_map.items()
        if actual_owner[tensor_name] != shard_name
    )
    if wrong_shard:
        raise ModelValidationError(f"Safetensors tensor 被映射到错误分片: {wrong_shard[:5]}")


def _validate_model_files(
    root: Path,
    files: Sequence[tuple[Path, Path]],
    cancelled: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    """校验可部署结构；供最终入库和 inbox 轻量扫描共同使用。"""

    _check_cancelled(cancelled)
    if not files:
        raise ModelValidationError("模型目录为空")
    if len(files) > MAX_MANIFEST_FILES:
        raise ModelValidationError(f"模型文件数超过 {MAX_MANIFEST_FILES} 限制")

    relative_paths = {relative.as_posix() for _, relative in files}
    if GENERATED_MANIFEST_NAME in relative_paths:
        raise ModelValidationError(f"模型目录包含保留文件名: {GENERATED_MANIFEST_NAME}")
    top_level_files = {relative.name: path for path, relative in files if len(relative.parts) == 1}
    config_path = top_level_files.get("config.json")
    tokenizer_config_path = top_level_files.get("tokenizer_config.json")
    if config_path is None or tokenizer_config_path is None:
        raise ModelValidationError("模型目录缺少 config.json 或 tokenizer_config.json")
    config = _load_json(config_path, "config.json")
    tokenizer_config = _load_json(tokenizer_config_path, "tokenizer_config.json")
    if not isinstance(config.get("model_type"), str) or not config["model_type"].strip():
        raise ModelValidationError("config.json 缺少有效 model_type")
    for label, value in (("config.json", config), ("tokenizer_config.json", tokenizer_config)):
        if value.get("auto_map"):
            raise ModelValidationError(f"{label} 声明 auto_map，需要远程代码，首版禁止导入")
        if value.get("trust_remote_code") is True:
            raise ModelValidationError(f"{label} 请求 trust_remote_code，首版禁止导入")

    payload_names = TOKENIZER_PAYLOADS & top_level_files.keys()
    if not payload_names:
        raise ModelValidationError(
            "模型目录缺少 tokenizer.json/tokenizer.model/vocab.json 等词表载荷"
        )
    empty_payloads = sorted(
        name for name in payload_names if top_level_files[name].stat().st_size == 0
    )
    if empty_payloads:
        raise ModelValidationError(f"Tokenizer 词表载荷为空: {empty_payloads[:5]}")

    forbidden = sorted(
        path for path in relative_paths if Path(path).suffix.casefold() in FORBIDDEN_SUFFIXES
    )
    if forbidden:
        raise ModelValidationError(f"发现非 Safetensors/可执行反序列化权重: {forbidden[:5]}")

    safetensors = {path for path in relative_paths if path.casefold().endswith(".safetensors")}
    weight_files, weight_map = _select_weight_files(root, top_level_files, safetensors)
    tensors_by_shard: dict[str, set[str]] = {}
    parameter_count = 0
    weight_dtypes: set[str] = set()
    for weight in weight_files:
        _check_cancelled(cancelled)
        summary = _validate_safetensors(weight, cancelled)
        tensors_by_shard[weight.name] = set(summary.tensor_names)
        parameter_count += summary.parameter_count
        if parameter_count > MAX_PARAMETER_COUNT:
            raise ModelValidationError("模型参数量超出数据库可表示范围")
        weight_dtypes.update(summary.dtypes)
    if weight_map is not None:
        _validate_index_tensor_mapping(weight_map, tensors_by_shard)
    return config, parameter_count, tuple(sorted(weight_dtypes))


def validate_model_directory(
    root: Path,
    *,
    progress: Callable[[str, int, int | None], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
) -> ModelManifest:
    """读取全部文件并生成不可变 SHA-256 清单。

    校验器不会 import 模型目录中的 Python 文件，也不会反序列化权重；这使恶意
    模型只能作为普通字节被检查，不能在控制面执行代码。
    """

    try:
        files = list(iter_regular_files(root))
    except UnsafePathError as exc:
        raise ModelValidationError(str(exc)) from exc
    total_size = sum(path.stat().st_size for path, _ in files)
    if progress:
        progress("validating", 0, total_size)
    _check_cancelled(cancelled)
    config, parameter_count, weight_dtypes = _validate_model_files(root, files, cancelled)

    digests: list[FileDigest] = []
    completed = 0

    def record_chunk(size: int) -> None:
        nonlocal completed
        completed += size
        if progress:
            progress("validating", completed, total_size)

    for path, relative in sorted(files, key=lambda item: item[1].as_posix()):
        size = path.stat().st_size
        digests.append(
            FileDigest(
                path=relative.as_posix(),
                size_bytes=size,
                sha256=_sha256(path, on_chunk=record_chunk, cancelled=cancelled),
            )
        )

    architectures = config.get("architectures")
    architecture = (
        str(architectures[0]) if isinstance(architectures, list) and architectures else None
    )
    # 整体 checksum 对排序后的“相对路径/大小/文件 SHA-256”清单再做一次摘要，
    # 因而不受目录绝对路径影响，可用于快速比较两次导入是否完全一致。
    checksum_payload = json.dumps(
        [asdict(item) for item in digests],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return ModelManifest(
        model_type=str(config.get("model_type", "unknown")),
        architecture=architecture,
        total_size_bytes=total_size,
        file_count=len(digests),
        files=tuple(digests),
        parameter_count=parameter_count,
        weight_dtypes=weight_dtypes,
        checksum=hashlib.sha256(checksum_payload).hexdigest(),
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
    )
