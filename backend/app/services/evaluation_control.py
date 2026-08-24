from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.models import Dataset, EvaluationRun, ModelAsset
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    EvaluationTemplate,
    ModelKind,
)
from app.schemas.evaluation import EvaluationSuccessMetadata

BUILTIN_EVALUATION_FILES = {
    "ceval": Path("ceval/ceval.jsonl"),
    "cmmlu": Path("cmmlu/cmmlu.jsonl"),
}
MAX_EVALUATION_SOURCE_BYTES = 256 * 1024 * 1024
MAX_EVALUATION_TOTAL_BYTES = 512 * 1024 * 1024
MAX_EVALUATION_LINE_BYTES = 1_048_576
MAX_EVALUATION_RECORDS = 200_000
MAX_BUILTIN_MANIFEST_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _DatasetExecution:
    name: str
    path: Path
    size_bytes: int
    record_count: int


class EvaluationControlError(ValueError):
    """评测任务在进入高权限 node-agent 前未通过控制面校验。"""


def _reject_json_constant(value: str) -> None:
    raise EvaluationControlError(f"评测 manifest 不允许非有限数值：{value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationControlError("评测 manifest 包含重复字段")
        result[key] = value
    return result


def template_for_model(model_kind: ModelKind) -> EvaluationTemplate:
    if model_kind == ModelKind.BASE:
        return EvaluationTemplate.BASE
    if model_kind == ModelKind.INSTRUCT:
        return EvaluationTemplate.INSTRUCT
    raise EvaluationControlError("Embedding 模型不支持生成式评测")


def _absolute_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise EvaluationControlError(f"{label}必须是绝对路径")
    return Path(os.path.abspath(path))


def _configured_root(root: Path, label: str) -> tuple[Path, Path]:
    raw_root = _absolute_path(root, label)
    try:
        root_stat = raw_root.lstat()
    except OSError as exc:
        raise EvaluationControlError(f"{label}不存在或不可读：{raw_root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise EvaluationControlError(f"{label}必须是非软链接目录：{raw_root}")
    try:
        resolved_root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvaluationControlError(f"{label}无法解析：{raw_root}") from exc
    return raw_root, resolved_root


def _reject_symlink_components(root: Path, candidate: Path, label: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise EvaluationControlError(f"{label}越出受控目录") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise EvaluationControlError(f"无法检查{label}") from exc
        if stat.S_ISLNK(mode):
            raise EvaluationControlError(f"{label}不能包含软链接")


def _strict_existing_path(
    candidate: Path,
    root: Path,
    *,
    directory: bool,
    label: str,
) -> Path:
    raw_root, resolved_root = _configured_root(root, f"{label}受控根目录")
    raw_candidate = _absolute_path(candidate, label)
    _reject_symlink_components(raw_root, raw_candidate, label)
    try:
        resolved = raw_candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        file_stat = raw_candidate.lstat()
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationControlError(f"{label}不存在、不可读或越出受控目录") from exc
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(file_stat.st_mode) or not expected_type(file_stat.st_mode):
        expected = "目录" if directory else "普通文件"
        raise EvaluationControlError(f"{label}必须是非软链接{expected}")
    try:
        if directory:
            with os.scandir(raw_candidate):
                pass
        else:
            descriptor = os.open(raw_candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise EvaluationControlError(f"{label}必须是普通文件")
            finally:
                os.close(descriptor)
    except EvaluationControlError:
        raise
    except OSError as exc:
        raise EvaluationControlError(f"{label}不可读") from exc
    return resolved


def _read_builtin_manifest(path: Path, benchmark: str, settings: Settings) -> dict[str, Any]:
    try:
        file_stat = path.stat()
        if not 1 <= file_stat.st_size <= MAX_BUILTIN_MANIFEST_BYTES:
            raise EvaluationControlError("内置评测 manifest 为空或超过 1 MiB")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            raw = source.read(MAX_BUILTIN_MANIFEST_BYTES + 1)
        if len(raw) > MAX_BUILTIN_MANIFEST_BYTES:
            raise EvaluationControlError("内置评测 manifest 超过 1 MiB")
        manifest = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except EvaluationControlError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvaluationControlError("内置评测 manifest 无法安全读取或 JSON 无效") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise EvaluationControlError("内置评测 manifest schema_version 无效")
    if manifest.get("benchmark") != benchmark:
        raise EvaluationControlError("内置评测 manifest benchmark 与所选数据集不一致")
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict) or conversion.get("format") != "openllmops-eval-jsonl-v1":
        raise EvaluationControlError("内置评测 manifest conversion 结构无效")
    partial = conversion.get("partial")
    if not isinstance(partial, bool):
        raise EvaluationControlError("内置评测 manifest conversion.partial 无效")
    if partial and not settings.evaluation_allow_partial_builtins:
        raise EvaluationControlError("内置评测数据是不完整 partial 集，当前配置禁止启动")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise EvaluationControlError("内置评测 manifest output 结构无效")
    return output


def _fingerprint_jsonl(path: Path, label: str) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    record_count = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_EVALUATION_SOURCE_BYTES:
                raise EvaluationControlError(f"{label}为空或超过 256 MiB")
            while True:
                raw_line = source.readline(MAX_EVALUATION_LINE_BYTES + 1)
                if not raw_line:
                    break
                if len(raw_line) > MAX_EVALUATION_LINE_BYTES:
                    raise EvaluationControlError(f"{label}存在超过 1 MiB 的单行")
                digest.update(raw_line)
                if raw_line.strip():
                    record_count += 1
                    if record_count > MAX_EVALUATION_RECORDS:
                        raise EvaluationControlError(f"{label}有效记录数超过 200000")
            after = os.fstat(source.fileno())
    except EvaluationControlError:
        raise
    except OSError as exc:
        raise EvaluationControlError(f"{label}无法安全读取") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EvaluationControlError(f"{label}读取期间发生变化")
    if record_count == 0:
        raise EvaluationControlError(f"{label}不含有效记录")
    return digest.hexdigest(), record_count, before.st_size


def derive_evaluation_output_dir(settings: Settings, run_id: uuid.UUID) -> Path:
    raw_root, _ = _configured_root(settings.evaluation_output_root, "评测输出根目录")
    output_dir = raw_root / str(run_id)
    _reject_symlink_components(raw_root, output_dir, "评测输出目录")
    try:
        output_stat = output_dir.lstat()
    except FileNotFoundError:
        return output_dir
    except OSError as exc:
        raise EvaluationControlError("无法检查评测输出目录") from exc
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise EvaluationControlError("评测输出目录必须是非软链接目录")
    return output_dir


def _require_empty_output_dir(output_dir: Path) -> None:
    try:
        if any(output_dir.iterdir()):
            raise EvaluationControlError("评测输出目录已存在且非空，不能复用")
    except OSError as exc:
        raise EvaluationControlError("无法检查评测输出目录内容") from exc


def _builtin_dataset_paths(settings: Settings, names: list[str]) -> list[_DatasetExecution]:
    root, _ = _configured_root(settings.evaluation_dataset_root, "内置评测数据根目录")
    datasets: list[_DatasetExecution] = []
    for name in sorted(names):
        relative = BUILTIN_EVALUATION_FILES.get(name)
        if relative is None:
            raise EvaluationControlError(f"不支持的内置评测数据集：{name}")
        raw_dataset_path = root / relative
        dataset_path = _strict_existing_path(
            raw_dataset_path,
            root,
            directory=False,
            label=f"内置评测数据集 {name}",
        )
        if dataset_path.suffix.lower() != ".jsonl":  # pragma: no cover - 固定映射的纵深检查。
            raise EvaluationControlError(f"内置评测数据集 {name} 必须是 JSONL 文件")
        manifest_path = _strict_existing_path(
            raw_dataset_path.with_suffix(".manifest.json"),
            root,
            directory=False,
            label=f"内置评测数据集 {name} manifest",
        )
        output = _read_builtin_manifest(manifest_path, name, settings)
        digest, record_count, size_bytes = _fingerprint_jsonl(
            dataset_path,
            f"内置评测数据集 {name}",
        )
        if output.get("path") != dataset_path.name:
            raise EvaluationControlError(f"内置评测数据集 {name} manifest output.path 无效")
        if output.get("sha256") != digest:
            raise EvaluationControlError(f"内置评测数据集 {name} 与 manifest SHA-256 不一致")
        manifest_records = output.get("record_count")
        if (
            isinstance(manifest_records, bool)
            or not isinstance(manifest_records, int)
            or manifest_records != record_count
        ):
            raise EvaluationControlError(f"内置评测数据集 {name} 与 manifest 记录数不一致")
        datasets.append(_DatasetExecution(name, dataset_path, size_bytes, record_count))
    return datasets


def _custom_dataset_execution(settings: Settings, dataset: Dataset) -> _DatasetExecution:
    if dataset.dataset_type != DatasetType.EVALUATION or dataset.status != DatasetStatus.READY:
        raise EvaluationControlError("自定义数据集必须是 ready 的 evaluation 数据集")
    dataset_path = _strict_existing_path(
        Path(dataset.local_path),
        settings.dataset_root,
        directory=False,
        label="自定义评测数据集",
    )
    if dataset_path.suffix.lower() != ".jsonl":
        raise EvaluationControlError("自定义评测数据集必须是 JSONL 文件")
    digest, record_count, size_bytes = _fingerprint_jsonl(dataset_path, "自定义评测数据集")
    if dataset.sha256 != digest or dataset.record_count != record_count or dataset.size_bytes != size_bytes:
        raise EvaluationControlError("自定义评测数据集文件与数据库指纹、记录数或大小不一致")
    return _DatasetExecution(
        f"custom-{dataset.id.hex[:12]}",
        dataset_path,
        size_bytes,
        record_count,
    )


def build_evaluation_execution(
    row: EvaluationRun,
    base: ModelAsset,
    candidate: ModelAsset,
    custom_dataset: Dataset | None,
    settings: Settings,
) -> dict[str, Any]:
    """重建完全由数据库和可信配置派生的 Agent execution。"""

    if base.status != AssetStatus.READY or candidate.status != AssetStatus.READY:
        raise EvaluationControlError("基线模型和候选模型必须处于 ready 状态")
    expected_base_template = template_for_model(base.model_kind)
    expected_candidate_template = template_for_model(candidate.model_kind)
    if row.base_template != expected_base_template or row.candidate_template != expected_candidate_template:
        raise EvaluationControlError("评测模板与模型类型不一致")
    base_path = _strict_existing_path(
        Path(base.local_path),
        settings.model_root,
        directory=True,
        label="基线模型路径",
    )
    candidate_path = _strict_existing_path(
        Path(candidate.local_path),
        settings.model_root,
        directory=True,
        label="候选模型路径",
    )

    expected_output = derive_evaluation_output_dir(settings, row.id)
    if expected_output.exists():
        _require_empty_output_dir(expected_output)
    if Path(row.output_dir) != expected_output:
        raise EvaluationControlError("评测输出目录不是控制面按 run UUID 派生的路径")
    if row.tensor_parallel_size != len(row.gpu_ids):
        raise EvaluationControlError("评测 tensor_parallel_size 必须等于 GPU 数量")
    if not 1 <= row.tensor_parallel_size <= 16:
        raise EvaluationControlError("评测 tensor_parallel_size 超出 1..16")
    if not 0.1 <= row.gpu_memory_utilization <= 0.95:
        raise EvaluationControlError("评测 gpu_memory_utilization 超出 0.1..0.95")
    if not 1 <= row.concurrency <= 32:
        raise EvaluationControlError("评测 concurrency 超出 1..32")
    if not 1 <= row.max_tokens <= 512:
        raise EvaluationControlError("评测 max_tokens 超出 1..512")

    datasets = _builtin_dataset_paths(settings, row.builtin_datasets)
    if row.custom_dataset_id is not None:
        if custom_dataset is None or custom_dataset.id != row.custom_dataset_id:
            raise EvaluationControlError("评测任务引用的自定义数据集不存在")
        datasets.append(_custom_dataset_execution(settings, custom_dataset))
    elif custom_dataset is not None:
        raise EvaluationControlError("评测任务包含未引用的自定义数据集")
    if not datasets:
        raise EvaluationControlError("评测任务没有可执行的数据集")
    resolved_paths = [dataset.path for dataset in datasets]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise EvaluationControlError("同一评测 JSONL 不能以多个 name 重复计权")
    if sum(dataset.size_bytes for dataset in datasets) > MAX_EVALUATION_TOTAL_BYTES:
        raise EvaluationControlError("评测数据集原始字节合计超过 512 MiB")
    if sum(dataset.record_count for dataset in datasets) > MAX_EVALUATION_RECORDS:
        raise EvaluationControlError("评测数据集有效记录合计超过 200000")

    return {
        "runner": "evaluation",
        "base_model_path": str(base_path),
        "candidate_model_path": str(candidate_path),
        "base_template": row.base_template.value,
        "candidate_template": row.candidate_template.value,
        "datasets": [{"name": item.name, "path": str(item.path)} for item in datasets],
        "output_dir": str(expected_output),
        "tensor_parallel_size": row.tensor_parallel_size,
        "gpu_memory_utilization": row.gpu_memory_utilization,
        "concurrency": row.concurrency,
        "max_tokens": row.max_tokens,
    }


def validate_evaluation_success_metadata(
    row: EvaluationRun,
    metadata: dict[str, Any],
    settings: Settings,
) -> EvaluationSuccessMetadata:
    """在落库前校验签名响应中的报告类型、交叉字段和系统派生路径。"""

    try:
        parsed = EvaluationSuccessMetadata.model_validate(metadata)
    except ValidationError as exc:
        raise EvaluationControlError("node-agent 返回的评测成功元数据结构无效") from exc
    if parsed.metrics.baseline.template != row.base_template.value:
        raise EvaluationControlError("评测报告 baseline template 与任务配置不一致")
    if parsed.metrics.candidate.template != row.candidate_template.value:
        raise EvaluationControlError("评测报告 candidate template 与任务配置不一致")

    expected_output = derive_evaluation_output_dir(settings, row.id)
    expected_result = expected_output / "pair-report.json"
    if Path(parsed.result_path) != expected_result:
        raise EvaluationControlError("评测 result_path 不是控制面派生的结果路径")
    _strict_existing_path(
        Path(parsed.result_path),
        settings.evaluation_output_root,
        directory=False,
        label="评测结果文件",
    )
    expected_manifest = (
        settings.node_agent_runtime_root
        / "contract"
        / "evaluation"
        / str(row.id)
        / str(row.runtime_generation)
        / "dataset-manifest.json"
    )
    if Path(parsed.dataset_manifest_path) != expected_manifest:
        raise EvaluationControlError("评测 dataset_manifest_path 与任务代际不一致")
    _strict_existing_path(
        Path(parsed.dataset_manifest_path),
        settings.node_agent_runtime_root,
        directory=False,
        label="评测数据 manifest",
    )
    if len(parsed.warnings) != len(set(parsed.warnings)):
        raise EvaluationControlError("评测 warnings 不能重复")
    return parsed
