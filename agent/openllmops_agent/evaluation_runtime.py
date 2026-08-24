from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

MAX_DATASET_FILES = 16
MAX_DATASET_FILE_BYTES = 256 * 1024 * 1024
MAX_DATASET_TOTAL_BYTES = 512 * 1024 * 1024
MAX_DATASET_LINE_BYTES = 1_048_576
MAX_DATASET_RECORDS = 200_000
# pair report 包含 baseline/candidate 两份 sample_ids；20 万条上限时需要
# 显式为两份列表留出空间，仍以文件和 JSON 节点双重上限防止内存滥用。
MAX_REPORT_BYTES = 128 * 1024 * 1024
MAX_REPORT_JSON_NODES = 600_000
MAX_REPORT_CATEGORIES = 4_096
MAX_MANIFEST_BYTES = 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationInputError(ValueError):
    """评测路径、数据或产物不满足节点侧安全约束。"""


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class EvaluationWorkspace:
    dataset_path: Path
    dataset_manifest_path: Path
    output_path: Path


def _absolute_path(path: Path) -> Path:
    if not path.is_absolute():
        raise EvaluationInputError(f"评测路径必须是绝对路径：{path}")
    return Path(os.path.abspath(path))


def _root_path(root: Path) -> tuple[Path, Path]:
    raw_root = _absolute_path(root)
    if raw_root.is_symlink():
        raise EvaluationInputError(f"评测受控根目录不能是软链接：{root}")
    try:
        resolved = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvaluationInputError(f"评测受控根目录不存在：{root}") from exc
    if not resolved.is_dir():
        raise EvaluationInputError(f"评测受控根路径不是目录：{root}")
    return raw_root, resolved


def _reject_symlink_components(raw_root: Path, candidate: Path) -> None:
    try:
        relative = candidate.relative_to(raw_root)
    except ValueError as exc:
        raise EvaluationInputError(f"评测路径越出受控目录：{candidate}") from exc
    current = raw_root
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise EvaluationInputError(f"无法检查评测路径：{candidate}") from exc
        if stat.S_ISLNK(mode):
            raise EvaluationInputError(f"评测路径不能包含软链接：{candidate}")


def strict_existing_path(
    candidate: Path,
    roots: tuple[Path, ...],
    *,
    directory: bool,
) -> tuple[Path, Path]:
    """返回（解析后路径、命中的解析后根目录），并拒绝链接与目录逃逸。"""

    raw_candidate = _absolute_path(candidate)
    for configured_root in roots:
        raw_root, resolved_root = _root_path(configured_root)
        try:
            raw_candidate.relative_to(raw_root)
        except ValueError:
            continue
        _reject_symlink_components(raw_root, raw_candidate)
        try:
            resolved = raw_candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvaluationInputError(f"评测路径不存在或越出受控目录：{candidate}") from exc
        if directory and not resolved.is_dir():
            raise EvaluationInputError(f"评测模型路径必须是普通目录：{candidate}")
        if not directory and not resolved.is_file():
            raise EvaluationInputError(f"评测数据路径必须是普通文件：{candidate}")
        return resolved, resolved_root
    raise EvaluationInputError(f"评测路径越出所有允许的受控目录：{candidate}")


def prepare_output_directory(candidate: Path, output_root: Path, run_id: UUID) -> Path:
    raw_root, resolved_root = _root_path(output_root)
    raw_candidate = _absolute_path(candidate)
    expected = raw_root / str(run_id)
    if raw_candidate != expected:
        raise EvaluationInputError(f"评测输出目录必须由系统派生为：{expected}")
    _reject_symlink_components(raw_root, raw_candidate)
    try:
        if raw_candidate.exists():
            if raw_candidate.is_symlink() or not raw_candidate.is_dir():
                raise EvaluationInputError("评测输出路径必须是非软链接目录")
            if any(raw_candidate.iterdir()):
                raise EvaluationInputError("新的评测任务不能复用非空输出目录")
        else:
            raw_candidate.mkdir(mode=0o700)
        raw_candidate.chmod(0o700)
        resolved = raw_candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except EvaluationInputError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationInputError("无法创建受控评测输出目录") from exc
    return resolved


def _derived_runtime_directory(runtime_root: Path, run_id: UUID, generation: int) -> Path:
    raw_root, resolved_root = _root_path(runtime_root)
    current = raw_root
    for part in ("contract", "evaluation", str(run_id), str(generation)):
        current /= part
        try:
            if current.is_symlink():
                raise EvaluationInputError("评测运行目录不能包含软链接")
            current.mkdir(mode=0o700, exist_ok=True)
            if not current.is_dir():
                raise EvaluationInputError("评测运行路径不是目录")
            current.chmod(0o700)
        except EvaluationInputError:
            raise
        except OSError as exc:
            raise EvaluationInputError("无法创建评测运行目录") from exc
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - 每段均为系统派生，保留纵深检查。
        raise EvaluationInputError("评测运行目录越出受控根目录") from exc
    return resolved


def _reject_json_constant(value: str) -> None:
    raise EvaluationInputError(f"JSON 不允许非有限数值：{value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInputError(f"JSON 对象包含重复字段：{key}")
        result[key] = value
    return result


def _loads_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except EvaluationInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvaluationInputError("评测 JSON 不是有效的 UTF-8 对象") from exc


def _atomic_json(path: Path, value: dict[str, Any], mode: int) -> None:
    body = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_file(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationInputError(f"无法安全打开评测数据文件：{path}") from exc
    return os.fdopen(descriptor, "rb")


def _safe_short_text(
    value: Any,
    field: str,
    *,
    default: str | None = None,
    max_length: int = 256,
) -> str:
    if value is None and default is not None:
        value = default
    if not isinstance(value, str):
        raise EvaluationInputError(f"评测数据字段 {field} 必须是字符串")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or any(ord(char) < 32 for char in normalized):
        raise EvaluationInputError(f"评测数据字段 {field} 为空、过长或包含控制字符")
    return normalized


def _validate_prepared_manifest(
    dataset_path: Path,
    evaluation_root: Path,
    *,
    sha256: str,
    record_count: int,
) -> str:
    manifest_path = dataset_path.with_suffix(".manifest.json")
    resolved_manifest, _ = strict_existing_path(
        manifest_path,
        (evaluation_root,),
        directory=False,
    )
    try:
        size = resolved_manifest.stat().st_size
    except OSError as exc:
        raise EvaluationInputError("无法读取内置评测集 manifest") from exc
    if not 1 <= size <= 1024 * 1024:
        raise EvaluationInputError("内置评测集 manifest 大小异常")
    with _source_file(resolved_manifest) as source:
        manifest = _loads_json(source.read(1024 * 1024 + 1))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("output"), dict):
        raise EvaluationInputError("内置评测集 manifest 结构无效")
    output = manifest["output"]
    if output.get("sha256") != sha256 or output.get("record_count") != record_count:
        raise EvaluationInputError("内置评测集 JSONL 与 manifest 指纹或记录数不一致")
    return str(resolved_manifest)


def prepare_evaluation_workspace(
    *,
    run_id: UUID,
    generation: int,
    sources: list[DatasetSource],
    dataset_root: Path,
    evaluation_dataset_root: Path,
    evaluation_output_root: Path,
    requested_output_path: Path,
    runtime_root: Path,
) -> EvaluationWorkspace:
    if not 1 <= len(sources) <= MAX_DATASET_FILES:
        raise EvaluationInputError(f"评测数据集数量必须位于 1..{MAX_DATASET_FILES}")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise EvaluationInputError("评测数据集 name 不能重复")

    output_path = prepare_output_directory(requested_output_path, evaluation_output_root, run_id)
    workspace = _derived_runtime_directory(runtime_root, run_id, generation)
    merged_path = workspace / "evaluation.jsonl"
    manifest_path = workspace / "dataset-manifest.json"
    temporary = workspace / f".evaluation.jsonl.{uuid4().hex}.tmp"
    source_manifests: list[dict[str, Any]] = []
    combined_digest = hashlib.sha256()
    total_bytes = 0
    total_records = 0
    resolved_sources: list[tuple[DatasetSource, Path, Path]] = []
    seen_paths: set[Path] = set()

    for source in sorted(sources, key=lambda item: (item.name, str(item.path))):
        resolved, matched_root = strict_existing_path(
            source.path,
            (evaluation_dataset_root, dataset_root),
            directory=False,
        )
        if resolved.suffix.lower() != ".jsonl":
            raise EvaluationInputError(f"评测数据必须是 .jsonl 文件：{source.path}")
        if resolved in seen_paths:
            raise EvaluationInputError("同一评测 JSONL 不能以多个 name 重复计权")
        seen_paths.add(resolved)
        resolved_sources.append((source, resolved, matched_root))

    try:
        with temporary.open("xb") as merged:
            seen_ids: set[str] = set()
            for source, source_path, matched_root in resolved_sources:
                with _source_file(source_path) as input_file:
                    before = os.fstat(input_file.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise EvaluationInputError("评测数据必须是普通文件")
                    if not 1 <= before.st_size <= MAX_DATASET_FILE_BYTES:
                        raise EvaluationInputError("单个评测数据集为空或超过 256 MiB")
                    total_bytes += before.st_size
                    if total_bytes > MAX_DATASET_TOTAL_BYTES:
                        raise EvaluationInputError("评测数据集总大小超过 512 MiB")

                    source_digest = hashlib.sha256()
                    source_records = 0
                    line_number = 0
                    while True:
                        raw_line = input_file.readline(MAX_DATASET_LINE_BYTES + 1)
                        if not raw_line:
                            break
                        line_number += 1
                        if len(raw_line) > MAX_DATASET_LINE_BYTES:
                            raise EvaluationInputError(
                                f"评测数据 {source.name} 第 {line_number} 行超过 1 MiB"
                            )
                        source_digest.update(raw_line)
                        if not raw_line.strip():
                            continue
                        row = _loads_json(raw_line)
                        if not isinstance(row, dict):
                            raise EvaluationInputError("评测 JSONL 每行必须是对象")
                        original_id = _safe_short_text(
                            str(row.get("id", line_number)),
                            "id",
                            # source name 最长 64，保证合并后 `<source>:<id>` 不超过 256。
                            max_length=191,
                        )
                        merged_id = f"{source.name}:{original_id}"
                        if merged_id in seen_ids:
                            raise EvaluationInputError(f"评测样本 ID 重复：{merged_id}")
                        seen_ids.add(merged_id)
                        category = _safe_short_text(
                            row.get("category"),
                            "category",
                            default="default",
                            # source name 最长 64，保证合并后 `<source>/<category>` 不超过 256。
                            max_length=191,
                        )
                        metadata = row.get("metadata", {})
                        if not isinstance(metadata, dict):
                            raise EvaluationInputError("评测数据 metadata 必须是对象")
                        if "openllmops_source" in metadata:
                            raise EvaluationInputError("评测数据 metadata 不能覆盖 openllmops_source")
                        row["id"] = merged_id
                        row["category"] = f"{source.name}/{category}"
                        row["metadata"] = {
                            **metadata,
                            "openllmops_source": {
                                "name": source.name,
                                "original_id": original_id,
                            },
                        }
                        canonical_line = (
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                            + b"\n"
                        )
                        merged.write(canonical_line)
                        combined_digest.update(canonical_line)
                        source_records += 1
                        total_records += 1
                        if total_records > MAX_DATASET_RECORDS:
                            raise EvaluationInputError(f"评测数据集记录数超过 {MAX_DATASET_RECORDS}")

                    after = os.fstat(input_file.fileno())
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise EvaluationInputError("读取期间评测数据文件发生变化")
                if source_records == 0:
                    raise EvaluationInputError(f"评测数据集 {source.name} 不含有效记录")
                fingerprint = source_digest.hexdigest()
                prepared_manifest = None
                if matched_root == evaluation_dataset_root.resolve(strict=True):
                    prepared_manifest = _validate_prepared_manifest(
                        source_path,
                        evaluation_dataset_root,
                        sha256=fingerprint,
                        record_count=source_records,
                    )
                source_manifests.append(
                    {
                        "name": source.name,
                        "path": str(source_path),
                        "sha256": fingerprint,
                        "size_bytes": before.st_size,
                        "record_count": source_records,
                        "prepared_manifest_path": prepared_manifest,
                    }
                )
            if total_records == 0:
                raise EvaluationInputError("评测数据集不包含有效记录")
            merged.flush()
            os.fsync(merged.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, merged_path)
    except EvaluationInputError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationInputError("无法生成确定性评测数据集") from exc
    finally:
        temporary.unlink(missing_ok=True)

    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "combined": {
                "path": str(merged_path),
                "record_count": total_records,
                "sha256": combined_digest.hexdigest(),
                "size_bytes": merged_path.stat().st_size,
            },
            "sources": source_manifests,
        },
        0o400,
    )
    return EvaluationWorkspace(merged_path, manifest_path, output_path)


def _finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationInputError(f"评测报告字段 {field} 必须是数字")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvaluationInputError(f"评测报告字段 {field} 数值无效") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvaluationInputError(f"评测报告字段 {field} 数值无效")
    return result


def _bounded_json_tree(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_REPORT_JSON_NODES or depth > 32:
        raise EvaluationInputError("评测报告结构过大或嵌套过深")
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationInputError("评测报告包含非有限数值")
        return
    if isinstance(value, str):
        if len(value) > 8192:
            raise EvaluationInputError("评测报告字符串字段过长")
        return
    if isinstance(value, list):
        for item in value:
            _bounded_json_tree(item, depth=depth + 1, counter=counter)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise EvaluationInputError("评测报告对象字段名无效")
            _bounded_json_tree(item, depth=depth + 1, counter=counter)
        return
    raise EvaluationInputError("评测报告包含不支持的 JSON 类型")


def load_dataset_manifest_summary(manifest_path: Path) -> tuple[str, int]:
    """只信任 agent 生成的有限字段，用于将容器报告绑定到调度时的数据指纹。"""

    try:
        manifest_stat = manifest_path.lstat()
    except OSError as exc:
        raise EvaluationInputError("评测数据 manifest 不存在") from exc
    if stat.S_ISLNK(manifest_stat.st_mode) or not stat.S_ISREG(manifest_stat.st_mode):
        raise EvaluationInputError("评测数据 manifest 必须是非软链接普通文件")
    if not 1 <= manifest_stat.st_size <= MAX_MANIFEST_BYTES:
        raise EvaluationInputError("评测数据 manifest 大小异常")
    with _source_file(manifest_path) as source:
        manifest = _loads_json(source.read(MAX_MANIFEST_BYTES + 1))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise EvaluationInputError("评测数据 manifest 版本或结构无效")
    combined = manifest.get("combined")
    sources = manifest.get("sources")
    if not isinstance(combined, dict) or not isinstance(sources, list) or not sources:
        raise EvaluationInputError("评测数据 manifest 缺少汇总或来源")
    digest = combined.get("sha256")
    record_count = combined.get("record_count")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise EvaluationInputError("评测数据 manifest 汇总指纹无效")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or not 1 <= record_count <= MAX_DATASET_RECORDS
    ):
        raise EvaluationInputError("评测数据 manifest 汇总记录数无效")
    return digest, record_count


def _metric_summary(report: Any, label: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise EvaluationInputError(f"评测报告缺少 {label} 对象")
    digest = report.get("dataset_sha256")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise EvaluationInputError(f"评测报告 {label}.dataset_sha256 无效")
    model_name = _safe_short_text(report.get("model_name"), f"{label}.model_name")
    template = report.get("template")
    if template not in {"base", "instruct"}:
        raise EvaluationInputError(f"评测报告 {label}.template 无效")
    total = report.get("total")
    correct = report.get("correct")
    invalid = report.get("invalid")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or not 1 <= total <= MAX_DATASET_RECORDS
        or isinstance(correct, bool)
        or not isinstance(correct, int)
        or not 0 <= correct <= total
        or isinstance(invalid, bool)
        or not isinstance(invalid, int)
        or not 0 <= invalid <= total
        or correct + invalid > total
    ):
        raise EvaluationInputError(f"评测报告 {label} 计数字段无效")
    accuracy = _finite_number(report.get("accuracy_percent"), f"{label}.accuracy_percent", minimum=0)
    latency = _finite_number(
        report.get("average_latency_ms"),
        f"{label}.average_latency_ms",
        minimum=0,
    )
    if accuracy > 100:
        raise EvaluationInputError(f"评测报告 {label}.accuracy_percent 超过 100")
    if accuracy != round(correct * 100 / total, 4):
        raise EvaluationInputError(f"评测报告 {label}.accuracy_percent 与计数不一致")
    categories = report.get("categories")
    sample_ids = report.get("sample_ids")
    if not isinstance(categories, list) or len(categories) > MAX_REPORT_CATEGORIES:
        raise EvaluationInputError(f"评测报告 {label}.categories 无效或过多")
    if not isinstance(sample_ids, list) or len(sample_ids) != total:
        raise EvaluationInputError(f"评测报告 {label}.sample_ids 与 total 不一致")
    normalized_sample_ids = [_safe_short_text(sample_id, f"{label}.sample_ids") for sample_id in sample_ids]
    if len(normalized_sample_ids) != len(set(normalized_sample_ids)):
        raise EvaluationInputError(f"评测报告 {label}.sample_ids 包含重复值")
    normalized_categories: list[dict[str, Any]] = []
    category_names: set[str] = set()
    category_total = 0
    category_correct = 0
    category_invalid = 0
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise EvaluationInputError(f"评测报告 {label}.categories[{index}] 不是对象")
        name = _safe_short_text(category.get("category"), f"{label}.categories[{index}].category")
        if name in category_names:
            raise EvaluationInputError(f"评测报告 {label}.categories 名称重复")
        category_names.add(name)
        item_total = category.get("total")
        item_correct = category.get("correct")
        item_invalid = category.get("invalid")
        if (
            isinstance(item_total, bool)
            or not isinstance(item_total, int)
            or not 1 <= item_total <= total
            or isinstance(item_correct, bool)
            or not isinstance(item_correct, int)
            or not 0 <= item_correct <= item_total
            or isinstance(item_invalid, bool)
            or not isinstance(item_invalid, int)
            or not 0 <= item_invalid <= item_total
            or item_correct + item_invalid > item_total
        ):
            raise EvaluationInputError(f"评测报告 {label}.categories[{index}] 计数无效")
        item_accuracy = _finite_number(
            category.get("accuracy_percent"),
            f"{label}.categories[{index}].accuracy_percent",
            minimum=0,
        )
        if item_accuracy > 100:
            raise EvaluationInputError(f"评测报告 {label}.categories[{index}].accuracy_percent 超过 100")
        if item_accuracy != round(item_correct * 100 / item_total, 4):
            raise EvaluationInputError(f"评测报告 {label}.categories[{index}] 正确率与计数不一致")
        category_total += item_total
        category_correct += item_correct
        category_invalid += item_invalid
        normalized_categories.append(
            {
                "category": name,
                "total": item_total,
                "correct": item_correct,
                "invalid": item_invalid,
                "accuracy_percent": item_accuracy,
            }
        )
    if (category_total, category_correct, category_invalid) != (total, correct, invalid):
        raise EvaluationInputError(f"评测报告 {label}.categories 汇总与总计数不一致")
    return {
        "dataset_sha256": digest,
        "model_name": model_name,
        "template": template,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy_percent": accuracy,
        "average_latency_ms": latency,
        "categories": normalized_categories,
        # 仅用于节点内前后一致性校验，返回 metadata 前会移除以限制体积。
        "sample_ids": normalized_sample_ids,
    }


def _comparison_summary(
    raw: Any,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("comparable") is not True:
        raise EvaluationInputError("评测报告前后结果不可比较")
    digest = raw.get("dataset_sha256")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise EvaluationInputError("评测报告 comparison.dataset_sha256 无效")
    baseline_model = _safe_short_text(raw.get("baseline_model"), "comparison.baseline_model")
    candidate_model = _safe_short_text(raw.get("candidate_model"), "comparison.candidate_model")
    baseline_percent = _finite_number(raw.get("baseline_percent"), "comparison.baseline_percent", minimum=0)
    candidate_percent = _finite_number(
        raw.get("candidate_percent"), "comparison.candidate_percent", minimum=0
    )
    point_change = _finite_number(raw.get("percentage_point_change"), "comparison.percentage_point_change")
    relative_raw = raw.get("relative_change_percent")
    relative_change = (
        None if relative_raw is None else _finite_number(relative_raw, "comparison.relative_change_percent")
    )
    reason = raw.get("reason")
    if reason is not None:
        raise EvaluationInputError("可比较的评测报告 comparison.reason 必须为 null")
    expected_delta = round(candidate["accuracy_percent"] - baseline["accuracy_percent"], 4)
    expected_relative = (
        round(expected_delta * 100 / baseline["accuracy_percent"], 4)
        if baseline["accuracy_percent"] != 0
        else None
    )
    if (
        digest != baseline["dataset_sha256"]
        or baseline_model != baseline["model_name"]
        or candidate_model != candidate["model_name"]
        or baseline_percent != baseline["accuracy_percent"]
        or candidate_percent != candidate["accuracy_percent"]
        or point_change != expected_delta
        or relative_change != expected_relative
    ):
        raise EvaluationInputError("评测报告 comparison 与前后汇总不一致")
    raw_changes = raw.get("category_changes")
    if not isinstance(raw_changes, list) or len(raw_changes) > MAX_REPORT_CATEGORIES:
        raise EvaluationInputError("评测报告 comparison.category_changes 无效")
    baseline_categories = {item["category"]: item for item in baseline["categories"]}
    candidate_categories = {item["category"]: item for item in candidate["categories"]}
    if baseline_categories.keys() != candidate_categories.keys():
        raise EvaluationInputError("评测报告前后分类集合不一致")
    for category_name in baseline_categories:
        if baseline_categories[category_name]["total"] != candidate_categories[category_name]["total"]:
            raise EvaluationInputError("评测报告前后同名分类的样本数不一致")
    expected_names = sorted(baseline_categories.keys() & candidate_categories.keys())
    normalized_changes: list[dict[str, Any]] = []
    for index, (change, expected_name) in enumerate(zip(raw_changes, expected_names, strict=False)):
        if not isinstance(change, dict):
            raise EvaluationInputError(f"评测报告 comparison.category_changes[{index}] 不是对象")
        name = _safe_short_text(change.get("category"), f"comparison.category_changes[{index}].category")
        base_value = _finite_number(
            change.get("baseline_percent"),
            f"comparison.category_changes[{index}].baseline_percent",
            minimum=0,
        )
        candidate_value = _finite_number(
            change.get("candidate_percent"),
            f"comparison.category_changes[{index}].candidate_percent",
            minimum=0,
        )
        delta = _finite_number(
            change.get("percentage_point_change"),
            f"comparison.category_changes[{index}].percentage_point_change",
        )
        expected_category_delta = round(
            candidate_categories[expected_name]["accuracy_percent"]
            - baseline_categories[expected_name]["accuracy_percent"],
            4,
        )
        if (
            name != expected_name
            or base_value != baseline_categories[expected_name]["accuracy_percent"]
            or candidate_value != candidate_categories[expected_name]["accuracy_percent"]
            or delta != expected_category_delta
        ):
            raise EvaluationInputError("评测报告分类对比与汇总不一致")
        normalized_changes.append(
            {
                "category": name,
                "baseline_percent": base_value,
                "candidate_percent": candidate_value,
                "percentage_point_change": delta,
            }
        )
    if len(raw_changes) != len(expected_names):
        raise EvaluationInputError("评测报告分类对比数量与汇总不一致")
    return {
        "dataset_sha256": digest,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "baseline_percent": baseline_percent,
        "candidate_percent": candidate_percent,
        "percentage_point_change": point_change,
        "relative_change_percent": relative_change,
        "comparable": True,
        "reason": None,
        "category_changes": normalized_changes,
    }


def load_pair_report_metadata(
    output_path: Path,
    *,
    expected_dataset_sha256: str | None = None,
    expected_total: int | None = None,
    expected_base_template: str | None = None,
    expected_candidate_template: str | None = None,
) -> dict[str, Any]:
    report_path = output_path / "pair-report.json"
    try:
        report_stat = report_path.lstat()
    except OSError as exc:
        raise EvaluationInputError("评测容器成功退出但缺少 pair-report.json") from exc
    if stat.S_ISLNK(report_stat.st_mode) or not stat.S_ISREG(report_stat.st_mode):
        raise EvaluationInputError("pair-report.json 必须是非软链接普通文件")
    if not 1 <= report_stat.st_size <= MAX_REPORT_BYTES:
        raise EvaluationInputError("pair-report.json 为空或超过 128 MiB")
    with _source_file(report_path) as source:
        raw = source.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise EvaluationInputError("pair-report.json 超过 128 MiB")
    pair = _loads_json(raw)
    _bounded_json_tree(pair)
    if not isinstance(pair, dict) or set(pair) != {"baseline", "candidate", "comparison"}:
        raise EvaluationInputError("pair-report.json 顶层结构无效")
    baseline = _metric_summary(pair["baseline"], "baseline")
    candidate = _metric_summary(pair["candidate"], "candidate")
    if baseline["model_name"] != "baseline" or candidate["model_name"] != "candidate":
        raise EvaluationInputError("评测报告模型名与节点固定合同不一致")
    if expected_base_template is not None and baseline["template"] != expected_base_template:
        raise EvaluationInputError("评测报告 baseline 模板与节点启动合同不一致")
    if expected_candidate_template is not None and candidate["template"] != expected_candidate_template:
        raise EvaluationInputError("评测报告 candidate 模板与节点启动合同不一致")
    comparison = _comparison_summary(pair["comparison"], baseline, candidate)
    fingerprints = {
        baseline["dataset_sha256"],
        candidate["dataset_sha256"],
        comparison.get("dataset_sha256"),
    }
    if len(fingerprints) != 1 or baseline["total"] != candidate["total"]:
        raise EvaluationInputError("评测报告前后数据指纹或样本数不一致")
    if baseline["sample_ids"] != candidate["sample_ids"]:
        raise EvaluationInputError("评测报告前后样本 ID 集合不一致")
    if expected_dataset_sha256 is not None and baseline["dataset_sha256"] != expected_dataset_sha256:
        raise EvaluationInputError("评测报告指纹与节点调度的数据版本不一致")
    if expected_total is not None and baseline["total"] != expected_total:
        raise EvaluationInputError("评测报告样本数与节点调度的数据版本不一致")
    baseline_metrics = {key: value for key, value in baseline.items() if key != "sample_ids"}
    candidate_metrics = {key: value for key, value in candidate.items() if key != "sample_ids"}
    metadata: dict[str, Any] = {
        "metrics": {"baseline": baseline_metrics, "candidate": candidate_metrics},
        "comparison": comparison,
        "result_path": str(report_path),
    }
    warnings: list[str] = []
    if baseline["invalid"] == baseline["total"]:
        warnings.append("baseline_all_outputs_invalid")
    if candidate["invalid"] == candidate["total"]:
        warnings.append("candidate_all_outputs_invalid")
    if warnings:
        metadata["warnings"] = warnings
    return metadata
