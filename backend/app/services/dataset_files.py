import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

from app.models.enums import DatasetType

MAX_DATASET_BYTES = 5 * 1024 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_REPORTED_ERRORS = 20


def _record_error(errors: list[dict[str, Any]], line: int, message: str) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append({"line": line, "message": message})


def _validate_shape(item: Any, dataset_type: DatasetType) -> str | None:
    if not isinstance(item, dict):
        return "每行必须是 JSON 对象"
    if dataset_type == DatasetType.CPT:
        if not isinstance(item.get("text"), str) and not isinstance(item.get("content"), str):
            return "CPT 数据至少需要字符串字段 text 或 content"
    elif dataset_type == DatasetType.SFT:
        has_messages = isinstance(item.get("messages"), list) or isinstance(item.get("conversations"), list)
        has_instruction = isinstance(item.get("instruction"), str) and isinstance(item.get("output"), str)
        if not has_messages and not has_instruction:
            return "SFT 数据需要 messages/conversations，或 instruction + output"
    else:
        has_qa = isinstance(item.get("question"), str) and "answer" in item
        has_classification = "input" in item and "label" in item
        if not has_qa and not has_classification:
            return "评测数据需要 question + answer，或 input + label"
    return None


def validate_and_store_jsonl(
    source: BinaryIO,
    temporary_path: Path,
    final_path: Path,
    dataset_type: DatasetType,
) -> tuple[int, int, str, list[dict[str, Any]], dict[str, Any]]:
    """边读边校验并计算摘要，避免大数据集整体载入内存。"""

    total_bytes = 0
    record_count = 0
    errors: list[dict[str, Any]] = []
    field_names: set[str] = set()
    digest = hashlib.sha256()

    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("wb") as target:
            for line_number, raw_line in enumerate(source, start=1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_DATASET_BYTES:
                    raise ValueError("数据集超过 5 GiB 限制")
                if len(raw_line) > MAX_LINE_BYTES:
                    _record_error(errors, line_number, "单行超过 16 MiB 限制")
                    continue

                target.write(raw_line)
                digest.update(raw_line)
                if not raw_line.strip():
                    _record_error(errors, line_number, "不允许空行")
                    continue
                try:
                    item = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    _record_error(errors, line_number, f"JSON 解析失败：{exc}")
                    continue

                record_count += 1
                if isinstance(item, dict):
                    field_names.update(str(key) for key in item)
                shape_error = _validate_shape(item, dataset_type)
                if shape_error:
                    _record_error(errors, line_number, shape_error)

        if record_count == 0:
            _record_error(errors, 0, "数据集没有有效记录")
        if errors:
            raise ValueError(json.dumps(errors, ensure_ascii=False))
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return (
        record_count,
        total_bytes,
        digest.hexdigest(),
        errors,
        {"fields": sorted(field_names), "format": "jsonl"},
    )


def preview_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for raw_line in source:
            if raw_line.strip():
                item = json.loads(raw_line)
                records.append(item if isinstance(item, dict) else {"value": item})
                if len(records) >= limit:
                    break
    return records


def ensure_path_within(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("文件路径不在系统受控目录内")
    return resolved_path
