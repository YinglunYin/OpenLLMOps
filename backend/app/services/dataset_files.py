import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, BinaryIO

from openllmops_eval.dataset import MAX_LINE_BYTES as MAX_EVALUATION_LINE_BYTES
from openllmops_eval.dataset import DatasetValidationError
from openllmops_eval.dataset import parse_row as parse_evaluation_row

from app.models.enums import DatasetType

MAX_DATASET_BYTES = 5 * 1024 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_REPORTED_ERRORS = 20
MAX_SCHEMA_FIELDS = 256
MAX_FIELD_NAME_LENGTH = 128
MAX_TRAINING_RECORDS = 10_000_000
MAX_EVALUATION_SOURCE_FIELD_LENGTH = 191
MAX_EVALUATION_DATASET_BYTES = 256 * 1024 * 1024
MAX_EVALUATION_RECORDS = 200_000
STALE_UPLOAD_PART_AGE_SECONDS = 24 * 60 * 60
UPLOAD_PART_NAME_PATTERN = re.compile(
    r"^\.[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.jsonl\.part$"
)


def cleanup_stale_upload_parts(
    dataset_root: Path,
    *,
    older_than_seconds: int = STALE_UPLOAD_PART_AGE_SECONDS,
    now: float | None = None,
) -> int:
    """清理进程被强杀后遗留的严格命名上传临时文件。

    只处理本系统 UUIDv4 命名、普通且非符号链接的文件，并设置 24 小时安全时龄，
    避免误删管理员文件或另一个仍在上传的实例。
    """

    if older_than_seconds < 0:
        raise ValueError("临时文件安全时龄不能为负数")
    cutoff = (time.time() if now is None else now) - older_than_seconds
    removed = 0
    try:
        entries = list(os.scandir(dataset_root))
    except FileNotFoundError:
        return 0
    for entry in entries:
        if not UPLOAD_PART_NAME_PATTERN.fullmatch(entry.name):
            continue
        try:
            file_stat = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode) or entry.is_symlink() or file_stat.st_mtime > cutoff:
                continue
            Path(entry.path).unlink()
            removed += 1
        except (FileNotFoundError, PermissionError, OSError):
            # 并发清理或单个损坏目录项不应阻塞控制面启动。
            continue
    return removed


def _record_error(errors: list[dict[str, Any]], line: int, message: str) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append({"line": line, "message": message})


def _validate_evaluation_source_text(value: str, field: str) -> str | None:
    if (
        not value
        or len(value) > MAX_EVALUATION_SOURCE_FIELD_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        return f"评测字段 {field} 必须非空、不含控制字符且不超过 {MAX_EVALUATION_SOURCE_FIELD_LENGTH} 字符"
    return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限数值：{value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象包含重复字段：{key}")
        result[key] = value
    return result


def validate_training_record(
    item: Any,
    dataset_type: DatasetType,
) -> tuple[str | None, str | None]:
    """验证单条训练数据并返回 Agent dataset_info 所需的唯一格式。"""

    if not isinstance(item, dict):
        return "每行必须是 JSON 对象", None
    if dataset_type == DatasetType.CPT:
        has_text = isinstance(item.get("text"), str) and bool(item["text"].strip())
        has_content = isinstance(item.get("content"), str) and bool(item["content"].strip())
        if has_text == has_content:
            return "CPT 每行必须且只能使用非空 text 或 content 之一", None
        return None, "cpt_text" if has_text else "cpt_content"
    elif dataset_type == DatasetType.SFT:
        formats = [
            name
            for name, present in (
                ("sft_messages", isinstance(item.get("messages"), list)),
                ("sft_conversations", isinstance(item.get("conversations"), list)),
                (
                    "sft_alpaca",
                    isinstance(item.get("instruction"), str) and isinstance(item.get("output"), str),
                ),
            )
            if present
        ]
        if len(formats) != 1:
            return "SFT 每行必须且只能使用 messages、conversations 或 instruction+output 一种格式", None
        record_format = formats[0]
        if record_format == "sft_alpaca":
            if not item["instruction"].strip() or not item["output"].strip():
                return "Alpaca instruction/output 必须是非空字符串", None
            if "input" in item and not isinstance(item["input"], str):
                return "Alpaca input 如存在必须是字符串", None
        else:
            field = "messages" if record_format == "sft_messages" else "conversations"
            role_key, content_key = ("role", "content") if field == "messages" else ("from", "value")
            messages = item[field]
            if not messages:
                return f"SFT {field} 不能为空", None
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get(role_key), str)
                    or not message[role_key].strip()
                    or not isinstance(message.get(content_key), str)
                    or not message[content_key].strip()
                ):
                    return f"SFT {field} 每项必须包含非空 {role_key}/{content_key}", None
        return None, record_format
    return None, None


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
    evaluation_sample_ids: set[str] = set()
    training_record_format: str | None = None
    digest = hashlib.sha256()

    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("wb") as target:
            line_number = 0
            byte_limit = (
                MAX_EVALUATION_DATASET_BYTES if dataset_type == DatasetType.EVALUATION else MAX_DATASET_BYTES
            )
            line_limit = (
                MAX_EVALUATION_LINE_BYTES if dataset_type == DatasetType.EVALUATION else MAX_LINE_BYTES
            )
            while True:
                # BinaryIO 的普通迭代会先把整条物理行读入内存，攻击者可用单条数 GiB
                # JSONL 触发 OOM。每次最多读取“业务上限 + 1”字节，超长行再分块排空。
                raw_line = source.readline(line_limit + 1)
                if not raw_line:
                    break
                line_number += 1
                total_bytes += len(raw_line)
                if total_bytes > byte_limit:
                    raise ValueError(f"数据集超过 {byte_limit} 字节限制")
                if len(raw_line) > line_limit:
                    # readline(size) 截断后必须排空到下一换行，否则一条恶意长行会被
                    # 错当成多条记录；排空本身同样受单次块大小和总文件上限约束。
                    while not raw_line.endswith(b"\n"):
                        drained = source.readline(min(1024 * 1024, line_limit + 1))
                        if not drained:
                            break
                        total_bytes += len(drained)
                        if total_bytes > byte_limit:
                            raise ValueError(f"数据集超过 {byte_limit} 字节限制")
                        raw_line = drained
                    _record_error(errors, line_number, f"单行超过 {line_limit} 字节限制")
                    continue

                target.write(raw_line)
                digest.update(raw_line)
                if not raw_line.strip():
                    # 与评测执行器 load_jsonl 一致：评测集允许分隔空行，训练集仍拒绝。
                    if dataset_type != DatasetType.EVALUATION:
                        _record_error(errors, line_number, "不允许空行")
                    continue
                try:
                    item = json.loads(
                        raw_line,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_unique_json_object,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    _record_error(errors, line_number, f"JSON 解析失败：{exc}")
                    continue

                record_count += 1
                if dataset_type == DatasetType.EVALUATION and record_count > MAX_EVALUATION_RECORDS:
                    raise ValueError("评测数据集有效记录数超过 200000")
                if dataset_type != DatasetType.EVALUATION and record_count > MAX_TRAINING_RECORDS:
                    raise ValueError(f"训练数据集有效记录数超过 {MAX_TRAINING_RECORDS}")
                if isinstance(item, dict):
                    for key in item:
                        if len(key) > MAX_FIELD_NAME_LENGTH:
                            _record_error(
                                errors,
                                line_number,
                                f"字段名超过 {MAX_FIELD_NAME_LENGTH} 字符限制",
                            )
                            continue
                        if key not in field_names and len(field_names) >= MAX_SCHEMA_FIELDS:
                            _record_error(
                                errors,
                                line_number,
                                f"全文件字段种类超过 {MAX_SCHEMA_FIELDS} 限制",
                            )
                            continue
                        field_names.add(key)
                if dataset_type == DatasetType.EVALUATION and isinstance(item, dict):
                    try:
                        sample = parse_evaluation_row(item, line_number)
                    except DatasetValidationError as exc:
                        _record_error(errors, line_number, str(exc))
                    else:
                        if not isinstance(item.get("category", "default"), str):
                            _record_error(errors, line_number, "评测字段 category 必须是字符串")
                        metadata = item.get("metadata", {})
                        if isinstance(metadata, dict) and "openllmops_source" in metadata:
                            _record_error(
                                errors,
                                line_number,
                                "评测 metadata 不能包含保留字段 openllmops_source",
                            )
                        for field, value in (
                            ("id", sample.sample_id),
                            ("category", sample.category),
                        ):
                            if source_error := _validate_evaluation_source_text(value, field):
                                _record_error(errors, line_number, source_error)
                        if sample.sample_id in evaluation_sample_ids:
                            _record_error(errors, line_number, f"样本 ID 重复: {sample.sample_id}")
                        evaluation_sample_ids.add(sample.sample_id)
                else:
                    shape_error, record_format = validate_training_record(item, dataset_type)
                    if shape_error:
                        _record_error(errors, line_number, shape_error)
                    elif record_format is not None:
                        if training_record_format is None:
                            training_record_format = record_format
                        elif training_record_format != record_format:
                            _record_error(
                                errors,
                                line_number,
                                f"训练数据格式必须全文件一致，首条为 {training_record_format}",
                            )

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
        {
            "fields": sorted(field_names),
            "format": "jsonl",
            **({"record_format": training_record_format} if training_record_format else {}),
        },
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
