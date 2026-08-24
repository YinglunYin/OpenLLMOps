"""JSONL 数据集读取、规范化和指纹计算。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Sample, TaskType

MAX_LINE_BYTES = 1_048_576


class DatasetValidationError(ValueError):
    """携带行号的用户可理解校验错误。"""


def _required_text(row: dict[str, Any], name: str, line_number: int) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"第 {line_number} 行字段 {name!r} 必须是非空字符串")
    return value.strip()


def _parse_answers(row: dict[str, Any], line_number: int) -> tuple[str, ...]:
    raw = row.get("answers", row.get("answer"))
    values = raw if isinstance(raw, list) else [raw]
    if not values or not all(isinstance(value, str) and value.strip() for value in values):
        raise DatasetValidationError(f"第 {line_number} 行 answer/answers 必须包含非空字符串")
    return tuple(value.strip() for value in values)


def _parse_choices(row: dict[str, Any], line_number: int) -> tuple[tuple[str, str], ...]:
    raw = row.get("choices")
    if isinstance(raw, dict):
        pairs = [(str(label), text) for label, text in raw.items()]
    elif isinstance(raw, list):
        # 列表形式自动生成 A、B、C……标签，最多 26 项。
        if len(raw) > 26:
            raise DatasetValidationError(f"第 {line_number} 行 choices 最多支持 26 项")
        pairs = [(chr(65 + index), text) for index, text in enumerate(raw)]
    else:
        raise DatasetValidationError(f"第 {line_number} 行选择题必须提供 choices")

    normalized: list[tuple[str, str]] = []
    for label, text in pairs:
        if not isinstance(text, str) or not text.strip():
            raise DatasetValidationError(f"第 {line_number} 行 choices 包含空选项")
        normalized.append((label.strip().upper(), text.strip()))
    if len(normalized) < 2 or len({label for label, _ in normalized}) != len(normalized):
        raise DatasetValidationError(f"第 {line_number} 行 choices 标签重复或少于两项")
    return tuple(normalized)


def parse_row(row: dict[str, Any], line_number: int) -> Sample:
    """把宽松的用户 JSON 规范化为稳定领域对象。"""

    try:
        task_type = TaskType(row.get("task_type", TaskType.MULTIPLE_CHOICE))
    except ValueError as exc:
        raise DatasetValidationError(f"第 {line_number} 行 task_type 不受支持") from exc

    sample_id = str(row.get("id", line_number)).strip()
    if not sample_id:
        raise DatasetValidationError(f"第 {line_number} 行 id 不能为空")

    choices = _parse_choices(row, line_number) if task_type == TaskType.MULTIPLE_CHOICE else ()
    answers = _parse_answers(row, line_number)
    if task_type == TaskType.MULTIPLE_CHOICE:
        labels = {label for label, _ in choices}
        if answers[0].upper() not in labels:
            raise DatasetValidationError(f"第 {line_number} 行标准答案不在 choices 标签中")
        answers = (answers[0].upper(),)

    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetValidationError(f"第 {line_number} 行 metadata 必须是对象")

    return Sample(
        sample_id=sample_id,
        task_type=task_type,
        question=_required_text(row, "question", line_number),
        answers=answers,
        category=str(row.get("category", "default")).strip() or "default",
        choices=choices,
        context=str(row["context"]) if row.get("context") is not None else None,
        metadata=metadata,
    )


def load_jsonl(path: Path) -> tuple[list[Sample], str]:
    """流式读取并对原始字节计算指纹，避免同名数据集被误当作同一版本。"""

    digest = hashlib.sha256()
    samples: list[Sample] = []
    seen_ids: set[str] = set()

    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            if len(raw_line) > MAX_LINE_BYTES:
                raise DatasetValidationError(f"第 {line_number} 行超过 {MAX_LINE_BYTES} 字节限制")
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(f"第 {line_number} 行不是有效 UTF-8 JSON") from exc
            if not isinstance(row, dict):
                raise DatasetValidationError(f"第 {line_number} 行必须是 JSON 对象")
            sample = parse_row(row, line_number)
            if sample.sample_id in seen_ids:
                raise DatasetValidationError(f"第 {line_number} 行样本 ID 重复: {sample.sample_id}")
            seen_ids.add(sample.sample_id)
            samples.append(sample)

    if not samples:
        raise DatasetValidationError("数据集不包含有效样本")
    return samples, digest.hexdigest()


def iter_prompts(samples: Iterable[Sample], template: str) -> Iterable[tuple[Sample, str]]:
    for sample in samples:
        yield sample, render_prompt(sample, template)


def render_prompt(sample: Sample, template: str) -> str:
    """Base 与 Instruct 模型使用明确模板，模板名会写入评测报告。"""

    context = f"材料：{sample.context}\n" if sample.context else ""
    if sample.task_type == TaskType.MULTIPLE_CHOICE:
        options = "\n".join(f"{label}. {text}" for label, text in sample.choices)
        body = f"{context}问题：{sample.question}\n{options}\n只输出正确选项的字母。"
    elif sample.task_type == TaskType.CLASSIFICATION:
        body = f"{context}文本：{sample.question}\n只输出分类标签。"
    else:
        body = f"{context}问题：{sample.question}\n请给出简短、直接的答案。"

    if template == "base":
        return f"题目：\n{body}\n答案："
    if template == "instruct":
        return f"你是严谨的中文考试助手。\n{body}"
    raise DatasetValidationError(f"未知模板: {template}")
