"""确定性答案解析、聚合与前后模型对比。"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from .models import (
    CategoryMetric,
    ComparisonReport,
    EvaluationReport,
    Sample,
    SampleResult,
    TaskType,
)


def normalize_text(value: str) -> str:
    """统一全半角、大小写、空白和常见中文标点，用于严格但稳定的匹配。"""

    value = unicodedata.normalize("NFKC", value).strip().casefold()
    value = re.sub(r"[\s\u3000]+", "", value)
    return re.sub(r"[，。！？；：、,.!?;:'\"`()\[\]{}<>《》]", "", value)


def _extract_choice(prediction: str, labels: set[str]) -> str | None:
    upper = unicodedata.normalize("NFKC", prediction).upper().strip()
    # 优先接受纯字母或“答案：A”，避免解释文本里提到多个选项时误判。
    direct = re.fullmatch(r"(?:答案|选项|选择)?\s*[:：]?\s*([A-Z])\s*[.。]?", upper)
    if direct and direct.group(1) in labels:
        return direct.group(1)
    leading = re.match(r"^(?:答案|选项|选择)?\s*[:：]?\s*([A-Z])(?:\b|[.、，。:：])", upper)
    return leading.group(1) if leading and leading.group(1) in labels else None


def score_answer(sample: Sample, prediction: str, latency_ms: float = 0.0) -> SampleResult:
    """解析失败与答错分开计数，防止格式问题被掩盖。"""

    if sample.task_type == TaskType.MULTIPLE_CHOICE:
        parsed = _extract_choice(prediction, {label for label, _ in sample.choices})
        valid = parsed is not None
        correct = valid and parsed == sample.answers[0]
    else:
        parsed = normalize_text(prediction)
        valid = bool(parsed)
        correct = valid and parsed in {normalize_text(answer) for answer in sample.answers}

    return SampleResult(
        sample_id=sample.sample_id,
        category=sample.category,
        prediction=prediction,
        expected=sample.answers,
        correct=correct,
        valid=valid,
        latency_ms=max(0.0, latency_ms),
    )


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator * 100 / denominator, 4) if denominator else 0.0


def aggregate_results(
    results: Iterable[SampleResult], *, dataset_sha256: str, model_name: str, template: str
) -> EvaluationReport:
    materialized = list(results)
    if not materialized:
        raise ValueError("不能聚合空评测结果")

    grouped: dict[str, list[SampleResult]] = defaultdict(list)
    for result in materialized:
        grouped[result.category].append(result)

    categories = tuple(
        CategoryMetric(
            category=name,
            total=len(items),
            correct=sum(item.correct for item in items),
            invalid=sum(not item.valid for item in items),
            accuracy_percent=_percent(sum(item.correct for item in items), len(items)),
        )
        for name, items in sorted(grouped.items())
    )
    correct = sum(result.correct for result in materialized)
    return EvaluationReport(
        dataset_sha256=dataset_sha256,
        model_name=model_name,
        template=template,
        total=len(materialized),
        correct=correct,
        invalid=sum(not result.valid for result in materialized),
        accuracy_percent=_percent(correct, len(materialized)),
        average_latency_ms=round(
            sum(result.latency_ms for result in materialized) / len(materialized), 3
        ),
        categories=categories,
        sample_ids=tuple(sorted(result.sample_id for result in materialized)),
    )


def compare_reports(baseline: EvaluationReport, candidate: EvaluationReport) -> ComparisonReport:
    """仅对完全相同的数据集与样本集合生成有效对比。"""

    reason: str | None = None
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        reason = "数据集版本指纹不同"
    elif baseline.sample_ids != candidate.sample_ids:
        reason = "参与评测的样本集合不同"

    delta = round(candidate.accuracy_percent - baseline.accuracy_percent, 4)
    relative = (
        round(delta * 100 / baseline.accuracy_percent, 4)
        if baseline.accuracy_percent != 0
        else None
    )
    base_categories = {metric.category: metric for metric in baseline.categories}
    candidate_categories = {metric.category: metric for metric in candidate.categories}
    category_changes = tuple(
        {
            "category": category,
            "baseline_percent": base_categories[category].accuracy_percent,
            "candidate_percent": candidate_categories[category].accuracy_percent,
            "percentage_point_change": round(
                candidate_categories[category].accuracy_percent
                - base_categories[category].accuracy_percent,
                4,
            ),
        }
        for category in sorted(base_categories.keys() & candidate_categories.keys())
    )

    return ComparisonReport(
        dataset_sha256=baseline.dataset_sha256,
        baseline_model=baseline.model_name,
        candidate_model=candidate.model_name,
        baseline_percent=baseline.accuracy_percent,
        candidate_percent=candidate.accuracy_percent,
        percentage_point_change=delta,
        relative_change_percent=relative,
        comparable=reason is None,
        reason=reason,
        category_changes=category_changes,
    )
