"""评测领域对象。执行器保持轻依赖，便于封装成短生命周期容器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    """MVP 支持的三类可量化任务。"""

    MULTIPLE_CHOICE = "multiple_choice"
    CLASSIFICATION = "classification"
    SHORT_QA = "short_qa"


@dataclass(frozen=True, slots=True)
class Sample:
    """规范化后的单条样本。

    ``answers`` 使用列表是为了兼容短问答的多个同义标准答案；选择题和分类
    仍只需提供一个元素。样本 ID 必须在数据集版本内稳定，才能可靠比较前后模型。
    """

    sample_id: str
    task_type: TaskType
    question: str
    answers: tuple[str, ...]
    category: str = "default"
    choices: tuple[tuple[str, str], ...] = ()
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleResult:
    """保存原始输出和结构化判分，便于追溯解析失败。"""

    sample_id: str
    category: str
    prediction: str
    expected: tuple[str, ...]
    correct: bool
    valid: bool
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CategoryMetric:
    category: str
    total: int
    correct: int
    invalid: int
    accuracy_percent: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """一次模型、数据集版本和固定参数组合对应一份不可变报告。"""

    dataset_sha256: str
    model_name: str
    template: str
    total: int
    correct: int
    invalid: int
    accuracy_percent: float
    average_latency_ms: float
    categories: tuple[CategoryMetric, ...]
    sample_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """正确率用百分比展示，差值用百分点，避免两个概念混淆。"""

    dataset_sha256: str
    baseline_model: str
    candidate_model: str
    baseline_percent: float
    candidate_percent: float
    percentage_point_change: float
    relative_change_percent: float | None
    comparable: bool
    reason: str | None
    category_changes: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
