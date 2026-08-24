from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_EVALUATION_RECORDS = 200_000
MAX_EVALUATION_CATEGORIES = 4_096


class StrictEvaluationModel(BaseModel):
    """node-agent 评测产物的严格信任边界。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class EmptyEvaluationResult(StrictEvaluationModel):
    """未完成任务保持 `{}`，兼容现有前端且不接受任意未知字段。"""


class EvaluationCategoryMetric(StrictEvaluationModel):
    category: str = Field(min_length=1, max_length=256)
    total: int = Field(ge=1, le=MAX_EVALUATION_RECORDS)
    correct: int = Field(ge=0)
    invalid: int = Field(ge=0)
    accuracy_percent: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_counts(self) -> EvaluationCategoryMetric:
        if self.correct > self.total or self.invalid > self.total or self.correct + self.invalid > self.total:
            raise ValueError("评测分类的 correct/invalid 不能超过 total")
        expected = round(self.correct * 100 / self.total, 4)
        if abs(self.accuracy_percent - expected) > 0.0001:
            raise ValueError("评测分类正确率与计数不一致")
        return self


class EvaluationMetricSummary(StrictEvaluationModel):
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str = Field(min_length=1, max_length=256)
    template: Literal["base", "instruct"]
    total: int = Field(ge=1, le=MAX_EVALUATION_RECORDS)
    correct: int = Field(ge=0)
    invalid: int = Field(ge=0)
    accuracy_percent: float = Field(ge=0, le=100)
    average_latency_ms: float = Field(ge=0)
    categories: list[EvaluationCategoryMetric] = Field(
        min_length=1,
        max_length=MAX_EVALUATION_CATEGORIES,
    )

    @model_validator(mode="after")
    def validate_aggregate(self) -> EvaluationMetricSummary:
        if self.correct > self.total or self.invalid > self.total or self.correct + self.invalid > self.total:
            raise ValueError("评测 correct/invalid 不能超过 total")
        if len({item.category for item in self.categories}) != len(self.categories):
            raise ValueError("评测分类名称不能重复")
        if sum(item.total for item in self.categories) != self.total:
            raise ValueError("评测分类 total 与总计不一致")
        if sum(item.correct for item in self.categories) != self.correct:
            raise ValueError("评测分类 correct 与总计不一致")
        if sum(item.invalid for item in self.categories) != self.invalid:
            raise ValueError("评测分类 invalid 与总计不一致")
        expected = round(self.correct * 100 / self.total, 4)
        if abs(self.accuracy_percent - expected) > 0.0001:
            raise ValueError("评测正确率与计数不一致")
        return self


class EvaluationMetrics(StrictEvaluationModel):
    baseline: EvaluationMetricSummary
    candidate: EvaluationMetricSummary

    @model_validator(mode="after")
    def validate_comparable_inputs(self) -> EvaluationMetrics:
        if self.baseline.dataset_sha256 != self.candidate.dataset_sha256:
            raise ValueError("基线与候选评测数据指纹不一致")
        if self.baseline.total != self.candidate.total:
            raise ValueError("基线与候选评测样本数不一致")
        return self


class EvaluationCategoryChange(StrictEvaluationModel):
    category: str = Field(min_length=1, max_length=256)
    baseline_percent: float = Field(ge=0, le=100)
    candidate_percent: float = Field(ge=0, le=100)
    percentage_point_change: float = Field(ge=-100, le=100)

    @model_validator(mode="after")
    def validate_delta(self) -> EvaluationCategoryChange:
        expected = round(self.candidate_percent - self.baseline_percent, 4)
        if abs(self.percentage_point_change - expected) > 0.0001:
            raise ValueError("评测分类百分点变化与前后正确率不一致")
        return self


class EvaluationComparison(StrictEvaluationModel):
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_model: str = Field(min_length=1, max_length=256)
    candidate_model: str = Field(min_length=1, max_length=256)
    baseline_percent: float = Field(ge=0, le=100)
    candidate_percent: float = Field(ge=0, le=100)
    percentage_point_change: float = Field(ge=-100, le=100)
    relative_change_percent: float | None
    comparable: Literal[True]
    reason: None
    category_changes: list[EvaluationCategoryChange] = Field(
        min_length=1,
        max_length=MAX_EVALUATION_CATEGORIES,
    )

    @model_validator(mode="after")
    def validate_delta(self) -> EvaluationComparison:
        if len({item.category for item in self.category_changes}) != len(self.category_changes):
            raise ValueError("评测分类变化名称不能重复")
        expected = round(self.candidate_percent - self.baseline_percent, 4)
        if abs(self.percentage_point_change - expected) > 0.0001:
            raise ValueError("评测百分点变化与前后正确率不一致")
        if self.baseline_percent == 0:
            if self.relative_change_percent is not None:
                raise ValueError("基线正确率为零时相对变化必须为空")
        else:
            expected_relative = round(expected * 100 / self.baseline_percent, 4)
            if (
                self.relative_change_percent is None
                or abs(self.relative_change_percent - expected_relative) > 0.0001
            ):
                raise ValueError("评测相对变化与前后正确率不一致")
        return self


EvaluationWarning = Literal[
    "baseline_all_outputs_invalid",
    "candidate_all_outputs_invalid",
]


class EvaluationSuccessMetadata(StrictEvaluationModel):
    metrics: EvaluationMetrics
    comparison: EvaluationComparison
    result_path: str = Field(min_length=1, max_length=1024)
    dataset_manifest_path: str = Field(min_length=1, max_length=1024)
    warnings: list[EvaluationWarning] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_report_consistency(self) -> EvaluationSuccessMetadata:
        baseline = self.metrics.baseline
        candidate = self.metrics.candidate
        comparison = self.comparison
        if baseline.model_name != "baseline" or candidate.model_name != "candidate":
            raise ValueError("评测报告模型名与节点固定合同不一致")
        if comparison.dataset_sha256 != baseline.dataset_sha256:
            raise ValueError("comparison 与 metrics 的数据指纹不一致")
        if comparison.baseline_model != baseline.model_name:
            raise ValueError("comparison 与 baseline 模型名不一致")
        if comparison.candidate_model != candidate.model_name:
            raise ValueError("comparison 与 candidate 模型名不一致")
        if comparison.baseline_percent != baseline.accuracy_percent:
            raise ValueError("comparison 与 baseline 正确率不一致")
        if comparison.candidate_percent != candidate.accuracy_percent:
            raise ValueError("comparison 与 candidate 正确率不一致")
        baseline_categories = {item.category for item in baseline.categories}
        candidate_categories = {item.category for item in candidate.categories}
        comparison_categories = {item.category for item in comparison.category_changes}
        if baseline_categories != candidate_categories:
            raise ValueError("基线与候选评测分类集合不一致")
        if comparison_categories != baseline_categories:
            raise ValueError("comparison 分类集合与 metrics 不一致")
        baseline_by_category = {item.category: item for item in baseline.categories}
        candidate_by_category = {item.category: item for item in candidate.categories}
        for change in comparison.category_changes:
            if change.baseline_percent != baseline_by_category[change.category].accuracy_percent:
                raise ValueError("comparison 分类基线正确率与 metrics 不一致")
            if change.candidate_percent != candidate_by_category[change.category].accuracy_percent:
                raise ValueError("comparison 分类候选正确率与 metrics 不一致")
        expected_warnings = {
            warning
            for warning, metric in (
                ("baseline_all_outputs_invalid", baseline),
                ("candidate_all_outputs_invalid", candidate),
            )
            if metric.invalid == metric.total
        }
        if set(self.warnings) != expected_warnings:
            raise ValueError("评测 warnings 与 invalid 计数不一致")
        return self
