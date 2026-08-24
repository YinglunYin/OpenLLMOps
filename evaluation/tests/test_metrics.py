from __future__ import annotations

from openllmops_eval.metrics import aggregate_results, compare_reports, score_answer
from openllmops_eval.models import Sample, TaskType


def _sample(sample_id: str, answer: str = "A", category: str = "general") -> Sample:
    return Sample(
        sample_id=sample_id,
        task_type=TaskType.MULTIPLE_CHOICE,
        question="示例题",
        answers=(answer,),
        category=category,
        choices=(("A", "甲"), ("B", "乙")),
    )


def test_choice_parser_distinguishes_invalid_output() -> None:
    correct = score_answer(_sample("1"), "答案：A")
    invalid = score_answer(_sample("2"), "我认为 A 或 B 都可能")

    assert correct.correct is True
    assert invalid.valid is False
    assert invalid.correct is False


def test_comparison_uses_percentage_points_and_relative_change() -> None:
    baseline_results = [score_answer(_sample("1"), "A"), score_answer(_sample("2"), "B")]
    candidate_results = [score_answer(_sample("1"), "A"), score_answer(_sample("2"), "A")]
    baseline = aggregate_results(
        baseline_results, dataset_sha256="same", model_name="base", template="base"
    )
    candidate = aggregate_results(
        candidate_results, dataset_sha256="same", model_name="fine-tuned", template="base"
    )

    comparison = compare_reports(baseline, candidate)

    assert comparison.comparable is True
    assert comparison.baseline_percent == 50.0
    assert comparison.candidate_percent == 100.0
    assert comparison.percentage_point_change == 50.0
    assert comparison.relative_change_percent == 100.0


def test_different_dataset_fingerprint_is_not_comparable() -> None:
    result = [score_answer(_sample("1"), "A")]
    baseline = aggregate_results(result, dataset_sha256="a", model_name="base", template="base")
    candidate = aggregate_results(result, dataset_sha256="b", model_name="candidate", template="base")

    comparison = compare_reports(baseline, candidate)

    assert comparison.comparable is False
    assert comparison.reason == "数据集版本指纹不同"
