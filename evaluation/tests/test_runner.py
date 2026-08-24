from __future__ import annotations

import pytest

from openllmops_eval.client import Generation
from openllmops_eval.models import Sample, TaskType
from openllmops_eval.runner import EvaluationRequestError, evaluate


def _sample() -> Sample:
    return Sample(
        sample_id="sample-1",
        task_type=TaskType.MULTIPLE_CHOICE,
        question="示例题",
        answers=("A",),
        category="general",
        choices=(("A", "甲"), ("B", "乙")),
    )


class _FailingClient:
    async def generate(self, **_: object) -> Generation:
        raise ConnectionError("service stopped")


class _InvalidAnswerClient:
    async def generate(self, **_: object) -> Generation:
        return Generation("无法确定 A 或 B", 12.5)


@pytest.mark.asyncio
async def test_transport_failure_aborts_evaluation_instead_of_becoming_zero_percent() -> None:
    with pytest.raises(EvaluationRequestError, match="ConnectionError"):
        await evaluate(
            client=_FailingClient(),  # type: ignore[arg-type]
            samples=[_sample()],
            dataset_sha256="a" * 64,
            model_name="baseline",
            template="base",
        )


@pytest.mark.asyncio
async def test_unparseable_model_answer_remains_a_quantified_invalid_output() -> None:
    report = await evaluate(
        client=_InvalidAnswerClient(),  # type: ignore[arg-type]
        samples=[_sample()],
        dataset_sha256="a" * 64,
        model_name="baseline",
        template="base",
    )

    assert report.total == 1
    assert report.correct == 0
    assert report.invalid == 1
    assert report.average_latency_ms == 12.5
