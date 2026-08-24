"""并发受控的评测运行器。"""

from __future__ import annotations

import asyncio

from .client import CompatibleClient
from .dataset import iter_prompts
from .metrics import aggregate_results
from .models import EvaluationReport, Sample, SampleResult


async def evaluate(
    *,
    client: CompatibleClient,
    samples: list[Sample],
    dataset_sha256: str,
    model_name: str,
    template: str,
    concurrency: int = 4,
    max_tokens: int = 32,
) -> EvaluationReport:
    if not 1 <= concurrency <= 64:
        raise ValueError("concurrency 必须在 1 到 64 之间")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(sample: Sample, prompt: str) -> SampleResult:
        async with semaphore:
            try:
                generated = await client.generate(
                    model=model_name,
                    prompt=prompt,
                    template=template,
                    max_tokens=max_tokens,
                )
                from .metrics import score_answer

                return score_answer(sample, generated.text, generated.latency_ms)
            except Exception as exc:  # noqa: BLE001 - 单样本失败不能中止整批评测
                return SampleResult(
                    sample_id=sample.sample_id,
                    category=sample.category,
                    prediction="",
                    expected=sample.answers,
                    correct=False,
                    valid=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

    tasks = [run_one(sample, prompt) for sample, prompt in iter_prompts(samples, template)]
    results = await asyncio.gather(*tasks)
    return aggregate_results(
        results,
        dataset_sha256=dataset_sha256,
        model_name=model_name,
        template=template,
    )

