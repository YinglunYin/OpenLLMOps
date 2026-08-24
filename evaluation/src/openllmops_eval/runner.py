"""并发受控的评测运行器。"""

from __future__ import annotations

import asyncio

from .client import CompatibleClient
from .dataset import iter_prompts
from .metrics import aggregate_results
from .models import EvaluationReport, Sample, SampleResult


class EvaluationRequestError(RuntimeError):
    """模型服务或 OpenAI Compatible 响应失败，不能计为模型答错。"""


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
            except Exception as exc:
                # 连接失败、HTTP 错误或响应结构损坏是基础设施失败，不能伪装成 0% 能力。
                raise EvaluationRequestError(
                    f"样本 {sample.sample_id} 推理请求失败（{type(exc).__name__}）"
                ) from exc

    tasks = [
        asyncio.create_task(run_one(sample, prompt))
        for sample, prompt in iter_prompts(samples, template)
    ]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return aggregate_results(
        results,
        dataset_sha256=dataset_sha256,
        model_name=model_name,
        template=template,
    )
