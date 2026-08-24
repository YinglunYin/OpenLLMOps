"""在一个隔离 GPU 容器内顺序评测基线与候选模型。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path

import httpx

from .client import CompatibleClient
from .metrics import compare_reports
from .models import ComparisonReport, EvaluationReport, Sample
from .runner import evaluate

SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class EvaluationRuntimeError(RuntimeError):
    """vLLM 子进程未健康启动或异常退出。"""


@dataclass(frozen=True, slots=True)
class ModelTarget:
    path: Path
    served_name: str
    template: str


@dataclass(frozen=True, slots=True)
class PairReport:
    baseline: EvaluationReport
    candidate: EvaluationReport
    comparison: ComparisonReport

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "comparison": self.comparison.as_dict(),
        }


def build_vllm_command(
    target: ModelTarget,
    *,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """所有参数由执行器生成，不拼接任意 shell 字符串。"""

    model_path = target.path.resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("评测模型路径必须是目录")
    if not SAFE_MODEL_NAME.fullmatch(target.served_name):
        raise ValueError("served_name 格式不安全")
    if target.template not in {"base", "instruct"}:
        raise ValueError("template 仅支持 base 或 instruct")
    if not 1024 <= port <= 65535:
        raise ValueError("vLLM 端口超出范围")
    if not 1 <= tensor_parallel_size <= 16:
        raise ValueError("tensor_parallel_size 超出范围")
    if not 0.1 <= gpu_memory_utilization <= 0.98:
        raise ValueError("gpu_memory_utilization 必须位于 0.1..0.98")

    return [
        "vllm",
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        target.served_name,
        "--runner",
        "generate",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--load-format",
        "safetensors",
        "--disable-log-requests",
    ]


async def _wait_until_ready(process: asyncio.subprocess.Process, port: int, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2) as client:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise EvaluationRuntimeError(f"vLLM 在健康检查前退出，状态码 {process.returncode}")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise EvaluationRuntimeError(f"vLLM 在 {timeout:.0f} 秒内未就绪")


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.wait_for(process.wait(), timeout=30)
    except ProcessLookupError:
        return
    except TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def _evaluate_target(
    target: ModelTarget,
    *,
    samples: list[Sample],
    dataset_sha256: str,
    log_path: Path,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    concurrency: int,
    max_tokens: int,
    startup_timeout_seconds: float,
) -> EvaluationReport:
    command = build_vllm_command(
        target,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await _wait_until_ready(process, port, startup_timeout_seconds)
            client = CompatibleClient(f"http://127.0.0.1:{port}", None)
            try:
                return await evaluate(
                    client=client,
                    samples=samples,
                    dataset_sha256=dataset_sha256,
                    model_name=target.served_name,
                    template=target.template,
                    concurrency=concurrency,
                    max_tokens=max_tokens,
                )
            finally:
                await client.close()
        finally:
            # 基线退出并释放显存后才启动候选，确保前后运行条件一致且不会双倍占卡。
            await _stop_process_group(process)


async def evaluate_pair(
    *,
    baseline: ModelTarget,
    candidate: ModelTarget,
    samples: list[Sample],
    dataset_sha256: str,
    output_dir: Path,
    port: int = 18000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    concurrency: int = 4,
    max_tokens: int = 32,
    startup_timeout_seconds: float = 600,
) -> PairReport:
    """使用同一组 GPU 和生成参数顺序运行，量化训练前后差异。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_report = await _evaluate_target(
        baseline,
        samples=samples,
        dataset_sha256=dataset_sha256,
        log_path=output_dir / "baseline-vllm.log",
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        concurrency=concurrency,
        max_tokens=max_tokens,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    (output_dir / "baseline-report.json").write_text(
        json.dumps(baseline_report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    candidate_report = await _evaluate_target(
        candidate,
        samples=samples,
        dataset_sha256=dataset_sha256,
        log_path=output_dir / "candidate-vllm.log",
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        concurrency=concurrency,
        max_tokens=max_tokens,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    comparison = compare_reports(baseline_report, candidate_report)
    pair = PairReport(baseline_report, candidate_report, comparison)
    (output_dir / "pair-report.json").write_text(
        json.dumps(pair.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pair
