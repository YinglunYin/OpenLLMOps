"""评测容器命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .benchmarks import convert_csv_directory
from .builtin_benchmarks import BenchmarkPreparationError, prepare_builtin_benchmark
from .client import CompatibleClient
from .dataset import load_jsonl
from .orchestrator import ModelTarget, evaluate_pair
from .runner import evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenLLMOps 模型评测执行器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="通过 OpenAI Compatible 接口运行评测")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--template", choices=("base", "instruct"), required=True)
    run.add_argument("--api-key-env", default="OPENLLMOPS_API_KEY")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--max-tokens", type=int, default=32)

    convert = subparsers.add_parser("convert-benchmark", help="转换 C-Eval/CMMLU CSV")
    convert.add_argument("--source-dir", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--benchmark", choices=("ceval", "cmmlu"), required=True)

    prepare = subparsers.add_parser(
        "prepare-benchmark",
        help="从固定官方在线制品或管理员提供的离线来源准备内置评测集",
    )
    prepare.add_argument("--benchmark", choices=("ceval", "cmmlu"), required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    source_group = prepare.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--online", action="store_true", help="下载内置的固定官方 revision")
    source_group.add_argument("--source", type=Path, help="官方 CSV 目录或 ZIP/TAR 归档")
    prepare.add_argument(
        "--source-revision",
        help="离线来源的官方 commit、tag 或发布版本（由管理员声明）",
    )
    prepare.add_argument(
        "--accept-non-commercial-license",
        action="store_true",
        help="确认接受 CC BY-NC-SA 4.0；在线下载必须提供",
    )
    prepare.add_argument(
        "--split",
        action="append",
        choices=("dev", "val", "test"),
        help="可重复；不指定时使用该评测集有答案的默认 split",
    )
    prepare.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许科目不完整的测试子集；正式内置数据不应使用",
    )
    prepare.add_argument("--overwrite", action="store_true", help="显式替换既有输出")

    pair = subparsers.add_parser("run-pair", help="顺序启动 vLLM 并比较训练前后模型")
    pair.add_argument("--dataset", type=Path, required=True)
    pair.add_argument("--output-dir", type=Path, required=True)
    pair.add_argument("--baseline-path", type=Path, required=True)
    pair.add_argument("--baseline-name", default="baseline")
    pair.add_argument("--baseline-template", choices=("base", "instruct"), required=True)
    pair.add_argument("--candidate-path", type=Path, required=True)
    pair.add_argument("--candidate-name", default="candidate")
    pair.add_argument("--candidate-template", choices=("base", "instruct"), required=True)
    pair.add_argument("--tensor-parallel-size", type=int, default=1)
    pair.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    pair.add_argument("--concurrency", type=int, default=4)
    pair.add_argument("--max-tokens", type=int, default=32)
    return parser


async def _run(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"环境变量 {args.api_key_env} 未设置")
    samples, fingerprint = load_jsonl(args.dataset)
    client = CompatibleClient(args.base_url, api_key)
    try:
        report = await evaluate(
            client=client,
            samples=samples,
            dataset_sha256=fingerprint,
            model_name=args.model,
            template=args.template,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    finally:
        await client.close()
    args.output.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def _run_pair(args: argparse.Namespace) -> None:
    samples, fingerprint = load_jsonl(args.dataset)
    await evaluate_pair(
        baseline=ModelTarget(args.baseline_path, args.baseline_name, args.baseline_template),
        candidate=ModelTarget(args.candidate_path, args.candidate_name, args.candidate_template),
        samples=samples,
        dataset_sha256=fingerprint,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-benchmark":
        try:
            result = prepare_builtin_benchmark(
                args.benchmark,
                args.output_dir,
                online=args.online,
                source=args.source,
                source_revision=args.source_revision,
                accept_noncommercial_license=args.accept_non_commercial_license,
                splits=tuple(args.split) if args.split else None,
                allow_partial=args.allow_partial,
                overwrite=args.overwrite,
            )
        except BenchmarkPreparationError as exc:
            raise SystemExit(f"准备评测集失败：{exc}") from exc
        print(
            json.dumps(
                {
                    "jsonl": str(result.jsonl_path),
                    "manifest": str(result.manifest_path),
                    "record_count": result.record_count,
                    "sha256": result.jsonl_sha256,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.command == "convert-benchmark":
        count = convert_csv_directory(args.source_dir, args.output, args.benchmark)
        print(f"已转换 {count} 条样本")
        return
    if args.command == "run-pair":
        asyncio.run(_run_pair(args))
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
