"""评测容器命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .benchmarks import convert_csv_directory
from .client import CompatibleClient
from .dataset import load_jsonl
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
    args.output.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "convert-benchmark":
        count = convert_csv_directory(args.source_dir, args.output, args.benchmark)
        print(f"已转换 {count} 条样本")
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

