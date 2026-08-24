"""训练运行时命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from openllmops_training_config import Algorithm, DatasetFormat, Stage

from .artifacts import TrainingArtifactError
from .contract import WORKSPACE_CONFIG, WORKSPACE_DATASET, WORKSPACE_MODEL, WORKSPACE_OUTPUT
from .runtime import (
    ProcessSupervisor,
    TrainingInterrupted,
    TrainingRuntimeError,
    TrainingSpec,
    run_training,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenLLMOps 安全训练与合并入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--stage", choices=[item.value for item in Stage], required=True)
    run.add_argument("--algorithm", choices=[item.value for item in Algorithm], required=True)
    run.add_argument(
        "--dataset-format",
        choices=[item.value for item in DatasetFormat],
        required=True,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected_paths = {
        "config": WORKSPACE_CONFIG,
        "model_path": WORKSPACE_MODEL,
        "dataset_dir": WORKSPACE_DATASET,
        "output_dir": WORKSPACE_OUTPUT,
    }
    for name, expected in expected_paths.items():
        if getattr(args, name) != expected:
            _parser().error(f"{name} 必须固定为 {expected}")
    spec = TrainingSpec(
        config_path=args.config,
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_path=args.output_dir,
        stage=Stage(args.stage),
        algorithm=Algorithm(args.algorithm),
        dataset_format=DatasetFormat(args.dataset_format),
    )
    supervisor = ProcessSupervisor()
    supervisor.install()
    try:
        run_training(spec, supervisor.run)
    except TrainingInterrupted as exc:
        return 128 + exc.signum
    except (TrainingRuntimeError, TrainingArtifactError) as exc:
        print(f"训练运行时失败：{exc}", flush=True)
        return 1
    finally:
        supervisor.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
