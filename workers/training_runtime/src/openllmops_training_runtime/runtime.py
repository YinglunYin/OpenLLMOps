"""无 shell 训练/合并执行器及 PID 1 信号转发。"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from uuid import uuid4

from openllmops_training_config import (
    Algorithm,
    DatasetFormat,
    Stage,
    TrainingHyperparameters,
    TrainingRequest,
    build_training_config,
)

from .artifacts import validate_adapter_directory, validate_full_model_directory

MAX_CONFIG_BYTES = 1024 * 1024
FORBIDDEN_ROOT_SUFFIXES = frozenset({".bin", ".ckpt", ".joblib", ".pkl", ".pickle", ".pt", ".pth"})


class TrainingRuntimeError(RuntimeError):
    """训练或合并子进程失败。"""


class TrainingInterrupted(TrainingRuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"训练收到终止信号 {signum}")
        self.signum = signum


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    config_path: Path
    model_path: Path
    dataset_dir: Path
    output_path: Path
    stage: Stage
    algorithm: Algorithm
    dataset_format: DatasetFormat


class ProcessSupervisor:
    """串行监管一个独立进程组，终止时先转发信号再强制清理。"""

    def __init__(self, *, grace_seconds: float = 20.0, poll_seconds: float = 0.1) -> None:
        self.grace_seconds = grace_seconds
        self.poll_seconds = poll_seconds
        self._requested_signal: int | None = None
        self._current_pgid: int | None = None
        self._previous_handlers: dict[int, signal.Handlers] = {}

    def install(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[signum] = signal.signal(signum, self._handle_signal)

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        if self._requested_signal is None:
            self._requested_signal = signum

    def run(self, command: Sequence[str]) -> None:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise TrainingRuntimeError("子进程参数数组为空或无效")
        if self._requested_signal is not None:
            raise TrainingInterrupted(self._requested_signal)
        try:
            process = subprocess.Popen(
                list(command),
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise TrainingRuntimeError(f"无法启动训练子进程：{command[0]}") from exc
        self._current_pgid = process.pid
        forwarded = False
        deadline: float | None = None
        try:
            while process.poll() is None:
                if self._requested_signal is not None and not forwarded:
                    self._signal_group(self._requested_signal)
                    forwarded = True
                    deadline = time.monotonic() + self.grace_seconds
                elif forwarded and deadline is not None and time.monotonic() >= deadline:
                    self._signal_group(signal.SIGKILL)
                    deadline = None
                time.sleep(self.poll_seconds)
            return_code = process.wait()
        finally:
            # 启动器异常退出时仍可能留下 torchrun worker；退出前清理整个进程组。
            self._cleanup_group()
            self._current_pgid = None
        if self._requested_signal is not None:
            raise TrainingInterrupted(self._requested_signal)
        if return_code != 0:
            raise TrainingRuntimeError(f"子进程 {command[0]} 失败，exit_code={return_code}")

    def _signal_group(self, signum: int) -> None:
        if self._current_pgid is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(self._current_pgid, signum)

    def _cleanup_group(self) -> None:
        if self._current_pgid is None:
            return
        try:
            os.killpg(self._current_pgid, 0)
        except ProcessLookupError:
            return
        self._signal_group(signal.SIGTERM)
        deadline = time.monotonic() + min(self.grace_seconds, 5.0)
        while time.monotonic() < deadline:
            try:
                os.killpg(self._current_pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(self.poll_seconds)
        self._signal_group(signal.SIGKILL)


def _load_json_config(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise TrainingRuntimeError(f"训练配置不允许非有限数值：{value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingRuntimeError(f"训练配置包含重复字段：{key}")
            result[key] = value
        return result

    try:
        info = path.lstat()
        if path.is_symlink() or not path.is_file() or not 1 <= info.st_size <= MAX_CONFIG_BYTES:
            raise TrainingRuntimeError("训练配置必须是有界的非软链接普通文件")
        with path.open("rb") as source:
            raw = source.read(MAX_CONFIG_BYTES + 1)
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except TrainingRuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingRuntimeError("训练配置不是有效的 UTF-8 JSON") from exc
    if len(raw) > MAX_CONFIG_BYTES or not isinstance(value, dict):
        raise TrainingRuntimeError("训练配置必须是有界 JSON 对象")
    return value


def validate_training_config(spec: TrainingSpec) -> dict[str, object]:
    actual = _load_json_config(spec.config_path)
    parameter_fields = TrainingHyperparameters.model_fields
    raw_parameters = {name: actual[name] for name in parameter_fields if name in actual}
    try:
        parameters = TrainingHyperparameters.model_validate(raw_parameters)
        expected = build_training_config(
            TrainingRequest(
                stage=spec.stage,
                algorithm=spec.algorithm,
                model_path=spec.model_path,
                dataset_dir=spec.dataset_dir,
                output_dir=spec.output_path,
                dataset_format=spec.dataset_format,
                **parameters.model_dump(),
            )
        )
    except (TypeError, ValueError) as exc:
        raise TrainingRuntimeError(f"训练配置未通过严格白名单：{exc}") from exc
    if actual != expected:
        unexpected = sorted(set(actual) - set(expected))
        detail = f"，额外字段={unexpected}" if unexpected else ""
        raise TrainingRuntimeError(f"训练配置与节点派生合同不一致{detail}")
    return actual


def _write_export_config(
    path: Path, spec: TrainingSpec, template: object, export_dir: Path
) -> None:
    config: dict[str, object] = {
        "model_name_or_path": str(spec.model_path),
        "adapter_name_or_path": str(spec.output_path),
        "finetuning_type": "lora",
        "trust_remote_code": False,
        "export_dir": str(export_dir),
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    if isinstance(template, str):
        config["template"] = template
    body = json.dumps(
        config, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("x", encoding="utf-8") as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    path.chmod(0o400)


def _remove_unsafe_training_files(output_path: Path) -> None:
    """删除 Trainer 生成的 pickle 状态，只保留可安全分发的模型快照。"""

    root = output_path.resolve(strict=True)
    if output_path.is_symlink() or not root.is_dir():
        raise TrainingRuntimeError("训练输出必须是非软链接目录")
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise TrainingRuntimeError("训练输出包含链接或特殊目录")
        for name in names:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise TrainingRuntimeError("训练输出包含链接或特殊文件")
            if candidate.suffix.casefold() in FORBIDDEN_ROOT_SUFFIXES:
                candidate.unlink()


def _safe_remove_merge_staging(candidate: Path, output_path: Path) -> None:
    if candidate.parent != output_path or not candidate.name.startswith(".openllmops-merge-"):
        raise TrainingRuntimeError("拒绝清理非系统派生的合并暂存目录")
    if not candidate.exists():
        return
    if candidate.is_symlink() or not candidate.is_dir():
        raise TrainingRuntimeError("合并暂存路径类型无效")
    shutil.rmtree(candidate)


def run_training(spec: TrainingSpec, execute: Callable[[Sequence[str]], None]) -> Path:
    """运行训练并返回最终可部署路径；任何阶段失败都不会返回成功。"""

    config = validate_training_config(spec)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "DISABLE_VERSION_CHECK": "1",
        }
    )
    execute(("llamafactory-cli", "train", str(spec.config_path)))
    _remove_unsafe_training_files(spec.output_path)

    if spec.algorithm == Algorithm.FREEZE:
        return validate_full_model_directory(spec.output_path)

    validate_adapter_directory(spec.output_path)
    merged_path = spec.output_path / "merged"
    if merged_path.exists() or merged_path.is_symlink():
        raise TrainingRuntimeError("合并输出目录已经存在，禁止覆盖")
    staging = spec.output_path / f".openllmops-merge-{uuid4()}"
    export_config = Path("/tmp") / f"openllmops-export-{uuid4()}.json"
    try:
        _write_export_config(export_config, spec, config.get("template"), staging)
        execute(("llamafactory-cli", "export", str(export_config)))
        validate_full_model_directory(staging)
        os.replace(staging, merged_path)
    finally:
        export_config.unlink(missing_ok=True)
        _safe_remove_merge_staging(staging, spec.output_path)
    return validate_full_model_directory(merged_path)
