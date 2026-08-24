from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from openllmops_training_config import (
    Algorithm,
    DatasetFormat,
    Stage,
    TrainingRequest,
    build_training_config,
)

from openllmops_training_runtime import (
    ProcessSupervisor,
    TrainingInterrupted,
    TrainingRuntimeError,
    TrainingSpec,
    run_training,
)


def spec(tmp_path: Path, algorithm: Algorithm) -> TrainingSpec:
    model = tmp_path / "model"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    config = tmp_path / "training.json"
    for directory in (model, dataset, output):
        directory.mkdir()
    request = TrainingRequest(
        stage=Stage.SFT,
        algorithm=algorithm,
        model_path=model,
        dataset_dir=dataset,
        output_dir=output,
        dataset_format=DatasetFormat.ALPACA,
        template="qwen",
        num_train_epochs=1.0,
    )
    config.write_text(
        json.dumps(build_training_config(request), sort_keys=True),
        encoding="utf-8",
    )
    return TrainingSpec(
        config_path=config,
        model_path=model,
        dataset_dir=dataset,
        output_path=output,
        stage=Stage.SFT,
        algorithm=algorithm,
        dataset_format=DatasetFormat.ALPACA,
    )


def write_adapter(root: Path) -> None:
    (root / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"adapter")


def write_full_model(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"model")


def test_lora_runs_train_then_export_without_shell_and_atomically_publishes(
    tmp_path: Path,
) -> None:
    training = spec(tmp_path, Algorithm.LORA)
    commands: list[list[str]] = []

    def execute(command: Sequence[str]) -> None:
        commands.append(list(command))
        if command[1] == "train":
            write_adapter(training.output_path)
            (training.output_path / "training_args.bin").write_bytes(b"unsafe pickle")
            checkpoint = training.output_path / "checkpoint-5"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
            (checkpoint / "optimizer.pt").write_bytes(b"unsafe optimizer pickle")
            return
        export = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        assert export["trust_remote_code"] is False
        write_full_model(Path(export["export_dir"]))

    result = run_training(training, execute)

    assert result == training.output_path / "merged"
    assert commands[0][:2] == ["llamafactory-cli", "train"]
    assert commands[1][:2] == ["llamafactory-cli", "export"]
    assert not (training.output_path / "training_args.bin").exists()
    assert not (training.output_path / "checkpoint-5" / "optimizer.pt").exists()
    assert not list(training.output_path.glob(".openllmops-merge-*"))


def test_freeze_output_itself_is_validated_as_deployable(tmp_path: Path) -> None:
    training = spec(tmp_path, Algorithm.FREEZE)

    def execute(command: Sequence[str]) -> None:
        assert list(command[:2]) == ["llamafactory-cli", "train"]
        write_full_model(training.output_path)

    assert run_training(training, execute) == training.output_path


def test_wrapper_rejects_tampered_generated_config_before_start(tmp_path: Path) -> None:
    training = spec(tmp_path, Algorithm.LORA)
    config = json.loads(training.config_path.read_text(encoding="utf-8"))
    config["deepspeed"] = "/tmp/arbitrary.json"
    training.config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TrainingRuntimeError, match="额外字段"):
        run_training(training, lambda _command: pytest.fail("不应启动子进程"))


def test_wrapper_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    training = spec(tmp_path, Algorithm.LORA)
    body = training.config_path.read_text(encoding="utf-8")
    training.config_path.write_text(
        body[:-1] + ',"trust_remote_code":true}',
        encoding="utf-8",
    )

    with pytest.raises(TrainingRuntimeError, match="重复字段"):
        run_training(training, lambda _command: pytest.fail("不应启动子进程"))


def test_failed_export_cleans_only_system_merge_staging(tmp_path: Path) -> None:
    training = spec(tmp_path, Algorithm.QLORA)

    def execute(command: Sequence[str]) -> None:
        if command[1] == "train":
            write_adapter(training.output_path)
            return
        export = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        staging = Path(export["export_dir"])
        staging.mkdir()
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise TrainingRuntimeError("export failed")

    with pytest.raises(TrainingRuntimeError, match="export failed"):
        run_training(training, execute)
    assert not list(training.output_path.glob(".openllmops-merge-*"))
    assert (training.output_path / "adapter_model.safetensors").is_file()


def test_process_supervisor_forwards_sigterm_and_reaps_process_group() -> None:
    supervisor = ProcessSupervisor(grace_seconds=0.2, poll_seconds=0.01)
    errors: list[BaseException] = []

    def target() -> None:
        try:
            supervisor.run([sys.executable, "-c", "import time; time.sleep(30)"])
        except BaseException as exc:  # 测试线程需要把异常转回主线程断言。
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    deadline = time.monotonic() + 2
    while supervisor._current_pgid is None and time.monotonic() < deadline:
        time.sleep(0.01)
    pgid = supervisor._current_pgid
    assert pgid is not None
    supervisor._handle_signal(signal.SIGTERM, None)
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TrainingInterrupted)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
