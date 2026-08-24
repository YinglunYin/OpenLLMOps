from __future__ import annotations

from pathlib import Path

import pytest

from openllmops_eval.orchestrator import ModelTarget, build_vllm_command


def test_vllm_command_has_no_remote_code_and_no_shell(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()

    command = build_vllm_command(
        ModelTarget(model, "baseline", "base"),
        port=18000,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )

    assert command[:2] == ["vllm", "serve"]
    assert "--load-format" in command
    assert "safetensors" in command
    assert "--trust-remote-code" not in command
    assert command[command.index("--tensor-parallel-size") + 1] == "2"


def test_rejects_unsafe_served_name(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(ValueError, match="格式不安全"):
        build_vllm_command(
            ModelTarget(model, "bad name; rm", "base"),
            port=18000,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
