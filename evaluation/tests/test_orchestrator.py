from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from openllmops_eval.orchestrator import (
    ModelTarget,
    _process_group_exists,
    _stop_process_group,
    build_vllm_command,
)


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
    # vLLM 0.27.1 默认不记录请求；旧版 --disable-log-requests 已被移除。
    assert "--disable-log-requests" not in command
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


@pytest.mark.asyncio
async def test_stop_process_group_kills_child_after_leader_already_exited() -> None:
    child = "import signal,time; signal.signal(signal.SIGHUP, signal.SIG_IGN); time.sleep(60)"
    leader = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        leader,
        start_new_session=True,
    )
    process_group_id = process.pid
    try:
        await process.wait()
        # 组长已结束，但其 vLLM worker 类比进程仍占据同一个进程组。
        assert _process_group_exists(process_group_id)
        await _stop_process_group(
            process,
            term_timeout_seconds=2,
            kill_timeout_seconds=2,
        )
        assert not _process_group_exists(process_group_id)
    finally:
        if _process_group_exists(process_group_id):
            os.killpg(process_group_id, signal.SIGKILL)
