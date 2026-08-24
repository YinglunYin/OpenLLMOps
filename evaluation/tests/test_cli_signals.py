from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sigterm_waits_for_async_cleanup(tmp_path: Path) -> None:
    marker = tmp_path / "cleanup-complete"
    program = f"""
import asyncio
import os
import signal
from pathlib import Path
from openllmops_eval.cli import _run_async

async def workload():
    asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
    try:
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0.05)
        Path({str(marker)!r}).write_text("done", encoding="utf-8")

try:
    _run_async(workload())
except SystemExit as exc:
    if exc.code != 128 + signal.SIGTERM:
        raise
else:
    raise RuntimeError("SIGTERM 未产生预期退出状态")
"""

    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "done"
