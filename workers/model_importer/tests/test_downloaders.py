from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self

import pytest

from openllmops_model_importer import downloaders
from openllmops_model_importer.downloaders import (
    DownloadCancelledError,
    DownloadResult,
)


def _pkt_line(payload: bytes) -> bytes:
    return f"{len(payload) + 4:04x}".encode("ascii") + payload


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self.payload[:size]


def _slow_download_worker(
    source: str,
    repository: str,
    requested_revision: str | None,
    destination: str,
    token: str | None,
    connection: Any,
) -> None:
    del source, repository, requested_revision, token
    resolved = "b" * 40
    connection.send(
        {"type": "metadata", "resolved_revision": resolved, "total_bytes": 64 * 1024 * 100}
    )
    with (Path(destination) / "partial.safetensors").open("wb") as output:
        for _ in range(100):
            output.write(b"\0" * (64 * 1024))
            output.flush()
            time.sleep(0.02)
    connection.send({"type": "done", "resolved_revision": resolved, "total_bytes": 64 * 1024 * 100})
    connection.close()


def test_huggingface_resolves_branch_before_snapshot_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = "a" * 40
    calls: list[dict[str, object]] = []
    metadata: list[DownloadResult] = []
    fake_module = ModuleType("huggingface_hub")

    class FakeApi:
        def model_info(self, repository: str, **kwargs: object) -> SimpleNamespace:
            assert repository == "Qwen/Test"
            assert kwargs["revision"] == "main"
            return SimpleNamespace(
                sha=resolved,
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=10),
                    SimpleNamespace(rfilename="model.safetensors", size=90),
                    SimpleNamespace(rfilename="README.md", size=1_000),
                ],
            )

    def fake_snapshot_download(**kwargs: object) -> str:
        assert metadata and metadata[0].resolved_revision == resolved
        calls.append(kwargs)
        return str(tmp_path)

    fake_module.HfApi = FakeApi  # type: ignore[attr-defined]
    fake_module.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    result = downloaders._download_huggingface_snapshot(
        "Qwen/Test",
        "main",
        tmp_path,
        "secret",
        metadata.append,
    )

    assert result == DownloadResult(resolved_revision=resolved, total_bytes=100)
    assert calls[0]["revision"] == resolved
    assert calls[0]["token"] == "secret"


def test_modelscope_named_tag_resolves_to_peeled_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tag_object = "1" * 40
    commit = "2" * 40
    payload = b"".join(
        (
            _pkt_line(b"# service=git-upload-pack\n"),
            b"0000",
            _pkt_line(f"{tag_object} refs/tags/v1\0capabilities\n".encode()),
            _pkt_line(f"{commit} refs/tags/v1^{{}}\n".encode()),
            b"0000",
        )
    )
    observed_urls: list[str] = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeResponse:
        assert timeout == 30
        observed_urls.append(request.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(downloaders, "urlopen", fake_urlopen)

    assert downloaders._resolve_modelscope_revision("Qwen/Test", "v1", None) == commit
    assert observed_urls == [
        "https://www.modelscope.cn/Qwen/Test.git/info/refs?service=git-upload-pack"
    ]


def test_download_process_reports_bytes_and_terminates_on_cancel(tmp_path: Path) -> None:
    cancel_requested = False
    progress_values: list[tuple[int, int | None]] = []

    def progress(stage: str, completed: int, total: int | None) -> None:
        nonlocal cancel_requested
        assert stage == "transferring"
        progress_values.append((completed, total))
        if completed > 0:
            cancel_requested = True

    started_at = time.monotonic()
    with pytest.raises(DownloadCancelledError, match="取消"):
        downloaders._run_download_process(
            "huggingface",
            "Qwen/Test",
            "main",
            tmp_path,
            None,
            progress=progress,
            cancelled=lambda: cancel_requested,
            worker=_slow_download_worker,
            poll_seconds=0.01,
        )

    assert time.monotonic() - started_at < 5
    assert any(completed > 0 for completed, _ in progress_values)
    assert any(total == 64 * 1024 * 100 for _, total in progress_values)
    assert (tmp_path / "partial.safetensors").stat().st_size < 64 * 1024 * 100
    # 子进程已 join；暂存目录清理由上层 ModelImporter 负责。
    assert not any(
        child.name == "model-download-huggingface" for child in multiprocessing.active_children()
    )
