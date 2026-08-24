"""在线模型仓库下载适配器；下载 SDK 在可终止的子进程中运行。"""

from __future__ import annotations

import fnmatch
import multiprocessing
import os
import re
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

ProgressCallback = Callable[[str, int, int | None], None]
CancelCallback = Callable[[], bool]
DownloadSource = Literal["huggingface", "modelscope"]

ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
)
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
DOWNLOAD_POLL_SECONDS = 0.25
MAX_GIT_ADVERTISEMENT_BYTES = 16 * 1024 * 1024
MAX_PROGRESS_BYTES = (1 << 63) - 1
MODELSCOPE_GIT_ENDPOINT = "https://www.modelscope.cn"


class DownloaderUnavailableError(RuntimeError):
    """运行镜像未安装对应在线来源依赖。"""


class DownloadCancelledError(RuntimeError):
    """在线下载被管理员取消。"""


class DownloadProcessError(RuntimeError):
    """隔离下载子进程失败。"""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """在线快照的不可变版本和远端声明大小。"""

    resolved_revision: str
    total_bytes: int | None


def _matches_allowed_file(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in ALLOW_PATTERNS)


def _normalize_commit(value: object, source: str) -> str:
    if not isinstance(value, str) or not COMMIT_PATTERN.fullmatch(value):
        raise ValueError(f"{source} 未返回完整、不可变的 commit SHA")
    return value.casefold()


def _declared_total(files: list[tuple[str, object]]) -> int | None:
    matched = [(path, size) for path, size in files if _matches_allowed_file(path)]
    if not matched or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0 for _, size in matched
    ):
        return None
    total = sum(size for _, size in matched if isinstance(size, int))
    if total > MAX_PROGRESS_BYTES:
        raise ValueError("远端声明大小超过系统可记录范围")
    return total


def _download_huggingface_snapshot(
    repository: str,
    requested_revision: str | None,
    destination: Path,
    token: str | None,
    metadata_callback: Callable[[DownloadResult], None] | None = None,
) -> DownloadResult:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise DownloaderUnavailableError("执行器未安装 huggingface 可选依赖") from exc

    # 先把分支/标签解析成 commit，再按该 commit 下载，避免查询与下载之间 main 漂移。
    info = HfApi().model_info(
        repository,
        revision=requested_revision,
        files_metadata=True,
        token=token,
    )
    resolved_revision = _normalize_commit(getattr(info, "sha", None), "Hugging Face")
    siblings = list(getattr(info, "siblings", None) or ())
    total = _declared_total(
        [
            (str(getattr(sibling, "rfilename", "")), getattr(sibling, "size", None))
            for sibling in siblings
        ]
    )
    result = DownloadResult(resolved_revision=resolved_revision, total_bytes=total)
    if metadata_callback:
        metadata_callback(result)
    snapshot_download(
        repo_id=repository,
        revision=resolved_revision,
        token=token,
        local_dir=destination,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    return result


def _parse_git_refs(payload: bytes) -> dict[str, str]:
    """解析 Git smart-HTTP v0 pkt-line ref advertisement。"""

    refs: dict[str, str] = {}
    cursor = 0
    while cursor < len(payload):
        if cursor + 4 > len(payload):
            raise ValueError("ModelScope Git ref 响应被截断")
        try:
            packet_size = int(payload[cursor : cursor + 4], 16)
        except ValueError as exc:
            raise ValueError("ModelScope Git ref 响应格式无效") from exc
        cursor += 4
        if packet_size == 0:
            continue
        if packet_size < 4 or cursor + packet_size - 4 > len(payload):
            raise ValueError("ModelScope Git ref 响应长度无效")
        packet = payload[cursor : cursor + packet_size - 4]
        cursor += packet_size - 4
        line = packet.split(b"\0", 1)[0].rstrip(b"\n")
        try:
            commit_bytes, ref_bytes = line.split(b" ", 1)
            commit = _normalize_commit(commit_bytes.decode("ascii"), "ModelScope")
            ref = ref_bytes.decode("utf-8")
        except (UnicodeError, ValueError):
            # service 声明等 pkt-line 不是 ref，直接忽略；疑似 ref 的坏行最终会表现为找不到版本。
            continue
        if ref and not any(ord(character) < 0x20 for character in ref):
            refs[ref] = commit
    return refs


def _modelscope_git_url(repository: str) -> str:
    parts = repository.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("ModelScope repository 必须使用 namespace/model 格式")
    encoded = "/".join(quote(part, safe="") for part in parts)
    return f"{MODELSCOPE_GIT_ENDPOINT}/{encoded}.git/info/refs?service=git-upload-pack"


def _resolve_modelscope_revision(
    repository: str,
    requested_revision: str | None,
    token: str | None,
) -> str:
    if requested_revision and COMMIT_PATTERN.fullmatch(requested_revision):
        return _normalize_commit(requested_revision, "ModelScope")

    headers = {"Accept": "*/*", "User-Agent": "git/2.39.5"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(_modelscope_git_url(repository), headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = response.read(MAX_GIT_ADVERTISEMENT_BYTES + 1)
    if len(payload) > MAX_GIT_ADVERTISEMENT_BYTES:
        raise ValueError("ModelScope Git ref 响应超过 16 MiB 限制")
    refs = _parse_git_refs(payload)
    revision = requested_revision or "master"
    candidates = (
        f"refs/tags/{revision}^{{}}",
        f"refs/tags/{revision}",
        f"refs/heads/{revision}",
        revision if revision.startswith("refs/") else "",
    )
    for candidate in candidates:
        if candidate and candidate in refs:
            return refs[candidate]
    raise ValueError(f"ModelScope revision 不存在或无法解析为 commit: {revision}")


def _download_modelscope_snapshot(
    repository: str,
    requested_revision: str | None,
    destination: Path,
    token: str | None,
    metadata_callback: Callable[[DownloadResult], None] | None = None,
) -> DownloadResult:
    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise DownloaderUnavailableError("执行器未安装 modelscope 可选依赖") from exc

    resolved_revision = _resolve_modelscope_revision(repository, requested_revision, token)
    # 空字符串是显式匿名，阻止 SDK 在 token=None 时回退读取运行用户的持久化凭证。
    effective_token = token if token is not None else ""
    files = HubApi(token=effective_token).get_model_files(repository, revision=resolved_revision)
    declared_files = [
        (str(item.get("Path", "")), item.get("Size")) for item in files if isinstance(item, dict)
    ]
    total = _declared_total(declared_files)
    result = DownloadResult(resolved_revision=resolved_revision, total_bytes=total)
    if metadata_callback:
        metadata_callback(result)
    snapshot_download(
        model_id=repository,
        revision=resolved_revision,
        local_dir=str(destination),
        allow_patterns=list(ALLOW_PATTERNS),
        token=effective_token,
    )
    return result


def _send_message(connection: Connection, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        # 父进程取消后会关闭管道；子进程随后被终止，无需覆盖原始结果。
        pass


def _download_worker(
    source: DownloadSource,
    repository: str,
    requested_revision: str | None,
    destination: str,
    token: str | None,
    connection: Connection,
) -> None:
    try:

        def report_metadata(result: DownloadResult) -> None:
            _send_message(
                connection,
                {
                    "type": "metadata",
                    "resolved_revision": result.resolved_revision,
                    "total_bytes": result.total_bytes,
                },
            )

        if source == "huggingface":
            result = _download_huggingface_snapshot(
                repository,
                requested_revision,
                Path(destination),
                token,
                report_metadata,
            )
        else:
            result = _download_modelscope_snapshot(
                repository,
                requested_revision,
                Path(destination),
                token,
                report_metadata,
            )
        _send_message(
            connection,
            {
                "type": "done",
                "resolved_revision": result.resolved_revision,
                "total_bytes": result.total_bytes,
            },
        )
    except Exception as exc:  # noqa: BLE001 - 子进程需把第三方 SDK 异常传回父进程
        _send_message(
            connection,
            {
                "type": "error",
                "error_type": type(exc).__name__,
                "message": str(exc) or type(exc).__name__,
            },
        )
    finally:
        connection.close()


def _directory_size(root: Path) -> int:
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            try:
                file_stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(file_stat.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(file_stat.st_mode):
                total += file_stat.st_size
    return total


def _stop_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _run_download_process(
    source: DownloadSource,
    repository: str,
    requested_revision: str | None,
    destination: Path,
    token: str | None,
    *,
    progress: ProgressCallback | None,
    cancelled: CancelCallback | None,
    worker: Callable[[DownloadSource, str, str | None, str, str | None, Connection], None]
    | None = None,
    poll_seconds: float = DOWNLOAD_POLL_SECONDS,
) -> DownloadResult:
    """监控隔离下载进程；取消时先杀进程，调用方随后可安全删除暂存目录。"""

    if cancelled and cancelled():
        raise DownloadCancelledError("模型在线下载已取消")
    if progress:
        progress("transferring", 0, None)

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=worker or _download_worker,
        args=(
            source,
            repository,
            requested_revision,
            str(destination),
            token,
            child_connection,
        ),
        daemon=True,
        name=f"model-download-{source}",
    )
    result: DownloadResult | None = None
    error: tuple[str, str] | None = None
    declared_total: int | None = None
    last_reported = -1
    started = False
    space_checked = False

    def receive_messages() -> None:
        nonlocal declared_total, error, result
        while parent_connection.poll():
            try:
                message = parent_connection.recv()
            except EOFError:
                return
            if not isinstance(message, dict):
                error = ("DownloadProcessError", "下载子进程返回了无效消息")
                continue
            if message.get("type") in {"metadata", "done"}:
                try:
                    resolved = _normalize_commit(message.get("resolved_revision"), source)
                    raw_total = message.get("total_bytes")
                    if raw_total is not None and (
                        isinstance(raw_total, bool)
                        or not isinstance(raw_total, int)
                        or raw_total < 0
                        or raw_total > MAX_PROGRESS_BYTES
                    ):
                        raise ValueError("远端声明大小无效")
                    declared_total = raw_total
                    if message.get("type") == "done":
                        result = DownloadResult(resolved_revision=resolved, total_bytes=raw_total)
                except ValueError as exc:
                    error = (type(exc).__name__, str(exc))
            elif message.get("type") == "error":
                error = (
                    str(message.get("error_type") or "DownloadProcessError"),
                    str(message.get("message") or "下载子进程失败"),
                )

    try:
        process.start()
        started = True
        child_connection.close()
        while process.is_alive():
            receive_messages()
            if cancelled and cancelled():
                raise DownloadCancelledError("模型在线下载已取消")
            completed = _directory_size(destination)
            if declared_total is not None and not space_checked:
                remaining = max(0, declared_total - completed)
                reserve = max(1024**3, declared_total // 10)
                free = shutil.disk_usage(destination).free
                if free < remaining + reserve:
                    raise OSError(
                        f"模型在线导入空间不足：还需要 {remaining + reserve} 字节（含安全余量）"
                    )
                space_checked = True
            if completed != last_reported:
                last_reported = completed
                if progress:
                    progress(
                        "transferring",
                        min(completed, declared_total) if declared_total is not None else completed,
                        declared_total,
                    )
            process.join(timeout=poll_seconds)
        process.join()
        receive_messages()
        if error is not None:
            error_type, message = error
            if error_type in {"ImportError", "ModuleNotFoundError", "DownloaderUnavailableError"}:
                raise DownloaderUnavailableError(message)
            raise DownloadProcessError(message)
        if process.exitcode != 0:
            raise DownloadProcessError(f"下载子进程异常退出: {process.exitcode}")
        if result is None:
            raise DownloadProcessError("下载子进程未返回 resolved commit")
        if progress:
            completed = result.total_bytes
            if completed is None:
                completed = _directory_size(destination)
            progress("transferring", completed, result.total_bytes)
        return result
    except BaseException:
        if started:
            _stop_process(process)
        raise
    finally:
        child_connection.close()
        parent_connection.close()


def download_huggingface(
    repository: str,
    revision: str | None,
    destination: Path,
    token: str | None,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> DownloadResult:
    return _run_download_process(
        "huggingface",
        repository,
        revision,
        destination,
        token,
        progress=progress,
        cancelled=cancelled,
    )


def download_modelscope(
    repository: str,
    revision: str | None,
    destination: Path,
    token: str | None,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> DownloadResult:
    return _run_download_process(
        "modelscope",
        repository,
        revision,
        destination,
        token,
        progress=progress,
        cancelled=cancelled,
    )
