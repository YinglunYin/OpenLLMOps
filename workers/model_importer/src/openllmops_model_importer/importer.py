"""下载/复制、校验和同文件系统原子入库流程。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .downloaders import (
    DownloadCancelledError,
    DownloadResult,
    download_huggingface,
    download_modelscope,
)
from .paths import UnsafePathError, iter_regular_files, resolve_inside
from .validation import (
    GENERATED_MANIFEST_NAME,
    ModelManifest,
    ModelValidationCancelled,
    validate_model_directory,
)

ProgressCallback = Callable[[str, int, int | None], None]
CancelCallback = Callable[[], bool]


class ModelSource(StrEnum):
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    CONTROLLED_DIRECTORY = "controlled_directory"


class ImportCancelledError(RuntimeError):
    """任务被管理员取消。"""


@dataclass(frozen=True, slots=True)
class ImportRequest:
    import_id: uuid.UUID
    source: ModelSource
    repository: str | None = None
    revision: str | None = None
    source_directory: Path | None = None
    access_token: str | None = None


class ModelImporter:
    def __init__(self, *, inbox_root: Path, staging_root: Path, store_root: Path) -> None:
        self.inbox_root = inbox_root.resolve(strict=True)
        self.staging_root = staging_root.resolve(strict=True)
        self.store_root = store_root.resolve(strict=True)
        # os.replace 只有在同一文件系统才具备真正原子语义，启动时提前拒绝错误布局。
        if self.staging_root.stat().st_dev != self.store_root.stat().st_dev:
            raise ValueError("暂存目录与模型仓库必须位于同一文件系统")

    @staticmethod
    def _notify(
        callback: ProgressCallback | None, stage: str, done: int, total: int | None
    ) -> None:
        if callback:
            callback(stage, done, total)

    @staticmethod
    def _check_cancelled(callback: CancelCallback | None) -> None:
        if callback and callback():
            raise ImportCancelledError("模型导入已取消")

    def _copy_controlled_directory(
        self,
        source: Path,
        destination: Path,
        progress: ProgressCallback | None,
        cancelled: CancelCallback | None,
    ) -> None:
        try:
            source = resolve_inside(self.inbox_root, source)
            files = list(iter_regular_files(source))
        except UnsafePathError as exc:
            raise ValueError(str(exc)) from exc
        total = sum(path.stat().st_size for path, _ in files)
        free = shutil.disk_usage(self.staging_root).free
        reserve = max(1024**3, total // 10)
        if free < total + reserve:
            raise OSError(f"模型导入空间不足：需要 {total + reserve} 字节（含安全余量）")
        done = 0
        for path, relative in files:
            self._check_cancelled(cancelled)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as reader, target.open("xb") as writer:
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    self._check_cancelled(cancelled)
                    writer.write(chunk)
                    done += len(chunk)
                    self._notify(progress, "transferring", done, total)

    def run(
        self,
        request: ImportRequest,
        *,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> tuple[Path, ModelManifest]:
        """完成一次不可重入导入，返回最终目录及其校验清单。"""

        staging = self.staging_root / str(request.import_id)
        final = self.store_root / str(request.import_id)
        if staging.exists() or final.exists():
            raise FileExistsError("导入 ID 已被使用")
        staging.mkdir(mode=0o750)

        try:
            download_result: DownloadResult | None = None
            self._check_cancelled(cancelled)
            if request.source == ModelSource.CONTROLLED_DIRECTORY:
                if request.source_directory is None:
                    raise ValueError("受控目录导入必须提供 source_directory")
                self._copy_controlled_directory(
                    request.source_directory, staging, progress, cancelled
                )
            elif request.source == ModelSource.HUGGINGFACE:
                if not request.repository:
                    raise ValueError("Hugging Face 导入必须提供 repository")
                download_result = download_huggingface(
                    request.repository,
                    request.revision,
                    staging,
                    request.access_token,
                    progress=progress,
                    cancelled=cancelled,
                )
            elif request.source == ModelSource.MODELSCOPE:
                if not request.repository:
                    raise ValueError("ModelScope 导入必须提供 repository")
                download_result = download_modelscope(
                    request.repository,
                    request.revision,
                    staging,
                    request.access_token,
                    progress=progress,
                    cancelled=cancelled,
                )
            else:  # pragma: no cover - StrEnum 已限制取值，保留防御性分支
                raise ValueError(f"不支持的模型来源: {request.source}")

            self._check_cancelled(cancelled)
            manifest = validate_model_directory(
                staging,
                progress=progress,
                cancelled=cancelled,
                requested_revision=(
                    request.revision
                    if request.source in {ModelSource.HUGGINGFACE, ModelSource.MODELSCOPE}
                    else None
                ),
                resolved_revision=(
                    download_result.resolved_revision if download_result is not None else None
                ),
            )
            self._check_cancelled(cancelled)
            (staging / GENERATED_MANIFEST_NAME).write_text(
                json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 清单 fsync 后再原子改名，避免掉电后出现名义 ready、内容未落盘的目录。
            manifest_fd = os.open(staging / GENERATED_MANIFEST_NAME, os.O_RDONLY)
            try:
                os.fsync(manifest_fd)
            finally:
                os.close(manifest_fd)
            self._check_cancelled(cancelled)
            os.replace(staging, final)
            store_fd = os.open(self.store_root, os.O_RDONLY)
            try:
                os.fsync(store_fd)
            finally:
                os.close(store_fd)
            self._notify(progress, "ready", manifest.total_size_bytes, manifest.total_size_bytes)
            return final, manifest
        except (DownloadCancelledError, ModelValidationCancelled) as exc:
            if staging.exists():
                shutil.rmtree(staging)
            raise ImportCancelledError(str(exc)) from exc
        except Exception:
            # 删除范围由 import_id 精确确定且已经验证在 staging_root 下，不接收外部路径。
            if staging.exists():
                shutil.rmtree(staging)
            raise
