import asyncio
import os
import shutil
import stat
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from openllmops_model_importer import (
    ImportRequest,
    ModelImporter,
    ModelManifest,
    ModelSource,
)
from openllmops_model_importer.importer import ImportCancelledError
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.models import ModelAsset, ModelImportJob
from app.models.enums import (
    AssetStatus,
    ModelImportSource,
    ModelImportStatus,
    ModelSourceType,
)

ONLINE_ENVIRONMENT_LOCK = threading.Lock()
ONLINE_SECRET_ENVIRONMENT_KEYS = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "HF_HOME",
    "MODELSCOPE_API_TOKEN",
    "MODELSCOPE_CACHE",
    "MODELSCOPE_CREDENTIALS_PATH",
    "MODELSCOPE_HOME",
    "MS_CACHE_HOME",
}
ACTIVE_IMPORT_STATUSES = {
    ModelImportStatus.TRANSFERRING,
    ModelImportStatus.VALIDATING,
    ModelImportStatus.CANCELING,
}
# 独立于 GPU scheduler 的锁命名空间（ASCII "MIMP"）。
MODEL_IMPORT_RECOVERY_LOCK_ID = 0x4D494D50
MODEL_IMPORT_INSTANCE_LOCK_ID = 0x4D494E53


class ImporterProtocol(Protocol):
    def run(
        self,
        request: ImportRequest,
        *,
        progress: Callable[[str, int, int | None], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[Path, ModelManifest]: ...


@asynccontextmanager
async def model_import_instance_lock(engine: AsyncEngine | None) -> AsyncIterator[None]:
    """生产 PostgreSQL 用会话级 advisory lock 强制只有一个导入协调器实例。"""

    if engine is None or engine.dialect.name != "postgresql":
        yield
        return
    async with engine.connect() as connection:
        acquired = await connection.scalar(select(func.pg_try_advisory_lock(MODEL_IMPORT_INSTANCE_LOCK_ID)))
        # 结束只读事务但保持同一个物理连接；会话级锁继续持有到 finally/断连。
        await connection.commit()
        if not acquired:
            raise RuntimeError("已有模型导入协调器实例持有 PostgreSQL 单例锁")
        try:
            yield
        finally:
            with suppress(Exception):
                await connection.scalar(select(func.pg_advisory_unlock(MODEL_IMPORT_INSTANCE_LOCK_ID)))
                await connection.commit()


def read_secret_file(path: Path | None) -> str | None:
    """只读取不可写普通文件；返回值仅存在于执行线程内存。"""

    if path is None:
        return None
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise ValueError("模型仓库访问令牌文件不可读") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("模型仓库访问令牌必须来自普通只读文件，不能使用软链接")
    if file_stat.st_mode & 0o222:
        raise ValueError("模型仓库访问令牌文件必须以只读权限挂载")
    if file_stat.st_size > 64 * 1024:
        raise ValueError("模型仓库访问令牌文件大小异常")
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("模型仓库访问令牌文件不可读") from exc
    if not token or "\x00" in token or "\n" in token or "\r" in token:
        raise ValueError("模型仓库访问令牌文件内容无效")
    return token


@contextmanager
def isolated_online_credentials(
    source: ModelImportSource,
    token: str | None,
    runtime_home: Path,
):  # type: ignore[no-untyped-def]
    """隔离 SDK 的隐式凭证来源，并在退出时恢复进程环境。"""

    with ONLINE_ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in ONLINE_SECRET_ENVIRONMENT_KEYS}
        try:
            for key in ONLINE_SECRET_ENVIRONMENT_KEYS:
                os.environ.pop(key, None)
            runtime_home.mkdir(parents=True, exist_ok=True, mode=0o750)
            if source == ModelImportSource.HUGGINGFACE:
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
                os.environ["HF_HOME"] = str(runtime_home / "huggingface")
            elif source == ModelImportSource.MODELSCOPE:
                os.environ["MODELSCOPE_HOME"] = str(runtime_home / "modelscope")
                os.environ["MODELSCOPE_CACHE"] = str(runtime_home / "modelscope" / "cache")
                os.environ["MODELSCOPE_CREDENTIALS_PATH"] = str(runtime_home / "modelscope" / "credentials")
                if token:
                    os.environ["MODELSCOPE_API_TOKEN"] = token
            yield
        finally:
            for key in ONLINE_SECRET_ENVIRONMENT_KEYS:
                os.environ.pop(key, None)
            for key, value in previous.items():
                if value is not None:
                    os.environ[key] = value


async def claim_next_model_import(
    session_factory: async_sessionmaker[AsyncSession],
    worker_id: str,
) -> ModelImportJob | None:
    """原子领取一个 FIFO 任务；PostgreSQL 使用 SKIP LOCKED 支持多协调器。"""

    now = datetime.now(UTC)
    async with session_factory() as session:
        dialect_name = session.bind.dialect.name if session.bind is not None else ""
        if dialect_name == "postgresql":
            async with session.begin():
                job = await session.scalar(
                    select(ModelImportJob)
                    .where(ModelImportJob.status == ModelImportStatus.PENDING)
                    .order_by(ModelImportJob.created_at, ModelImportJob.id)
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                if job is None:
                    return None
                job.status = ModelImportStatus.TRANSFERRING
                job.claimed_by = worker_id
                job.claimed_at = now
                job.started_at = now
                job.error_message = None
            return job

        # SQLite 没有 SKIP LOCKED，用单条 CAS UPDATE 保证测试/开发环境不会双领同一任务。
        candidate_id = (
            select(ModelImportJob.id)
            .where(ModelImportJob.status == ModelImportStatus.PENDING)
            .order_by(ModelImportJob.created_at, ModelImportJob.id)
            .limit(1)
            .scalar_subquery()
        )
        claimed_id = await session.scalar(
            update(ModelImportJob)
            .where(
                ModelImportJob.id == candidate_id,
                ModelImportJob.status == ModelImportStatus.PENDING,
            )
            .values(
                status=ModelImportStatus.TRANSFERRING,
                claimed_by=worker_id,
                claimed_at=now,
                started_at=now,
                error_message=None,
            )
            .returning(ModelImportJob.id)
        )
        await session.commit()
        if claimed_id is None:
            return None
        return await session.get(ModelImportJob, claimed_id)


async def request_model_import_cancel(
    session: AsyncSession,
    job_id: uuid.UUID,
) -> ModelImportJob | None:
    """用单条条件更新处理取消与 claim 竞争，重复取消不会改变终态。"""

    now = datetime.now(UTC)
    cancellable = {
        ModelImportStatus.PENDING,
        ModelImportStatus.TRANSFERRING,
        ModelImportStatus.VALIDATING,
    }
    updated_id = await session.scalar(
        update(ModelImportJob)
        .where(ModelImportJob.id == job_id)
        .values(
            status=case(
                (ModelImportJob.status == ModelImportStatus.PENDING, ModelImportStatus.CANCELED),
                (
                    ModelImportJob.status.in_({ModelImportStatus.TRANSFERRING, ModelImportStatus.VALIDATING}),
                    ModelImportStatus.CANCELING,
                ),
                else_=ModelImportJob.status,
            ),
            cancel_requested_at=case(
                (ModelImportJob.status.in_(cancellable), now),
                else_=ModelImportJob.cancel_requested_at,
            ),
            finished_at=case(
                (ModelImportJob.status == ModelImportStatus.PENDING, now),
                else_=ModelImportJob.finished_at,
            ),
        )
        .returning(ModelImportJob.id)
    )
    await session.commit()
    if updated_id is None:
        return None
    return await session.get(ModelImportJob, updated_id)


def _job_directory(root: Path, job_id: uuid.UUID) -> Path:
    """只构造固定 UUID 子目录，不接受任务或请求提供的路径片段。"""

    resolved_root = root.resolve(strict=False)
    candidate = resolved_root / str(job_id)
    if candidate.parent != resolved_root:  # pragma: no cover - UUID 字符串不可能越界
        raise ValueError("模型导入恢复目录越界")
    return candidate


def _remove_job_directory(root: Path, job_id: uuid.UUID) -> None:
    candidate = _job_directory(root, job_id)
    try:
        file_stat = candidate.lstat()
    except FileNotFoundError:
        return
    # 顶层软链接只删除链接自身，绝不跟随到受控目录之外。
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        candidate.unlink()
        return
    shutil.rmtree(candidate)


async def recover_stale_model_imports(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    staging_root: Path,
    store_root: Path,
    worker_id: str,
    claim_timeout_seconds: int,
    now: datetime | None = None,
) -> int:
    """恢复失去心跳的任务，并清理仅属于该任务且尚未发布的目录。

    PostgreSQL advisory transaction lock 让多个 API 实例不会同时做恢复；任务行锁同时
    串行化管理员取消。仍存活协调器每轮刷新 claimed_at，因此只有超过租约时间且不属于
    当前实例的任务会进入恢复路径。
    """

    recovered_at = now or datetime.now(UTC)
    cutoff = recovered_at - timedelta(seconds=claim_timeout_seconds)
    recovered = 0
    async with session_factory() as session, session.begin():
        dialect_name = session.bind.dialect.name if session.bind is not None else ""
        if dialect_name == "postgresql":
            locked = await session.scalar(
                select(func.pg_try_advisory_xact_lock(MODEL_IMPORT_RECOVERY_LOCK_ID))
            )
            if not locked:
                return 0
        statement = (
            select(ModelImportJob)
            .where(
                ModelImportJob.status.in_(ACTIVE_IMPORT_STATUSES),
                or_(ModelImportJob.claimed_at.is_(None), ModelImportJob.claimed_at < cutoff),
                or_(ModelImportJob.claimed_by.is_(None), ModelImportJob.claimed_by != worker_id),
            )
            .order_by(ModelImportJob.claimed_at, ModelImportJob.id)
        )
        if dialect_name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        jobs = list(await session.scalars(statement))
        for job in jobs:
            if job.result_asset_id is not None:
                # DB 事务一旦已关联资产，ready 是唯一安全终态，恢复绝不触碰资产目录。
                asset = await session.get(ModelAsset, job.result_asset_id)
                job.status = ModelImportStatus.READY
                job.finished_at = recovered_at
                job.error_message = None
                if asset is not None and asset.size_bytes is not None:
                    job.progress_completed = asset.size_bytes
                    job.progress_total = asset.size_bytes
                recovered += 1
                continue
            final_path = _job_directory(store_root, job.id)
            existing_asset = await session.scalar(
                select(ModelAsset.id).where(ModelAsset.local_path == str(final_path)).limit(1)
            )
            if existing_asset is not None:
                # 绝不删除已有资产路径；异常状态交给管理员处理，避免重复发布同一路径。
                job.status = ModelImportStatus.FAILED
                job.finished_at = recovered_at
                job.error_message = "恢复中发现最终目录已关联其他模型资产，请人工检查"
                recovered += 1
                continue
            try:
                await asyncio.to_thread(_remove_job_directory, staging_root, job.id)
                await asyncio.to_thread(_remove_job_directory, staging_root / ".sdk-home", job.id)
                await asyncio.to_thread(_remove_job_directory, store_root, job.id)
            except OSError as exc:
                job.status = ModelImportStatus.FAILED
                job.finished_at = recovered_at
                job.error_message = f"清理中断导入目录失败：{type(exc).__name__}"
                recovered += 1
                continue
            if job.status == ModelImportStatus.CANCELING:
                job.status = ModelImportStatus.CANCELED
                job.finished_at = recovered_at
            else:
                job.status = ModelImportStatus.PENDING
                job.started_at = None
                job.finished_at = None
                job.progress_completed = 0
                job.progress_total = None
                job.cancel_requested_at = None
            job.claimed_by = None
            job.claimed_at = None
            job.error_message = None
            recovered += 1
    return recovered


class ModelImportCoordinator:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        importer: ImporterProtocol,
        *,
        inbox_root: Path,
        staging_root: Path,
        store_root: Path | None = None,
        huggingface_token_file: Path | None = None,
        modelscope_token_file: Path | None = None,
        poll_interval_seconds: float = 1.0,
        concurrency: int = 1,
        claim_timeout_seconds: int = 120,
        worker_id: str | None = None,
        lock_engine: AsyncEngine | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.importer = importer
        self.inbox_root = inbox_root
        self.staging_root = staging_root
        self.store_root = store_root or staging_root.parent / "models"
        self.huggingface_token_file = huggingface_token_file
        self.modelscope_token_file = modelscope_token_file
        self.poll_interval_seconds = poll_interval_seconds
        self.concurrency = concurrency
        self.claim_timeout_seconds = claim_timeout_seconds
        self.worker_id = worker_id or f"import-coordinator-{uuid.uuid4()}"
        self.lock_engine = lock_engine
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._cancel_events: dict[uuid.UUID, threading.Event] = {}
        self._shutting_down = False

    async def _record_progress(
        self,
        job_id: uuid.UUID,
        stage: str,
        completed: int,
        total: int | None,
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(ModelImportJob, job_id)
            if job is None or job.status in {
                ModelImportStatus.READY,
                ModelImportStatus.FAILED,
                ModelImportStatus.CANCELED,
            }:
                return
            if job.status != ModelImportStatus.CANCELING:
                if stage == "validating" or stage == "ready":
                    job.status = ModelImportStatus.VALIDATING
                else:
                    job.status = ModelImportStatus.TRANSFERRING
            job.progress_completed = max(0, completed)
            job.progress_total = max(0, total) if total is not None else None
            job.claimed_at = datetime.now(UTC)
            await session.commit()

    def _token_file_for(self, source: ModelImportSource) -> Path | None:
        if source == ModelImportSource.HUGGINGFACE:
            return self.huggingface_token_file
        if source == ModelImportSource.MODELSCOPE:
            return self.modelscope_token_file
        return None

    def _run_blocking(
        self,
        job: ModelImportJob,
        token: str | None,
        progress: Callable[[str, int, int | None], None],
        cancelled: Callable[[], bool],
    ) -> tuple[Path, ModelManifest]:
        source = ModelSource(job.source.value)
        request = ImportRequest(
            import_id=job.id,
            source=source,
            repository=job.repository,
            revision=job.revision,
            source_directory=(self.inbox_root / job.source_directory if job.source_directory else None),
            # 仅在线程/隔离下载子进程内传递；不会进入任务、资产、命令行或日志。
            access_token=token,
        )
        if job.source == ModelImportSource.CONTROLLED_DIRECTORY:
            return self.importer.run(request, progress=progress, cancelled=cancelled)
        runtime_home = self.staging_root / ".sdk-home" / str(job.id)
        with isolated_online_credentials(job.source, token, runtime_home):
            return self.importer.run(request, progress=progress, cancelled=cancelled)

    @staticmethod
    def _safe_error(error: Exception, token: str | None) -> str:
        message = str(error) or type(error).__name__
        if token:
            message = message.replace(token, "[REDACTED]")
        return message[:4000]

    async def _finalize_ready(
        self,
        job_id: uuid.UUID,
        final_path: Path,
        manifest: ModelManifest,
    ) -> None:
        expected_path = _job_directory(self.store_root, job_id)
        if final_path.resolve(strict=False) != expected_path:
            raise RuntimeError("模型导入 worker 返回了受控仓库之外的最终路径")
        manifest_json = manifest.as_dict()
        async with self.session_factory() as session, session.begin():
            job = await session.get(ModelImportJob, job_id, with_for_update=True)
            if job is None:
                raise RuntimeError("模型导入任务在完成前被删除")
            # worker 的原子移动是发布提交点：到达这里的迟到取消不能覆盖 ready，否则会
            # 留下没有资产记录的最终目录。更早的取消仍由 worker 的检查点收敛为 canceled。
            source_type = {
                ModelImportSource.HUGGINGFACE: ModelSourceType.HUGGINGFACE,
                ModelImportSource.MODELSCOPE: ModelSourceType.MODELSCOPE,
                ModelImportSource.CONTROLLED_DIRECTORY: ModelSourceType.MANUAL,
            }[job.source]
            online_source = job.source in {
                ModelImportSource.HUGGINGFACE,
                ModelImportSource.MODELSCOPE,
            }
            if online_source:
                if manifest.requested_revision != job.revision:
                    raise RuntimeError("模型导入 worker 返回的 requested revision 与任务不一致")
                if not manifest.resolved_revision:
                    raise RuntimeError("在线模型导入缺少 resolved commit，拒绝创建不可复现资产")
            source_uri = job.repository if job.repository else f"inbox://{job.source_directory or 'unknown'}"
            asset = ModelAsset(
                name=job.name,
                source_type=source_type,
                source_uri=source_uri,
                revision=manifest.resolved_revision,
                local_path=str(final_path),
                model_kind=job.model_kind,
                format="safetensors",
                status=AssetStatus.READY,
                family=manifest.model_type,
                parameter_count=manifest.parameter_count,
                size_bytes=manifest.total_size_bytes,
                checksum=manifest.checksum,
                metadata_json={
                    "import_job_id": str(job.id),
                    "architecture": manifest.architecture,
                    "weight_dtypes": list(manifest.weight_dtypes),
                    "requested_revision": manifest.requested_revision,
                    "resolved_revision": manifest.resolved_revision,
                    "manifest": manifest_json,
                },
            )
            session.add(asset)
            await session.flush()
            job.result_asset_id = asset.id
            job.manifest_json = manifest_json
            job.status = ModelImportStatus.READY
            job.progress_completed = manifest.total_size_bytes
            job.progress_total = manifest.total_size_bytes
            job.finished_at = datetime.now(UTC)
            job.error_message = None

    async def _mark_failed(
        self,
        job_id: uuid.UUID,
        error: Exception,
        token: str | None,
    ) -> None:
        async with self.session_factory() as session:
            job = await session.get(ModelImportJob, job_id)
            if job is None:
                return
            if job.status == ModelImportStatus.CANCELING:
                job.status = ModelImportStatus.CANCELED
                job.error_message = None
            else:
                job.status = ModelImportStatus.FAILED
                job.error_message = self._safe_error(error, token)
            job.finished_at = datetime.now(UTC)
            await session.commit()

    async def _mark_canceled_or_retry(self, job_id: uuid.UUID) -> None:
        async with self.session_factory() as session:
            job = await session.get(ModelImportJob, job_id)
            if job is None:
                return
            if self._shutting_down and job.status != ModelImportStatus.CANCELING:
                job.status = ModelImportStatus.PENDING
                job.claimed_by = None
                job.claimed_at = None
                job.started_at = None
                job.progress_completed = 0
                job.progress_total = None
            else:
                job.status = ModelImportStatus.CANCELED
                job.finished_at = datetime.now(UTC)
            await session.commit()

    async def _execute(self, job_id: uuid.UUID, cancel_event: threading.Event) -> None:
        token: str | None = None
        try:
            async with self.session_factory() as session:
                job = await session.get(ModelImportJob, job_id)
                if job is None:
                    return
                # 脱离 session 前把 ORM 标量全部加载；expire_on_commit=False 保证线程只读安全。
                source = job.source
                if job.status == ModelImportStatus.CANCELING:
                    cancel_event.set()
            token = await asyncio.to_thread(read_secret_file, self._token_file_for(source))
            loop = asyncio.get_running_loop()

            def progress(stage: str, completed: int, total: int | None) -> None:
                future = asyncio.run_coroutine_threadsafe(
                    self._record_progress(job_id, stage, completed, total),
                    loop,
                )
                future.result(timeout=30)

            final_path, manifest = await asyncio.to_thread(
                self._run_blocking,
                job,
                token,
                progress,
                cancel_event.is_set,
            )
            await self._finalize_ready(job_id, final_path, manifest)
        except ImportCancelledError:
            await self._mark_canceled_or_retry(job_id)
        except Exception as exc:
            await self._mark_failed(job_id, exc, token)

    async def _signal_canceling_jobs(self) -> None:
        if not self._tasks:
            return
        async with self.session_factory() as session:
            ids = set(
                await session.scalars(
                    select(ModelImportJob.id).where(
                        ModelImportJob.id.in_(self._tasks),
                        ModelImportJob.status == ModelImportStatus.CANCELING,
                    )
                )
            )
        for job_id in ids:
            self._cancel_events[job_id].set()

    async def _collect_finished(self) -> None:
        for job_id, task in list(self._tasks.items()):
            if not task.done():
                continue
            await task
            self._tasks.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

    async def _heartbeat_active_jobs(self) -> None:
        if not self._tasks:
            return
        async with self.session_factory() as session:
            await session.execute(
                update(ModelImportJob)
                .where(
                    ModelImportJob.id.in_(self._tasks),
                    ModelImportJob.claimed_by == self.worker_id,
                    ModelImportJob.status.in_(ACTIVE_IMPORT_STATUSES),
                )
                .values(claimed_at=datetime.now(UTC))
            )
            await session.commit()

    async def _recover_stale_jobs(self) -> int:
        return await recover_stale_model_imports(
            self.session_factory,
            staging_root=self.staging_root,
            store_root=self.store_root,
            worker_id=self.worker_id,
            claim_timeout_seconds=self.claim_timeout_seconds,
        )

    async def run_once(self) -> None:
        await self._signal_canceling_jobs()
        await self._collect_finished()
        await self._heartbeat_active_jobs()
        await self._recover_stale_jobs()
        while len(self._tasks) < self.concurrency:
            job = await claim_next_model_import(self.session_factory, self.worker_id)
            if job is None:
                break
            cancel_event = threading.Event()
            self._cancel_events[job.id] = cancel_event
            self._tasks[job.id] = asyncio.create_task(
                self._execute(job.id, cancel_event),
                name=f"model-import-{job.id}",
            )

    async def wait_for_idle(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks.values())
        await self._collect_finished()

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        async with model_import_instance_lock(self.lock_engine):
            try:
                while not stop_event.is_set():
                    await self.run_once()
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_seconds)
            finally:
                self._shutting_down = True
                for cancel_event in self._cancel_events.values():
                    cancel_event.set()
                await self.wait_for_idle()


def build_model_importer(inbox_root: Path, staging_root: Path, store_root: Path) -> ModelImporter:
    for directory in (inbox_root, staging_root, store_root):
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
    return ModelImporter(
        inbox_root=inbox_root,
        staging_root=staging_root,
        store_root=store_root,
    )
