import asyncio
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from openllmops_model_importer import ImportRequest, ModelManifest
from openllmops_model_importer.importer import ImportCancelledError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, ModelAsset, ModelImportJob
from app.models.enums import (
    AssetStatus,
    ModelImportSource,
    ModelImportStatus,
    ModelKind,
    ModelSourceType,
)
from app.services.model_import_coordinator import (
    ModelImportCoordinator,
    claim_next_model_import,
    isolated_online_credentials,
    read_secret_file,
    recover_stale_model_imports,
    request_model_import_cancel,
)


@pytest_asyncio.fixture
async def import_session_factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """文件型 SQLite 允许用两个连接真实覆盖 CAS 双领取竞争。"""

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'imports.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield factory
    await engine.dispose()


async def _seed_import(
    factory,  # type: ignore[no-untyped-def]
    *,
    source: ModelImportSource = ModelImportSource.CONTROLLED_DIRECTORY,
    repository: str | None = None,
    source_directory: str | None = "candidate",
    revision: str | None = None,
) -> uuid.UUID:
    async with factory() as session, session.begin():
        job = ModelImportJob(
            name=f"import-{uuid.uuid4()}",
            source=source,
            repository=repository,
            revision=revision,
            source_directory=source_directory,
            model_kind=ModelKind.BASE,
            status=ModelImportStatus.PENDING,
            progress_completed=0,
        )
        session.add(job)
        await session.flush()
        return job.id


class SuccessfulImporter:
    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path

    def run(self, request: ImportRequest, *, progress=None, cancelled=None):  # type: ignore[no-untyped-def]
        assert cancelled is not None and not cancelled()
        if progress is not None:
            progress("transferring", 5, 10)
            progress("validating", 0, None)
        manifest = ModelManifest(
            model_type="qwen2",
            architecture="Qwen2ForCausalLM",
            total_size_bytes=10,
            file_count=2,
            files=(),
            parameter_count=7_000_000_000,
            weight_dtypes=("BF16",),
            checksum="c" * 64,
            requested_revision=(request.revision if request.source.value != "controlled_directory" else None),
            resolved_revision=("a" * 40 if request.source.value != "controlled_directory" else None),
        )
        return self.final_path / str(request.import_id), manifest


class FailingImporter:
    def __init__(self) -> None:
        self.received_token: str | None = None

    def run(self, request: ImportRequest, *, progress=None, cancelled=None):  # type: ignore[no-untyped-def]
        self.received_token = request.access_token
        raise ValueError(f"远端下载失败：{request.access_token}")


class CancelableImporter:
    def __init__(self) -> None:
        self.started = threading.Event()

    def run(self, request: ImportRequest, *, progress=None, cancelled=None):  # type: ignore[no-untyped-def]
        del request, progress
        assert cancelled is not None
        self.started.set()
        while not cancelled():
            time.sleep(0.001)
        raise ImportCancelledError("测试主动取消")


class LateCompletingImporter:
    """模拟 worker 已越过最后取消检查、即将完成原子移动的竞态。"""

    def __init__(self, store_root: Path) -> None:
        self.store_root = store_root
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, request: ImportRequest, *, progress=None, cancelled=None):  # type: ignore[no-untyped-def]
        del progress, cancelled
        self.started.set()
        assert self.release.wait(2)
        final_path = self.store_root / str(request.import_id)
        final_path.mkdir(parents=True)
        return final_path, ModelManifest(
            model_type="qwen2",
            architecture="Qwen2ForCausalLM",
            total_size_bytes=10,
            file_count=2,
            files=(),
        )


async def test_sqlite_atomic_claim_does_not_double_claim(import_session_factory) -> None:
    job_id = await _seed_import(import_session_factory)

    results = await asyncio.gather(
        claim_next_model_import(import_session_factory, "worker-a"),
        claim_next_model_import(import_session_factory, "worker-b"),
    )

    claimed = [job for job in results if job is not None]
    assert len(claimed) == 1
    assert claimed[0].id == job_id
    async with import_session_factory() as session:
        job = await session.get(ModelImportJob, job_id)
        assert job is not None
        assert job.status == ModelImportStatus.TRANSFERRING
        assert job.claimed_by in {"worker-a", "worker-b"}


async def test_success_creates_ready_asset_only_after_validation(
    import_session_factory,
    tmp_path: Path,
) -> None:
    job_id = await _seed_import(import_session_factory)
    coordinator = ModelImportCoordinator(
        import_session_factory,
        SuccessfulImporter(tmp_path / "models"),
        inbox_root=tmp_path / "inbox",
        staging_root=tmp_path / "staging",
    )

    await coordinator.run_once()
    await coordinator.wait_for_idle()

    async with import_session_factory() as session:
        job = await session.get(ModelImportJob, job_id)
        assets = list(await session.scalars(select(ModelAsset)))
        assert job is not None and job.status == ModelImportStatus.READY, job.error_message if job else None
        assert job.result_asset_id is not None
        assert job.progress_completed == 10 == job.progress_total
        assert len(assets) == 1
        assert assets[0].id == job.result_asset_id
        assert assets[0].status == AssetStatus.READY
        assert assets[0].family == "qwen2"
        assert assets[0].parameter_count == 7_000_000_000
        assert assets[0].checksum == "c" * 64
        assert assets[0].metadata_json["weight_dtypes"] == ["BF16"]


async def test_online_success_persists_requested_and_resolved_revisions(
    import_session_factory,
    tmp_path: Path,
) -> None:
    job_id = await _seed_import(
        import_session_factory,
        source=ModelImportSource.HUGGINGFACE,
        repository="Qwen/Test",
        source_directory=None,
        revision="main",
    )
    coordinator = ModelImportCoordinator(
        import_session_factory,
        SuccessfulImporter(tmp_path / "models"),
        inbox_root=tmp_path / "inbox",
        staging_root=tmp_path / "staging",
    )

    await coordinator.run_once()
    await coordinator.wait_for_idle()

    async with import_session_factory() as session:
        job = await session.get(ModelImportJob, job_id)
        asset = await session.get(ModelAsset, job.result_asset_id if job else None)
        assert job is not None and job.status == ModelImportStatus.READY
        assert job.revision == "main"
        assert job.manifest_json is not None
        assert job.manifest_json["requested_revision"] == "main"
        assert job.manifest_json["resolved_revision"] == "a" * 40
        assert asset is not None and asset.revision == "a" * 40
        assert asset.metadata_json["requested_revision"] == "main"
        assert asset.metadata_json["resolved_revision"] == "a" * 40


async def test_failure_redacts_secret_and_never_creates_asset(
    import_session_factory,
    tmp_path: Path,
) -> None:
    secret = "hf_private_value"
    secret_file = tmp_path / "hf-token"
    secret_file.write_text(secret, encoding="utf-8")
    secret_file.chmod(0o400)
    job_id = await _seed_import(
        import_session_factory,
        source=ModelImportSource.HUGGINGFACE,
        repository="example/model",
        source_directory=None,
    )
    importer = FailingImporter()
    coordinator = ModelImportCoordinator(
        import_session_factory,
        importer,
        inbox_root=tmp_path / "inbox",
        staging_root=tmp_path / "staging",
        huggingface_token_file=secret_file,
    )

    await coordinator.run_once()
    await coordinator.wait_for_idle()

    assert importer.received_token == secret
    async with import_session_factory() as session:
        job = await session.get(ModelImportJob, job_id)
        assert job is not None and job.status == ModelImportStatus.FAILED
        assert job.result_asset_id is None
        assert job.error_message is not None
        assert secret not in job.error_message
        assert "[REDACTED]" in job.error_message
        assert await session.scalar(select(func.count()).select_from(ModelAsset)) == 0


async def test_cancel_running_import_is_idempotent_and_creates_no_asset(
    import_session_factory,
    tmp_path: Path,
) -> None:
    job_id = await _seed_import(import_session_factory)
    importer = CancelableImporter()
    coordinator = ModelImportCoordinator(
        import_session_factory,
        importer,
        inbox_root=tmp_path / "inbox",
        staging_root=tmp_path / "staging",
    )

    await coordinator.run_once()
    assert await asyncio.to_thread(importer.started.wait, 2)
    async with import_session_factory() as session:
        first = await request_model_import_cancel(session, job_id)
        assert first is not None and first.status == ModelImportStatus.CANCELING
    await coordinator.run_once()
    await coordinator.wait_for_idle()
    async with import_session_factory() as session:
        second = await request_model_import_cancel(session, job_id)
        assert second is not None and second.status == ModelImportStatus.CANCELED
        assert await session.scalar(select(func.count()).select_from(ModelAsset)) == 0


async def test_atomic_move_is_ready_commit_point_for_late_cancel(
    import_session_factory,
    tmp_path: Path,
) -> None:
    job_id = await _seed_import(import_session_factory)
    store_root = tmp_path / "models"
    importer = LateCompletingImporter(store_root)
    coordinator = ModelImportCoordinator(
        import_session_factory,
        importer,
        inbox_root=tmp_path / "inbox",
        staging_root=tmp_path / "staging",
        store_root=store_root,
    )

    await coordinator.run_once()
    assert await asyncio.to_thread(importer.started.wait, 2)
    async with import_session_factory() as session:
        canceling = await request_model_import_cancel(session, job_id)
        assert canceling is not None and canceling.status == ModelImportStatus.CANCELING
    # 不再触发协调器取消轮询，模拟请求落在 worker 最后检查点之后。
    importer.release.set()
    await coordinator.wait_for_idle()

    async with import_session_factory() as session:
        job = await session.get(ModelImportJob, job_id)
        asset = await session.get(ModelAsset, job.result_asset_id if job else None)
        assert job is not None and job.status == ModelImportStatus.READY
        assert asset is not None and asset.local_path == str(store_root / str(job_id))
        assert Path(asset.local_path).is_dir()


async def test_stale_claim_recovery_is_safe_and_does_not_delete_published_asset(
    import_session_factory,
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    old = now - timedelta(minutes=10)
    staging_root = tmp_path / "staging"
    store_root = tmp_path / "models"
    protected_id = uuid.uuid4()
    async with import_session_factory() as session, session.begin():
        stale = ModelImportJob(
            name="stale-transfer",
            source=ModelImportSource.CONTROLLED_DIRECTORY,
            source_directory="candidate",
            model_kind=ModelKind.BASE,
            status=ModelImportStatus.TRANSFERRING,
            progress_completed=50,
            claimed_by="dead-worker",
            claimed_at=old,
            started_at=old,
        )
        canceling = ModelImportJob(
            name="stale-cancel",
            source=ModelImportSource.CONTROLLED_DIRECTORY,
            source_directory="candidate",
            model_kind=ModelKind.BASE,
            status=ModelImportStatus.CANCELING,
            progress_completed=50,
            claimed_by="dead-worker",
            claimed_at=old,
            started_at=old,
        )
        fresh = ModelImportJob(
            name="current-worker-transfer",
            source=ModelImportSource.CONTROLLED_DIRECTORY,
            source_directory="candidate",
            model_kind=ModelKind.BASE,
            status=ModelImportStatus.TRANSFERRING,
            progress_completed=1,
            # 即使时间异常偏旧，也不能恢复当前仍持有任务的协调器。
            claimed_by="new-worker",
            claimed_at=old,
            started_at=old,
        )
        published_asset = ModelAsset(
            name="published",
            source_type=ModelSourceType.MANUAL,
            local_path=str(store_root.resolve() / str(protected_id)),
            model_kind=ModelKind.BASE,
            status=AssetStatus.READY,
        )
        session.add_all([stale, canceling, fresh, published_asset])
        await session.flush()
        protected = ModelImportJob(
            id=protected_id,
            name="published-active-anomaly",
            source=ModelImportSource.CONTROLLED_DIRECTORY,
            source_directory="candidate",
            model_kind=ModelKind.BASE,
            status=ModelImportStatus.TRANSFERRING,
            progress_completed=1,
            claimed_by="dead-worker",
            claimed_at=old,
            started_at=old,
            result_asset_id=published_asset.id,
        )
        session.add(protected)
        await session.flush()
        ids = stale.id, canceling.id, fresh.id, protected.id

    for job_id in ids:
        (staging_root / str(job_id)).mkdir(parents=True)
        (staging_root / str(job_id) / "partial").write_text("partial", encoding="utf-8")
        (store_root / str(job_id)).mkdir(parents=True)
        (store_root / str(job_id) / "sentinel").write_text("keep?", encoding="utf-8")

    recovered = await recover_stale_model_imports(
        import_session_factory,
        staging_root=staging_root,
        store_root=store_root,
        worker_id="new-worker",
        claim_timeout_seconds=120,
        now=now,
    )
    assert recovered == 3

    async with import_session_factory() as session:
        stale, canceling, fresh, protected = [await session.get(ModelImportJob, job_id) for job_id in ids]
        assert stale is not None and stale.status == ModelImportStatus.PENDING
        assert stale.claimed_by is None and stale.progress_completed == 0
        assert canceling is not None and canceling.status == ModelImportStatus.CANCELED
        assert fresh is not None and fresh.status == ModelImportStatus.TRANSFERRING
        assert protected is not None and protected.status == ModelImportStatus.READY
    assert not (staging_root / str(ids[0])).exists()
    assert not (store_root / str(ids[0])).exists()
    assert not (store_root / str(ids[1])).exists()
    assert (store_root / str(ids[2]) / "sentinel").is_file()
    assert (store_root / str(ids[3]) / "sentinel").is_file()


def test_secret_file_must_be_regular_and_read_only(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    token_file.chmod(0o400)
    assert read_secret_file(token_file) == "secret"

    token_file.chmod(0o600)
    with pytest.raises(ValueError, match="只读权限"):
        read_secret_file(token_file)


def test_modelscope_credentials_and_cache_are_isolated_and_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "host-token")
    monkeypatch.setenv("MODELSCOPE_CREDENTIALS_PATH", "/host/modelscope/credentials")
    monkeypatch.setenv("MODELSCOPE_CACHE", "/host/modelscope/cache")
    runtime_home = tmp_path / "sdk-home"

    with isolated_online_credentials(
        ModelImportSource.MODELSCOPE,
        "job-token",
        runtime_home,
    ):
        assert os.environ["MODELSCOPE_API_TOKEN"] == "job-token"
        assert os.environ["MODELSCOPE_CREDENTIALS_PATH"] == str(runtime_home / "modelscope" / "credentials")
        assert os.environ["MODELSCOPE_CACHE"] == str(runtime_home / "modelscope" / "cache")

    assert os.environ["MODELSCOPE_API_TOKEN"] == "host-token"
    assert os.environ["MODELSCOPE_CREDENTIALS_PATH"] == "/host/modelscope/credentials"
    assert os.environ["MODELSCOPE_CACHE"] == "/host/modelscope/cache"


def test_model_import_api_and_inbox_reject_directory_escape(client, test_root: Path) -> None:
    inbox = test_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    candidate_name = f"candidate-{uuid.uuid4()}"
    candidate = inbox / candidate_name
    candidate.mkdir()
    (candidate / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (candidate / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (candidate / "tokenizer.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    (candidate / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")

    inbox_response = client.get("/api/v1/model-inbox")
    assert inbox_response.status_code == 200
    assert any(item["name"] == candidate_name for item in inbox_response.json())

    escaped = client.post(
        "/api/v1/model-imports",
        json={
            "name": "escape",
            "source": "controlled_directory",
            "source_directory": "../outside",
            "model_kind": "base",
        },
    )
    assert escaped.status_code == 422

    leaked_secret = client.post(
        "/api/v1/model-imports",
        json={
            "name": "secret-in-request",
            "source": "huggingface",
            "repository": "example/model",
            "model_kind": "base",
            "access_token": "must-not-be-accepted",
        },
    )
    assert leaked_secret.status_code == 422

    created = client.post(
        "/api/v1/model-imports",
        json={
            "name": "controlled-import",
            "source": "controlled_directory",
            "source_directory": candidate_name,
            "model_kind": "base",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "pending"

    detail = client.get(f"/api/v1/model-imports/{body['id']}")
    assert detail.status_code == 200
    canceled = client.post(f"/api/v1/model-imports/{body['id']}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    repeated = client.post(f"/api/v1/model-imports/{body['id']}/cancel")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "canceled"
