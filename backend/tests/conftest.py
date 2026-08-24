import os
import tempfile
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

TEST_ROOT = Path(tempfile.mkdtemp(prefix="openllmops-backend-tests-")).resolve()
for directory in (
    TEST_ROOT / "models",
    TEST_ROOT / "datasets",
    TEST_ROOT / "evaluation-datasets",
    TEST_ROOT / "evaluation-output",
    TEST_ROOT / "checkpoints",
    TEST_ROOT / "runtime",
):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.update(
    {
        "ENVIRONMENT": "test",
        "AUTH_ENABLED": "false",
        "AUTO_CREATE_TABLES": "true",
        "DATABASE_URL": f"sqlite+aiosqlite:///{TEST_ROOT / 'test.db'}",
        "MODEL_ROOT": str(TEST_ROOT / "models"),
        "MODEL_INBOX_ROOT": str(TEST_ROOT / "inbox"),
        "DATASET_ROOT": str(TEST_ROOT / "datasets"),
        "CHECKPOINT_ROOT": str(TEST_ROOT / "checkpoints"),
        "EVALUATION_DATASET_ROOT": str(TEST_ROOT / "evaluation-datasets"),
        "EVALUATION_OUTPUT_ROOT": str(TEST_ROOT / "evaluation-output"),
        "NODE_AGENT_RUNTIME_ROOT": str(TEST_ROOT / "runtime"),
        "GPU_COUNT": "2",
    }
)

from app.core.database import AsyncSessionFactory
from app.main import app
from app.models import Base, ModelAsset
from app.models.enums import AssetStatus, ModelKind, ModelSourceType


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    # HTTPS 基址确保 TestClient 与真实浏览器一样遵守 Secure Cookie。
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client


@pytest.fixture(scope="session")
def test_root() -> Path:
    return TEST_ROOT


@pytest.fixture
def seed_model_asset(client: TestClient) -> Callable[..., dict[str, str]]:
    """测试专用数据库种子，不经过已从生产 API 移除的通用资产写入口。"""

    def seed(
        path: Path,
        *,
        kind: ModelKind | str = ModelKind.BASE,
        name: str | None = None,
    ) -> dict[str, str]:
        async def persist() -> dict[str, str]:
            async with AsyncSessionFactory() as session, session.begin():
                asset = ModelAsset(
                    name=name or f"test-model-{uuid.uuid4()}",
                    source_type=ModelSourceType.MANUAL,
                    local_path=str(path),
                    model_kind=ModelKind(kind),
                    status=AssetStatus.READY,
                )
                session.add(asset)
                await session.flush()
                return {
                    "id": str(asset.id),
                    "name": asset.name,
                    "local_path": asset.local_path,
                    "model_kind": asset.model_kind.value,
                }

        assert client.portal is not None
        return client.portal.call(persist)

    return seed


@pytest_asyncio.fixture
async def isolated_session_factory():  # type: ignore[no-untyped-def]
    """为调度并发与状态机测试提供互不污染的 SQLite 数据库。

    SQLite 只验证业务状态机；生产 PostgreSQL 的 advisory lock/行锁路径由服务代码
    明确实现，真正的跨进程互斥不能由 SQLite 假装覆盖。
    """

    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with test_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield factory
    await test_engine.dispose()
