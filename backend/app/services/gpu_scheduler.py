import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import GPULease
from app.models.enums import LeaseOwnerType

# 固定 advisory lock key 只在本应用数据库中使用，用于串行化所有资源分配事务。
POSTGRES_SCHEDULER_LOCK_KEY = 1330396237


class LeaseInvariantError(RuntimeError):
    """同一工作负载出现跨代或部分租约等不安全状态。"""


@dataclass(frozen=True, slots=True)
class LeaseOwner:
    type: LeaseOwnerType
    id: uuid.UUID
    name: str
    generation: int


@dataclass(frozen=True, slots=True)
class LeaseAcquisition:
    acquired: bool
    gpu_ids: tuple[int, ...]
    lease_group_id: uuid.UUID | None = None
    blocking_owner: LeaseOwner | None = None
    idempotent: bool = False


class GPULeaseManager:
    """管理整卡独占、全有或全无的 GPU 租约。

    生产 PostgreSQL 路径先取得事务级 advisory lock，再以 ``FOR UPDATE`` 读取冲突
    行；GPU 唯一约束作为最后一道并发安全网。插入整组租约和工作负载状态迁移由调用
    方放在同一事务中，因此任一 GPU 不可用时不会遗留部分租约。

    SQLite 不支持行锁或 advisory lock，仅用进程内 asyncio 锁模拟串行化，专供单元
    测试和单进程开发。它不是多进程生产调度的一致性实现。
    """

    def __init__(self, ttl_seconds: int = 30) -> None:
        if ttl_seconds < 1:
            raise ValueError("GPU 租约 TTL 必须大于 0")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._sqlite_lock = asyncio.Lock()

    @staticmethod
    def _normalize_gpu_ids(gpu_ids: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(sorted(gpu_ids))
        if not normalized:
            raise ValueError("至少申请一张 GPU")
        if normalized[0] < 0 or len(normalized) != len(set(normalized)):
            raise ValueError("GPU 编号必须是互不重复的非负整数")
        return normalized

    @asynccontextmanager
    async def scheduler_lock(self, session: AsyncSession) -> AsyncIterator[None]:
        """取得覆盖当前事务的全局调度锁。"""

        dialect_name = session.get_bind().dialect.name
        if dialect_name == "postgresql":
            # advisory xact lock 随提交/回滚自动释放，不会因进程异常永久占锁。
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": POSTGRES_SCHEDULER_LOCK_KEY},
            )
            yield
            return
        if dialect_name == "sqlite":
            async with self._sqlite_lock:
                yield
            return
        raise RuntimeError(f"不支持在 {dialect_name} 数据库上执行 GPU 调度")

    async def try_acquire(
        self,
        session: AsyncSession,
        owner: LeaseOwner,
        gpu_ids: Sequence[int],
        *,
        now: datetime | None = None,
    ) -> LeaseAcquisition:
        """在调用方事务内原子申请整组 GPU。"""

        async with self.scheduler_lock(session):
            return await self.try_acquire_locked(session, owner, gpu_ids, now=now)

    async def try_acquire_locked(
        self,
        session: AsyncSession,
        owner: LeaseOwner,
        gpu_ids: Sequence[int],
        *,
        now: datetime | None = None,
    ) -> LeaseAcquisition:
        """已持有 ``scheduler_lock`` 时申请租约，避免重复嵌套 SQLite 锁。"""

        requested = self._normalize_gpu_ids(gpu_ids)
        current_time = now or datetime.now(UTC)

        owned = list(
            await session.scalars(
                select(GPULease)
                .where(
                    GPULease.owner_type == owner.type,
                    GPULease.owner_id == owner.id,
                )
                .order_by(GPULease.gpu_index)
                .with_for_update()
            )
        )
        if owned:
            owned_gpu_ids = tuple(lease.gpu_index for lease in owned)
            generations = {lease.generation for lease in owned}
            groups = {lease.lease_group_id for lease in owned}
            if owned_gpu_ids != requested or generations != {owner.generation} or len(groups) != 1:
                raise LeaseInvariantError("工作负载已有不同 GPU、代次或不完整的租约组")
            expires_at = current_time + self._ttl
            for lease in owned:
                lease.heartbeat_at = current_time
                lease.expires_at = expires_at
            await session.flush()
            return LeaseAcquisition(
                acquired=True,
                gpu_ids=requested,
                lease_group_id=owned[0].lease_group_id,
                idempotent=True,
            )

        conflicts = list(
            await session.scalars(
                select(GPULease)
                .where(GPULease.gpu_index.in_(requested))
                .order_by(GPULease.gpu_index)
                .with_for_update()
            )
        )
        if conflicts:
            conflict = conflicts[0]
            return LeaseAcquisition(
                acquired=False,
                gpu_ids=requested,
                blocking_owner=LeaseOwner(
                    type=conflict.owner_type,
                    id=conflict.owner_id,
                    name=conflict.owner_name,
                    generation=conflict.generation,
                ),
            )

        lease_group_id = uuid.uuid4()
        expires_at = current_time + self._ttl
        leases = [
            GPULease(
                gpu_index=gpu_id,
                lease_group_id=lease_group_id,
                owner_type=owner.type,
                owner_id=owner.id,
                owner_name=owner.name,
                generation=owner.generation,
                acquired_at=current_time,
                heartbeat_at=current_time,
                expires_at=expires_at,
            )
            for gpu_id in requested
        ]
        try:
            # SAVEPOINT 让唯一约束竞争只回滚本组插入，不污染调用方的状态迁移事务。
            async with session.begin_nested():
                session.add_all(leases)
                await session.flush()
        except IntegrityError:
            # 非调度器写入或部署期间旧代码仍可能触发竞争，数据库约束保证不会部分成功。
            conflict = await session.scalar(
                select(GPULease)
                .where(GPULease.gpu_index.in_(requested))
                .order_by(GPULease.gpu_index)
                .limit(1)
            )
            blocking_owner = None
            if conflict is not None:
                blocking_owner = LeaseOwner(
                    type=conflict.owner_type,
                    id=conflict.owner_id,
                    name=conflict.owner_name,
                    generation=conflict.generation,
                )
            return LeaseAcquisition(
                acquired=False,
                gpu_ids=requested,
                blocking_owner=blocking_owner,
            )
        return LeaseAcquisition(
            acquired=True,
            gpu_ids=requested,
            lease_group_id=lease_group_id,
        )

    async def heartbeat(
        self,
        session: AsyncSession,
        owner: LeaseOwner,
        *,
        now: datetime | None = None,
    ) -> int:
        """续期同一代工作负载的完整租约组，返回续期卡数。"""

        current_time = now or datetime.now(UTC)
        leases = list(
            await session.scalars(
                select(GPULease)
                .where(
                    GPULease.owner_type == owner.type,
                    GPULease.owner_id == owner.id,
                    GPULease.generation == owner.generation,
                )
                .with_for_update()
            )
        )
        expires_at = current_time + self._ttl
        for lease in leases:
            lease.heartbeat_at = current_time
            lease.expires_at = expires_at
        await session.flush()
        return len(leases)

    async def release(self, session: AsyncSession, owner: LeaseOwner) -> int:
        """仅释放指定工作负载代次的租约，防止迟到响应释放新一代任务。"""

        result = await session.execute(
            delete(GPULease).where(
                GPULease.owner_type == owner.type,
                GPULease.owner_id == owner.id,
                GPULease.generation == owner.generation,
            )
        )
        return int(result.rowcount or 0)

    async def reap_expired_locked(
        self,
        session: AsyncSession,
        confirmed_owners: set[tuple[LeaseOwnerType, uuid.UUID, int]],
        *,
        now: datetime | None = None,
    ) -> list[LeaseOwner]:
        """删除已确认可回收的过期租约，并返回受影响的唯一工作负载。

        调用方必须先从 node-agent 确认工作负载不存在或已结束。仅凭 TTL 直接复用 GPU
        可能与仍在运行但暂时失联的容器冲突；这种情况应隔离资源而不是冒险重分配。
        """

        current_time = now or datetime.now(UTC)
        expired = list(
            await session.scalars(
                select(GPULease)
                .where(GPULease.expires_at <= current_time)
                .order_by(GPULease.expires_at, GPULease.gpu_index)
                .with_for_update()
            )
        )
        owners: dict[tuple[LeaseOwnerType, uuid.UUID, int], LeaseOwner] = {}
        for lease in expired:
            key = (lease.owner_type, lease.owner_id, lease.generation)
            if key not in confirmed_owners:
                continue
            owners[key] = LeaseOwner(
                type=lease.owner_type,
                id=lease.owner_id,
                name=lease.owner_name,
                generation=lease.generation,
            )
            await session.delete(lease)
        await session.flush()
        return list(owners.values())
