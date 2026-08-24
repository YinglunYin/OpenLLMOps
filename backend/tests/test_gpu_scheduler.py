import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.models import GPULease
from app.models.enums import LeaseOwnerType
from app.services.gpu_scheduler import GPULeaseManager, LeaseOwner


async def test_group_acquisition_is_all_or_nothing_and_exclusive(
    isolated_session_factory,
) -> None:
    manager = GPULeaseManager(ttl_seconds=30)
    inference = LeaseOwner(LeaseOwnerType.DEPLOYMENT, uuid.uuid4(), "chat", 2)
    training = LeaseOwner(LeaseOwnerType.TRAINING, uuid.uuid4(), "sft", 1)
    now = datetime(2026, 8, 24, tzinfo=UTC)

    async with isolated_session_factory() as session, session.begin():
        acquired = await manager.try_acquire(session, inference, [1, 0], now=now)
        assert acquired.acquired
        assert acquired.gpu_ids == (0, 1)

    async with isolated_session_factory() as session, session.begin():
        blocked = await manager.try_acquire(session, training, [1, 2], now=now)
        assert not blocked.acquired
        assert blocked.blocking_owner == inference

    async with isolated_session_factory() as session:
        leases = list(await session.scalars(select(GPULease).order_by(GPULease.gpu_index)))
        # GPU 2 没有成为“半组租约”，证明任一不足时整组申请均不落库。
        assert [lease.gpu_index for lease in leases] == [0, 1]
        assert len({lease.lease_group_id for lease in leases}) == 1


async def test_heartbeat_idempotency_generation_and_confirmed_ttl_reaping(
    isolated_session_factory,
) -> None:
    manager = GPULeaseManager(ttl_seconds=20)
    owner = LeaseOwner(LeaseOwnerType.EVALUATION, uuid.uuid4(), "ceval", 3)
    acquired_at = datetime(2026, 8, 24, tzinfo=UTC)

    async with isolated_session_factory() as session, session.begin():
        first = await manager.try_acquire(session, owner, [0, 1], now=acquired_at)
        second = await manager.try_acquire(
            session,
            owner,
            [1, 0],
            now=acquired_at + timedelta(seconds=5),
        )
        assert first.lease_group_id == second.lease_group_id
        assert second.idempotent

    async with isolated_session_factory() as session, session.begin():
        renewed = await manager.heartbeat(
            session,
            owner,
            now=acquired_at + timedelta(seconds=10),
        )
        assert renewed == 2

    async with isolated_session_factory() as session:
        leases = list(await session.scalars(select(GPULease)))
        assert {lease.expires_at for lease in leases} == {
            (acquired_at + timedelta(seconds=30)).replace(tzinfo=None)
        }

    expired_at = acquired_at + timedelta(seconds=31)
    async with (
        isolated_session_factory() as session,
        session.begin(),
        manager.scheduler_lock(session),
    ):
        # 只过 TTL 不能回收：调用者还必须提供 node-agent/终态确认。
        not_reaped = await manager.reap_expired_locked(session, set(), now=expired_at)
        assert not_reaped == []

    key = {(owner.type, owner.id, owner.generation)}
    async with (
        isolated_session_factory() as session,
        session.begin(),
        manager.scheduler_lock(session),
    ):
        reaped = await manager.reap_expired_locked(session, key, now=expired_at)
        assert reaped == [owner]

    async with isolated_session_factory() as session:
        assert list(await session.scalars(select(GPULease))) == []
