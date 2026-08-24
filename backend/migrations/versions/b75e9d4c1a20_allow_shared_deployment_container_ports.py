"""allow shared deployment container ports

Revision ID: b75e9d4c1a20
Revises: c4f3a81d2e70
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b75e9d4c1a20"
down_revision: str | None = "c4f3a81d2e70"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 每个 vLLM 容器拥有独立网络命名空间，容器内端口不需要全局唯一。
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_constraint(op.f("uq_deployments_port"), type_="unique")


def downgrade() -> None:
    deployments = sa.table("deployments", sa.column("port", sa.Integer()))
    duplicate_ports = list(
        op.get_bind()
        .execute(
            sa.select(deployments.c.port)
            .where(deployments.c.port.is_not(None))
            .group_by(deployments.c.port)
            .having(sa.func.count() > 1)
            .limit(10)
        )
        .scalars()
    )
    if duplicate_ports:
        ports = ", ".join(str(port) for port in duplicate_ports)
        raise RuntimeError(
            "无法回退端口唯一约束：已有多个部署共享容器端口 "
            f"{ports}。请先删除重复部署或为它们手工分配互不重复的端口，再重试降级。"
        )
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.create_unique_constraint(op.f("uq_deployments_port"), ["port"])
