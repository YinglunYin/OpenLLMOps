"""add deployment observability

Revision ID: e93c4b7a15d2
Revises: d82f6a1e093b
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e93c4b7a15d2"
down_revision: str | None = "d82f6a1e093b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 字符串列避免把仅三种运行时信号升级为 PostgreSQL 全局枚举，便于将来兼容新探针。
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.add_column(sa.Column("health_status", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_column("started_at")
        batch_op.drop_column("health_status")
