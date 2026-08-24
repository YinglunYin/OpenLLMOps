"""soft delete model assets

Revision ID: d82f6a1e093b
Revises: b75e9d4c1a20
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d82f6a1e093b"
down_revision: str | None = "b75e9d4c1a20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("model_assets") as batch_op:
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index(op.f("ix_model_assets_deleted_at"), ["deleted_at"], unique=False)


def downgrade() -> None:
    tombstone_count = op.get_bind().scalar(
        sa.text("SELECT COUNT(*) FROM model_assets WHERE deleted_at IS NOT NULL")
    )
    if tombstone_count:
        raise RuntimeError(
            "检测到已软删除的模型资产，拒绝移除 deleted_at 列以免旧版本重新暴露；"
            "请恢复到升级前备份，或先制定显式 tombstone 迁移方案"
        )
    with op.batch_alter_table("model_assets") as batch_op:
        batch_op.drop_index(op.f("ix_model_assets_deleted_at"))
        batch_op.drop_column("deleted_at")
