"""add model import jobs

Revision ID: 6b8a2d9c4e11
Revises: 2863c3b78f7c
Create Date: 2026-08-25 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6b8a2d9c4e11"
down_revision: str | None = "2863c3b78f7c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_import_jobs",
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column(
            "source",
            sa.Enum(
                "huggingface",
                "modelscope",
                "controlled_directory",
                name="modelimportsource",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("repository", sa.String(length=512), nullable=True),
        sa.Column("revision", sa.String(length=255), nullable=True),
        sa.Column("source_directory", sa.String(length=255), nullable=True),
        sa.Column(
            "model_kind",
            sa.Enum("base", "instruct", "embedding", name="modelkind", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "transferring",
                "validating",
                "ready",
                "failed",
                "canceling",
                "canceled",
                name="modelimportstatus",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("progress_completed", sa.BigInteger(), nullable=False),
        sa.Column("progress_total", sa.BigInteger(), nullable=True),
        sa.Column("claimed_by", sa.String(length=128), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_asset_id", sa.Uuid(), nullable=True),
        sa.Column("manifest_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["result_asset_id"],
            ["model_assets.id"],
            name=op.f("fk_model_import_jobs_result_asset_id_model_assets"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_import_jobs")),
    )
    op.create_index(
        op.f("ix_model_import_jobs_claimed_by"),
        "model_import_jobs",
        ["claimed_by"],
        unique=False,
    )
    op.create_index(op.f("ix_model_import_jobs_name"), "model_import_jobs", ["name"], unique=False)
    op.create_index(
        op.f("ix_model_import_jobs_result_asset_id"),
        "model_import_jobs",
        ["result_asset_id"],
        unique=True,
    )
    op.create_index(op.f("ix_model_import_jobs_source"), "model_import_jobs", ["source"], unique=False)
    op.create_index(op.f("ix_model_import_jobs_status"), "model_import_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_model_import_jobs_status"), table_name="model_import_jobs")
    op.drop_index(op.f("ix_model_import_jobs_source"), table_name="model_import_jobs")
    op.drop_index(op.f("ix_model_import_jobs_result_asset_id"), table_name="model_import_jobs")
    op.drop_index(op.f("ix_model_import_jobs_name"), table_name="model_import_jobs")
    op.drop_index(op.f("ix_model_import_jobs_claimed_by"), table_name="model_import_jobs")
    op.drop_table("model_import_jobs")
