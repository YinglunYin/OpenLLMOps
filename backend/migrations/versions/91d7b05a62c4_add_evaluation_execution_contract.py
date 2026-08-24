"""add evaluation execution contract

Revision ID: 91d7b05a62c4
Revises: 6b8a2d9c4e11
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "91d7b05a62c4"
down_revision: str | None = "6b8a2d9c4e11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _template_for_legacy_model(kind: str) -> str:
    return "instruct" if kind == "instruct" else "base"


def upgrade() -> None:
    # 先以 nullable 形式增加列，逐行安全回填历史任务后再收紧 NOT NULL。
    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "base_template",
                sa.Enum("base", "instruct", name="evaluationtemplate", native_enum=False),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "candidate_template",
                sa.Enum("base", "instruct", name="evaluationtemplate", native_enum=False),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("output_dir", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("tensor_parallel_size", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("gpu_memory_utilization", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("concurrency", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("max_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("result_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("dataset_manifest_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("warnings", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True))

    runs = sa.table(
        "evaluation_runs",
        sa.column("id", sa.Uuid()),
        sa.column("base_model_asset_id", sa.Uuid()),
        sa.column("candidate_model_asset_id", sa.Uuid()),
        sa.column("gpu_ids", sa.JSON()),
        sa.column("actual_state", sa.String()),
        sa.column("metrics", sa.JSON()),
        sa.column("comparison", sa.JSON()),
        sa.column("error_message", sa.Text()),
        sa.column("finished_at", sa.DateTime(timezone=True)),
        sa.column("base_template", sa.String()),
        sa.column("candidate_template", sa.String()),
        sa.column("output_dir", sa.String()),
        sa.column("tensor_parallel_size", sa.Integer()),
        sa.column("gpu_memory_utilization", sa.Float()),
        sa.column("concurrency", sa.Integer()),
        sa.column("max_tokens", sa.Integer()),
        sa.column("warnings", sa.JSON()),
    )
    assets = sa.table(
        "model_assets",
        sa.column("id", sa.Uuid()),
        sa.column("model_kind", sa.String()),
    )
    base_asset = assets.alias("base_asset")
    candidate_asset = assets.alias("candidate_asset")
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(
            runs.c.id,
            runs.c.gpu_ids,
            runs.c.actual_state,
            runs.c.error_message,
            base_asset.c.model_kind.label("base_kind"),
            candidate_asset.c.model_kind.label("candidate_kind"),
        )
        .join(base_asset, base_asset.c.id == runs.c.base_model_asset_id)
        .join(candidate_asset, candidate_asset.c.id == runs.c.candidate_model_asset_id)
    ).all()
    for row in rows:
        raw_id = str(row.id)
        try:
            normalized_id = str(uuid.UUID(raw_id))
        except ValueError:
            normalized_id = raw_id
        gpu_ids = row.gpu_ids if isinstance(row.gpu_ids, list) else []
        connection.execute(
            runs.update()
            .where(runs.c.id == row.id)
            .values(
                base_template=_template_for_legacy_model(str(row.base_kind)),
                candidate_template=_template_for_legacy_model(str(row.candidate_kind)),
                output_dir=f"/srv/openllmops/evaluation-output/{normalized_id}",
                tensor_parallel_size=max(1, len(gpu_ids)),
                gpu_memory_utilization=0.9,
                concurrency=4,
                max_tokens=32,
                # 旧控制面发送的是 Agent 不接受的 execution shape，不能把未验证 JSON
                # 冒充为新合同结果；清空后由管理员重新创建评测任务。
                metrics={},
                comparison={},
                warnings=[],
                actual_state="failed" if str(row.actual_state) == "succeeded" else row.actual_state,
                error_message=(
                    "升级后旧评测结果不符合严格执行合同，请重新创建评测任务"
                    if str(row.actual_state) == "succeeded"
                    else row.error_message
                ),
                finished_at=sa.func.current_timestamp() if str(row.actual_state) == "succeeded" else None,
            )
        )

    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.alter_column("base_template", existing_type=sa.String(length=8), nullable=False)
        batch_op.alter_column("candidate_template", existing_type=sa.String(length=8), nullable=False)
        batch_op.alter_column("output_dir", existing_type=sa.String(length=1024), nullable=False)
        batch_op.alter_column("tensor_parallel_size", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("gpu_memory_utilization", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("concurrency", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("max_tokens", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("warnings", existing_type=sa.JSON(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("evaluation_runs") as batch_op:
        batch_op.drop_column("finished_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("warnings")
        batch_op.drop_column("dataset_manifest_path")
        batch_op.drop_column("result_path")
        batch_op.drop_column("max_tokens")
        batch_op.drop_column("concurrency")
        batch_op.drop_column("gpu_memory_utilization")
        batch_op.drop_column("tensor_parallel_size")
        batch_op.drop_column("output_dir")
        batch_op.drop_column("candidate_template")
        batch_op.drop_column("base_template")
