"""secure training contract and publication

Revision ID: c4f3a81d2e70
Revises: 91d7b05a62c4
Create Date: 2026-08-25 00:00:00.000000
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "c4f3a81d2e70"
down_revision: str | None = "91d7b05a62c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TEMPLATE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$")
DEFAULTS: dict[str, Any] = {
    "num_train_epochs": 3.0,
    "learning_rate": 2e-4,
    "cutoff_len": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps": 10,
    "save_steps": 100,
    "warmup_ratio": 0.03,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "freeze_trainable_layers": 2,
    "seed": 42,
}


def _number(
    raw: dict[str, Any], key: str, minimum: float, maximum: float, *, open_min: bool = False
) -> float:
    value = raw.get(key)
    valid = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and (value > minimum if open_min else value >= minimum)
        and value <= maximum
    )
    return float(value) if valid else float(DEFAULTS[key])


def _integer(raw: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = raw.get(key)
    if not isinstance(value, bool) and isinstance(value, int) and minimum <= value <= maximum:
        return value
    return int(DEFAULTS[key])


def _normalize_config(value: Any, stage: str) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    normalized: dict[str, Any] = {
        "num_train_epochs": _number(raw, "num_train_epochs", 0, 100, open_min=True),
        "learning_rate": _number(raw, "learning_rate", 1e-7, 1),
        "cutoff_len": _integer(raw, "cutoff_len", 128, 65_536),
        "per_device_train_batch_size": _integer(raw, "per_device_train_batch_size", 1, 128),
        "gradient_accumulation_steps": _integer(raw, "gradient_accumulation_steps", 1, 4096),
        "logging_steps": _integer(raw, "logging_steps", 1, 100_000),
        "save_steps": _integer(raw, "save_steps", 1, 1_000_000),
        "warmup_ratio": _number(raw, "warmup_ratio", 0, 1),
        "lora_rank": _integer(raw, "lora_rank", 1, 1024),
        "lora_alpha": _integer(raw, "lora_alpha", 1, 4096),
        "lora_dropout": (
            float(raw["lora_dropout"])
            if not isinstance(raw.get("lora_dropout"), bool)
            and isinstance(raw.get("lora_dropout"), int | float)
            and 0 <= raw["lora_dropout"] < 1
            else float(DEFAULTS["lora_dropout"])
        ),
        "freeze_trainable_layers": _integer(raw, "freeze_trainable_layers", 1, 256),
        "seed": _integer(raw, "seed", 0, 2_147_483_647),
    }
    max_samples = raw.get("max_samples")
    if (
        max_samples is not None
        and not isinstance(max_samples, bool)
        and isinstance(max_samples, int)
        and 1 <= max_samples <= 10_000_000
    ):
        normalized["max_samples"] = max_samples
    template = raw.get("template")
    if isinstance(template, str) and TEMPLATE.fullmatch(template):
        normalized["template"] = template
    elif stage == "sft":
        # 历史任务会被置为 failed；占位符仅确保 Read schema 能稳定展示迁移记录。
        normalized["template"] = "legacy"
    return normalized


def upgrade() -> None:
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.add_column(sa.Column("published_model_asset_id", sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            op.f("fk_training_jobs_published_model_asset_id_model_assets"),
            "model_assets",
            ["published_model_asset_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        op.f("ix_training_jobs_published_model_asset_id"),
        "training_jobs",
        ["published_model_asset_id"],
        unique=True,
    )

    jobs = sa.table(
        "training_jobs",
        sa.column("id", sa.Uuid()),
        sa.column("stage", sa.String()),
        sa.column("actual_state", sa.String()),
        sa.column("training_config", sa.JSON()),
        sa.column("output_dir", sa.String()),
        sa.column("checkpoint_path", sa.String()),
        sa.column("adapter_path", sa.String()),
        sa.column("merged_model_path", sa.String()),
        sa.column("error_message", sa.Text()),
        sa.column("queued_at", sa.DateTime(timezone=True)),
        sa.column("finished_at", sa.DateTime(timezone=True)),
    )
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(jobs.c.id, jobs.c.stage, jobs.c.actual_state, jobs.c.training_config)
    ).all()
    for row in rows:
        try:
            normalized_id = str(uuid.UUID(str(row.id)))
        except ValueError:
            normalized_id = str(row.id)
        state = str(row.actual_state)
        invalidated = state not in {"failed", "canceled"}
        connection.execute(
            jobs.update()
            .where(jobs.c.id == row.id)
            .values(
                training_config=_normalize_config(row.training_config, str(row.stage)),
                output_dir=f"/srv/openllmops/checkpoints/{normalized_id}",
                checkpoint_path=None,
                adapter_path=None,
                merged_model_path=None,
                actual_state="failed" if invalidated else state,
                queued_at=None if invalidated else jobs.c.queued_at,
                finished_at=sa.func.current_timestamp() if invalidated else jobs.c.finished_at,
                error_message=(
                    "升级后旧训练任务不符合 UUID 输出目录和严格参数合同，请重新创建任务"
                    if invalidated
                    else jobs.c.error_message
                ),
            )
        )


def downgrade() -> None:
    op.drop_index(op.f("ix_training_jobs_published_model_asset_id"), table_name="training_jobs")
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.drop_constraint(
            op.f("fk_training_jobs_published_model_asset_id_model_assets"),
            type_="foreignkey",
        )
        batch_op.drop_column("published_model_asset_id")
