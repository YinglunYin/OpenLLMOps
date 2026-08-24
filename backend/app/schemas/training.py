from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import JobState


class TrainingParameters(BaseModel):
    """控制面与 node-agent 共同接受的有限训练参数集合。"""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    template: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}$",
    )
    num_train_epochs: float = Field(default=3.0, gt=0, le=100)
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1)
    cutoff_len: int = Field(default=2048, ge=128, le=65_536)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=4096)
    logging_steps: int = Field(default=10, ge=1, le=100_000)
    save_steps: int = Field(default=100, ge=1, le=1_000_000)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    lora_rank: int = Field(default=16, ge=1, le=1024)
    lora_alpha: int = Field(default=32, ge=1, le=4096)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    freeze_trainable_layers: int = Field(default=2, ge=1, le=256)
    max_samples: int | None = Field(default=None, ge=1, le=10_000_000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


TrainingArtifactKind = Literal["checkpoint", "adapter", "merged", "full"]


class TrainingArtifactRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TrainingArtifactKind
    path: str
    file_count: int = Field(ge=1)
    size_bytes: int = Field(ge=1)
    archive_filename: str


class TrainingArtifactManifestRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: uuid.UUID
    state: JobState
    artifacts: list[TrainingArtifactRead]


class TrainingObservationMetadata(BaseModel):
    """node-agent 训练状态响应的有限、严格 metadata。"""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    progress: float | None = Field(default=None, ge=0, le=100)
    current_step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=1)
    metrics: dict[str, str | int | float | bool] | None = None
    checkpoint_path: str | None = Field(default=None, min_length=1, max_length=1024)
    adapter_path: str | None = Field(default=None, min_length=1, max_length=1024)
    merged_model_path: str | None = Field(default=None, min_length=1, max_length=1024)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls,
        value: dict[str, str | int | float | bool] | None,
    ) -> dict[str, str | int | float | bool] | None:
        if value is None:
            return None
        if len(value) > 128:
            raise ValueError("训练 metrics 字段数量超过 128")
        if any(not key or len(key) > 128 for key in value):
            raise ValueError("训练 metrics 键名为空或过长")
        return value

    @model_validator(mode="after")
    def validate_progress(self) -> TrainingObservationMetadata:
        if (
            self.current_step is not None
            and self.total_steps is not None
            and self.current_step > self.total_steps
        ):
            raise ValueError("训练 current_step 不能超过 total_steps")
        return self
