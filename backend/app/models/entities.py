import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_type
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DeploymentState,
    DeploymentTaskType,
    DesiredJobState,
    DesiredServiceState,
    JobState,
    LeaseOwnerType,
    ModelImportSource,
    ModelImportStatus,
    ModelKind,
    ModelSourceType,
    TrainingAlgorithm,
    TrainingStage,
)


class ModelAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_assets"

    name: Mapped[str] = mapped_column(String(128), index=True)
    source_type: Mapped[ModelSourceType] = mapped_column(enum_type(ModelSourceType))
    source_uri: Mapped[str | None] = mapped_column(String(1024))
    revision: Mapped[str | None] = mapped_column(String(255))
    local_path: Mapped[str] = mapped_column(String(1024), unique=True)
    model_kind: Mapped[ModelKind] = mapped_column(enum_type(ModelKind))
    format: Mapped[str] = mapped_column(String(32), default="safetensors")
    status: Mapped[AssetStatus] = mapped_column(
        enum_type(AssetStatus), default=AssetStatus.IMPORTING, index=True
    )
    family: Mapped[str | None] = mapped_column(String(128))
    parameter_count: Mapped[int | None] = mapped_column(BigInteger)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelImportJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_import_jobs"

    name: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[ModelImportSource] = mapped_column(enum_type(ModelImportSource), index=True)
    repository: Mapped[str | None] = mapped_column(String(512))
    revision: Mapped[str | None] = mapped_column(String(255))
    # 受控目录只持久化 inbox 下的一层相对名称，不保存任意绝对来源路径。
    source_directory: Mapped[str | None] = mapped_column(String(255))
    model_kind: Mapped[ModelKind] = mapped_column(enum_type(ModelKind))
    status: Mapped[ModelImportStatus] = mapped_column(
        enum_type(ModelImportStatus), default=ModelImportStatus.PENDING, index=True
    )
    progress_completed: Mapped[int] = mapped_column(BigInteger, default=0)
    progress_total: Mapped[int | None] = mapped_column(BigInteger)
    claimed_by: Mapped[str | None] = mapped_column(String(128), index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_assets.id", ondelete="SET NULL"), unique=True, index=True
    )
    manifest_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)


class Dataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(128), index=True)
    dataset_type: Mapped[DatasetType] = mapped_column(enum_type(DatasetType), index=True)
    status: Mapped[DatasetStatus] = mapped_column(
        enum_type(DatasetStatus), default=DatasetStatus.VALIDATING, index=True
    )
    file_name: Mapped[str] = mapped_column(String(255))
    local_path: Mapped[str] = mapped_column(String(1024), unique=True)
    record_count: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    schema_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    validation_errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    description: Mapped[str | None] = mapped_column(Text)


class Deployment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "deployments"
    __table_args__ = (
        UniqueConstraint("name", name="uq_deployments_name"),
        UniqueConstraint("served_model_name", name="uq_deployments_served_model_name"),
    )

    name: Mapped[str] = mapped_column(String(128))
    served_model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_assets.id", ondelete="RESTRICT"), index=True
    )
    task_type: Mapped[DeploymentTaskType] = mapped_column(enum_type(DeploymentTaskType))
    desired_state: Mapped[DesiredServiceState] = mapped_column(
        enum_type(DesiredServiceState), default=DesiredServiceState.STOPPED
    )
    actual_state: Mapped[DeploymentState] = mapped_column(
        enum_type(DeploymentState), default=DeploymentState.CREATED, index=True
    )
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    tensor_parallel_size: Mapped[int] = mapped_column(Integer, default=1)
    port: Mapped[int | None] = mapped_column(Integer, unique=True)
    internal_url: Mapped[str | None] = mapped_column(String(512))
    simplified_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    vllm_args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    runtime_generation: Mapped[int] = mapped_column(Integer, default=0)


class TrainingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "training_jobs"
    __table_args__ = (UniqueConstraint("name", name="uq_training_jobs_name"),)

    name: Mapped[str] = mapped_column(String(128))
    model_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_assets.id", ondelete="RESTRICT"), index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="RESTRICT"), index=True)
    stage: Mapped[TrainingStage] = mapped_column(enum_type(TrainingStage))
    algorithm: Mapped[TrainingAlgorithm] = mapped_column(enum_type(TrainingAlgorithm))
    desired_state: Mapped[DesiredJobState] = mapped_column(
        enum_type(DesiredJobState), default=DesiredJobState.RUNNING
    )
    actual_state: Mapped[JobState] = mapped_column(enum_type(JobState), default=JobState.CREATED, index=True)
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[int | None] = mapped_column(Integer)
    total_steps: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_dir: Mapped[str] = mapped_column(String(1024))
    checkpoint_path: Mapped[str | None] = mapped_column(String(1024))
    adapter_path: Mapped[str | None] = mapped_column(String(1024))
    merged_model_path: Mapped[str | None] = mapped_column(String(1024))
    error_message: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    runtime_generation: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (UniqueConstraint("name", name="uq_evaluation_runs_name"),)

    name: Mapped[str] = mapped_column(String(128))
    base_model_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_assets.id", ondelete="RESTRICT"), index=True
    )
    candidate_model_asset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_assets.id", ondelete="RESTRICT"), index=True
    )
    custom_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), index=True
    )
    builtin_datasets: Mapped[list[str]] = mapped_column(JSON, default=list)
    desired_state: Mapped[DesiredJobState] = mapped_column(
        enum_type(DesiredJobState), default=DesiredJobState.RUNNING
    )
    actual_state: Mapped[JobState] = mapped_column(enum_type(JobState), default=JobState.CREATED, index=True)
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    comparison: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    state_version: Mapped[int] = mapped_column(Integer, default=1)
    runtime_generation: Mapped[int] = mapped_column(Integer, default=0)


class GPULease(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "gpu_leases"
    __table_args__ = (
        # 单卡唯一约束是最终安全网：即使错误代码绕过调度器，数据库也不允许整卡被复用。
        UniqueConstraint("gpu_index", name="uq_gpu_leases_gpu_index"),
        UniqueConstraint(
            "owner_type",
            "owner_id",
            "gpu_index",
            name="uq_gpu_leases_owner_gpu",
        ),
    )

    gpu_index: Mapped[int] = mapped_column(Integer, index=True)
    lease_group_id: Mapped[uuid.UUID] = mapped_column(index=True)
    owner_type: Mapped[LeaseOwnerType] = mapped_column(enum_type(LeaseOwnerType), index=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(index=True)
    owner_name: Mapped[str] = mapped_column(String(128))
    generation: Mapped[int] = mapped_column(Integer, default=1)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class APIKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(128), unique=True)
    prefix: Mapped[str] = mapped_column(String(24), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """只保存安全审计所需元数据，禁止存储请求体、密码、Cookie 或 API Key。"""

    __tablename__ = "audit_logs"

    request_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(128), index=True)
    auth_method: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(128), index=True)
    method: Mapped[str] = mapped_column(String(16))
    path: Mapped[str] = mapped_column(String(1024))
    status_code: Mapped[int] = mapped_column(Integer, index=True)
    succeeded: Mapped[bool] = mapped_column(Boolean, index=True)
    source_ip: Mapped[str] = mapped_column(String(64), index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
