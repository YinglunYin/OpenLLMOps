import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DeploymentState,
    DeploymentTaskType,
    DesiredJobState,
    DesiredServiceState,
    EvaluationTemplate,
    JobState,
    LeaseOwnerType,
    ModelKind,
    ModelSourceType,
    TrainingAlgorithm,
    TrainingStage,
)
from app.schemas.common import TimestampedModel
from app.schemas.evaluation import (
    EmptyEvaluationResult,
    EvaluationComparison,
    EvaluationMetrics,
    EvaluationWarning,
)
from app.schemas.training import TrainingParameters
from app.schemas.vllm import validate_vllm_arguments


class ModelAssetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    source_type: ModelSourceType
    source_uri: str | None = Field(default=None, max_length=1024)
    revision: str | None = Field(default=None, max_length=255)
    local_path: str = Field(min_length=1, max_length=1024)
    model_kind: ModelKind
    format: Literal["safetensors"] = "safetensors"
    status: AssetStatus = AssetStatus.IMPORTING
    family: str | None = Field(default=None, max_length=128)
    parameter_count: int | None = Field(default=None, ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    checksum: str | None = Field(default=None, max_length=128)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ModelAssetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)


class ModelAssetRead(TimestampedModel):
    name: str
    source_type: ModelSourceType
    source_uri: str | None
    revision: str | None
    local_path: str
    model_kind: ModelKind
    format: str
    status: AssetStatus
    family: str | None
    parameter_count: int | None
    size_bytes: int | None
    checksum: str | None
    error_message: str | None
    metadata_json: dict[str, Any]


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    dataset_type: DatasetType
    status: DatasetStatus = DatasetStatus.VALIDATING
    file_name: str = Field(min_length=1, max_length=255)
    local_path: str = Field(min_length=1, max_length=1024)
    record_count: int | None = Field(default=None, ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    schema_summary: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    description: str | None = None

    @field_validator("file_name", "local_path")
    @classmethod
    def require_jsonl(cls, value: str) -> str:
        if not value.lower().endswith(".jsonl"):
            raise ValueError("数据集必须为 jsonl 格式")
        return value


class DatasetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None


class DatasetRead(TimestampedModel):
    name: str
    dataset_type: DatasetType
    status: DatasetStatus
    file_name: str
    local_path: str
    record_count: int | None
    size_bytes: int | None
    sha256: str | None
    schema_summary: dict[str, Any]
    validation_errors: list[dict[str, Any]]
    description: str | None


SYSTEM_MANAGED_VLLM_ARGS = {
    "api_key",
    "host",
    "model",
    "pipeline_parallel_size",
    "port",
    "served_model_name",
    "tensor_parallel_size",
    "trust_remote_code",
}


class DeploymentSimplifiedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    max_model_len: int | None = Field(default=None, ge=1024, le=131_072)
    gpu_memory_utilization: float | None = Field(default=None, ge=0.1, le=0.98)
    dtype: Literal["auto", "float16", "bfloat16"] | None = None


class DeploymentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    served_model_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$",
    )
    model_asset_id: uuid.UUID
    task_type: DeploymentTaskType
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1)
    tensor_parallel_size: int = Field(default=1, ge=1, le=32)
    simplified_config: DeploymentSimplifiedConfig = Field(default_factory=DeploymentSimplifiedConfig)
    vllm_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gpu_and_vllm_config(self) -> "DeploymentCreate":
        if len(self.gpu_ids) != len(set(self.gpu_ids)) or min(self.gpu_ids) < 0:
            raise ValueError("GPU 编号必须是互不重复的非负整数")
        if self.tensor_parallel_size != len(self.gpu_ids):
            raise ValueError("tensor_parallel_size 必须等于所选 GPU 数量")
        normalized_keys = {key.lstrip("-").replace("-", "_") for key in self.vllm_args}
        blocked = normalized_keys & SYSTEM_MANAGED_VLLM_ARGS
        if blocked:
            raise ValueError(f"以下 vLLM 参数由系统管理，不能覆盖：{', '.join(sorted(blocked))}")
        validate_vllm_arguments(self.vllm_args)
        return self


class DeploymentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    gpu_ids: list[int] | None = Field(default=None, min_length=1)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=32)
    simplified_config: DeploymentSimplifiedConfig | None = None
    vllm_args: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_detail_args(self) -> "DeploymentUpdate":
        if self.gpu_ids is not None and (
            len(self.gpu_ids) != len(set(self.gpu_ids)) or min(self.gpu_ids) < 0
        ):
            raise ValueError("GPU 编号必须是互不重复的非负整数")
        if (
            self.gpu_ids is not None
            and self.tensor_parallel_size is not None
            and len(self.gpu_ids) != self.tensor_parallel_size
        ):
            raise ValueError("tensor_parallel_size 必须等于所选 GPU 数量")
        if self.vllm_args is not None:
            normalized = {key.lstrip("-").replace("-", "_") for key in self.vllm_args}
            blocked = normalized & SYSTEM_MANAGED_VLLM_ARGS
            if blocked:
                raise ValueError(f"以下 vLLM 参数由系统管理，不能覆盖：{', '.join(sorted(blocked))}")
            validate_vllm_arguments(self.vllm_args)
        return self


class DeploymentRead(TimestampedModel):
    name: str
    served_model_name: str
    model_asset_id: uuid.UUID
    task_type: DeploymentTaskType
    desired_state: DesiredServiceState
    actual_state: DeploymentState
    gpu_ids: list[int]
    tensor_parallel_size: int
    simplified_config: dict[str, Any]
    vllm_args: dict[str, Any]
    health_status: Literal["starting", "healthy", "unhealthy"] | None
    started_at: datetime | None
    error_message: str | None
    queued_at: datetime | None
    state_version: int
    runtime_generation: int


class TrainingJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model_asset_id: uuid.UUID
    dataset_id: uuid.UUID
    stage: TrainingStage
    algorithm: TrainingAlgorithm
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    training_config: TrainingParameters = Field(default_factory=TrainingParameters)

    @model_validator(mode="after")
    def validate_training_mode(self) -> "TrainingJobCreate":
        if self.stage == TrainingStage.CPT and self.algorithm != TrainingAlgorithm.LORA:
            raise ValueError("继续预训练（CPT）首版仅支持 LoRA")
        if self.stage == TrainingStage.SFT and self.training_config.template is None:
            raise ValueError("SFT 训练必须指定受支持的 template")
        if len(self.gpu_ids) != len(set(self.gpu_ids)) or min(self.gpu_ids) < 0:
            raise ValueError("GPU 编号必须是互不重复的非负整数")
        return self


class TrainingJobRead(TimestampedModel):
    name: str
    model_asset_id: uuid.UUID
    dataset_id: uuid.UUID
    stage: TrainingStage
    algorithm: TrainingAlgorithm
    desired_state: DesiredJobState
    actual_state: JobState
    gpu_ids: list[int]
    progress: float
    current_step: int | None
    total_steps: int | None
    metrics: dict[str, Any]
    training_config: TrainingParameters
    output_dir: str
    checkpoint_path: str | None
    adapter_path: str | None
    merged_model_path: str | None
    published_model_asset_id: uuid.UUID | None
    error_message: str | None
    queued_at: datetime | None
    state_version: int
    runtime_generation: int
    started_at: datetime | None
    finished_at: datetime | None


class EvaluationRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    base_model_asset_id: uuid.UUID
    candidate_model_asset_id: uuid.UUID
    custom_dataset_id: uuid.UUID | None = None
    builtin_datasets: list[Literal["ceval", "cmmlu"]] = Field(
        default_factory=list,
        max_length=2,
    )
    gpu_ids: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_evaluation(self) -> "EvaluationRunCreate":
        if not self.builtin_datasets and self.custom_dataset_id is None:
            raise ValueError("至少选择 C-Eval、CMMLU 或一个自定义评测数据集")
        if len(self.builtin_datasets) != len(set(self.builtin_datasets)):
            raise ValueError("内置评测数据集不能重复")
        if len(self.gpu_ids) != len(set(self.gpu_ids)) or min(self.gpu_ids) < 0:
            raise ValueError("GPU 编号必须是互不重复的非负整数")
        return self


class EvaluationRunRead(TimestampedModel):
    name: str
    base_model_asset_id: uuid.UUID
    candidate_model_asset_id: uuid.UUID
    custom_dataset_id: uuid.UUID | None
    builtin_datasets: list[str]
    base_template: EvaluationTemplate
    candidate_template: EvaluationTemplate
    output_dir: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    concurrency: int
    max_tokens: int
    desired_state: DesiredJobState
    actual_state: JobState
    gpu_ids: list[int]
    metrics: EvaluationMetrics | EmptyEvaluationResult
    comparison: EvaluationComparison | EmptyEvaluationResult
    result_path: str | None
    dataset_manifest_path: str | None
    warnings: list[EvaluationWarning]
    error_message: str | None
    queued_at: datetime | None
    state_version: int
    runtime_generation: int
    started_at: datetime | None
    finished_at: datetime | None


class GPULeaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: uuid.UUID
    gpu_index: int
    lease_group_id: uuid.UUID
    owner_type: LeaseOwnerType
    owner_id: uuid.UUID
    owner_name: str
    generation: int
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime


class APIKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class APIKeyRead(TimestampedModel):
    name: str
    prefix: str
    is_active: bool
    last_used_at: datetime | None


class APIKeyCreated(APIKeyRead):
    # 明文只在创建响应出现一次，数据库永远只保存摘要。
    key: str


class OpenAIProxyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
