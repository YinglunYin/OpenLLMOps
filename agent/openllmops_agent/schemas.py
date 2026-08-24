from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonScalar = str | int | float | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InferenceLaunchRequest(StrictModel):
    deployment_id: UUID
    generation: int = Field(default=1, ge=1)
    image: str = Field(min_length=1, max_length=255)
    gpu_ids: list[int] = Field(min_length=1, max_length=16)
    model_path: Path
    served_model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
    service_type: Literal["generate", "embedding"] = "generate"
    port: int = Field(default=8000, ge=1024, le=65535)
    vllm_args: dict[str, JsonScalar] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("gpu_ids")
    @classmethod
    def unique_gpu_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("gpu_ids 不能重复")
        return value


class TrainingLaunchRequest(StrictModel):
    job_id: UUID
    generation: int = Field(default=1, ge=1)
    image: str = Field(min_length=1, max_length=255)
    gpu_ids: list[int] = Field(min_length=1, max_length=16)
    model_path: Path
    dataset_path: Path
    dataset_dir: Path | None = None
    config_path: Path
    output_path: Path
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("gpu_ids")
    @classmethod
    def unique_gpu_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("gpu_ids 不能重复")
        return value


class EvaluationLaunchRequest(StrictModel):
    run_id: UUID
    generation: int = Field(default=1, ge=1)
    image: str = Field(min_length=1, max_length=255)
    gpu_ids: list[int] = Field(min_length=1, max_length=16)
    baseline_model_path: Path
    candidate_model_path: Path
    dataset_path: Path
    dataset_manifest_path: Path
    output_path: Path
    base_template: Literal["base", "instruct"]
    candidate_template: Literal["base", "instruct"]
    tensor_parallel_size: int = Field(ge=1, le=16)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=0.95)
    concurrency: int = Field(default=4, ge=1, le=32)
    max_tokens: int = Field(default=32, ge=1, le=512)

    @field_validator("gpu_ids")
    @classmethod
    def unique_evaluation_gpu_ids(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("gpu_ids 不能重复")
        return value


class StopRequest(StrictModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class WorkloadInfo(StrictModel):
    name: str
    workload_id: UUID
    kind: Literal["inference", "training", "evaluation"]
    image: str
    status: str
    gpu_ids: list[int]
    service_type: Literal["generate", "embedding"] | None = None
    endpoint: str | None = None
    port: int | None = None
    generation: int = Field(default=1, ge=1)
    exit_code: int | None = None
    created_at: datetime | None = None


class GPUInfo(StrictModel):
    index: int
    uuid: str | None = None
    name: str | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    memory_free_mib: int | None = None
    utilization_percent: int | None = None
    temperature_celsius: int | None = None
    power_watts: float | None = None
    allocated_to: str | None = None


class GPUInventory(StrictModel):
    driver_available: bool
    error: str | None = None
    gpus: list[GPUInfo]


class HealthResponse(StrictModel):
    status: Literal["ok"]
    docker_connected: bool
    runtime_network: str
