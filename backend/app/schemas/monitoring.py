import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.enums import LeaseOwnerType


class GPUHistoryMetric(StrEnum):
    """前端可查询的固定指标名；枚举值不会直接作为 PromQL。"""

    UTILIZATION = "utilization"
    MEMORY_USED_MIB = "memory_used_mib"
    MEMORY_FREE_MIB = "memory_free_mib"
    TEMPERATURE_CELSIUS = "temperature_celsius"
    POWER_WATTS = "power_watts"


class GPUStatusRead(BaseModel):
    index: int = Field(ge=0)
    name: str | None
    memory_total_mib: float | None = Field(default=None, ge=0)
    memory_used_mib: float | None = Field(default=None, ge=0)
    memory_free_mib: float | None = Field(default=None, ge=0)
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    temperature_celsius: float | None
    power_watts: float | None = Field(default=None, ge=0)
    telemetry_available: bool
    degraded_reason: str | None
    owner_type: LeaseOwnerType | None = None
    owner_id: uuid.UUID | None = None
    owner_name: str | None = None
    lease_expires_at: datetime | None = None


class GPUHistoryPoint(BaseModel):
    timestamp: datetime
    value: float


class GPUHistoryRead(BaseModel):
    gpu_index: int = Field(ge=0)
    metric: GPUHistoryMetric
    unit: str
    start: datetime
    end: datetime
    step_seconds: int
    telemetry_available: bool
    degraded_reason: str | None
    points: list[GPUHistoryPoint]
