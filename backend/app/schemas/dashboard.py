import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.enums import LeaseOwnerType


class DashboardModelSummary(BaseModel):
    total: int = Field(ge=0)
    ready: int = Field(ge=0)
    importing: int = Field(ge=0)
    failed: int = Field(ge=0)


class DashboardWorkloadSummary(BaseModel):
    total: int = Field(ge=0)
    running: int = Field(ge=0)
    queued: int = Field(ge=0)
    failed: int = Field(ge=0)


class DashboardQueueSummary(BaseModel):
    total: int = Field(ge=0)
    deployments: int = Field(ge=0)
    training_jobs: int = Field(ge=0)
    evaluation_runs: int = Field(ge=0)
    model_imports: int = Field(ge=0)


class DashboardLeaseRead(BaseModel):
    gpu_index: int = Field(ge=0)
    owner_type: LeaseOwnerType
    owner_id: uuid.UUID
    owner_name: str
    expires_at: datetime


class DashboardGPUSummary(BaseModel):
    total: int = Field(ge=0)
    leased: int = Field(ge=0)
    free: int = Field(ge=0)
    leases: list[DashboardLeaseRead]


class RecentActivityRead(BaseModel):
    resource_type: Literal[
        "model_asset",
        "model_import",
        "deployment",
        "training_job",
        "evaluation_run",
    ]
    resource_id: uuid.UUID
    name: str
    status: str
    occurred_at: datetime


class DashboardSummaryRead(BaseModel):
    generated_at: datetime
    models: DashboardModelSummary
    deployments: DashboardWorkloadSummary
    training_jobs: DashboardWorkloadSummary
    evaluation_runs: DashboardWorkloadSummary
    queue: DashboardQueueSummary
    gpus: DashboardGPUSummary
    recent_activity: list[RecentActivityRead]
