import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class Message(ORMModel):
    message: str


class StateActionResponse(ORMModel):
    id: uuid.UUID
    desired_state: str
    actual_state: str
    message: str


class TimestampedModel(ORMModel):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class PaginationParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
