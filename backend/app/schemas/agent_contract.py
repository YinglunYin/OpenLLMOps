import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import LeaseOwnerType


class AgentAction(StrEnum):
    """控制面允许 node-agent 执行的幂等动作。"""

    START = "start"
    STOP = "stop"
    STATUS = "status"


class AgentWorkloadState(StrEnum):
    """node-agent 观察到的运行态；不存在与业务失败必须明确区分。"""

    ABSENT = "absent"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentOwner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: LeaseOwnerType
    id: uuid.UUID
    name: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=1)


class AgentResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_ids: list[int] = Field(min_length=1)


class AgentCommand(BaseModel):
    """控制面发给 node-agent 的稳定 HTTP 合同。

    request_id 用于传输层去重，owner.generation 用于拒绝旧状态命令。两者配合可让
    reconciler 在超时重试时保持幂等，而不会误停用户刚刚重启的新一代工作负载。
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1"] = "1"
    request_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    action: AgentAction
    owner: AgentOwner
    resources: AgentResourceRequest
    execution: dict[str, Any] = Field(default_factory=dict)


class AgentCommandResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1"] = "1"
    request_id: uuid.UUID
    accepted: bool
    observed_state: AgentWorkloadState
    observed_at: datetime
    message: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
