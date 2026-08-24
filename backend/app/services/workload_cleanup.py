import uuid
from typing import Protocol

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import GPULease
from app.models.enums import LeaseOwnerType
from app.schemas.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
)
from app.services.node_agent import NodeAgentError

CLEANUP_EXECUTION = {"cleanup_terminal": True}


class CleanupAgentGateway(Protocol):
    async def execute(self, command: AgentCommand) -> AgentCommandResponse: ...


class WorkloadCleanupError(RuntimeError):
    """终态容器尚未被安全确认为 absent。"""


class WorkloadCleanupUnavailable(WorkloadCleanupError):
    """Agent 响应不确定，数据库与租约必须原样保留。"""


class WorkloadCleanupBlocked(WorkloadCleanupError):
    """Agent 明确观察到容器仍存在或拒绝清理。"""


async def confirm_absent_and_release_leases(
    session: AsyncSession,
    agent: CleanupAgentGateway,
    *,
    owner_type: LeaseOwnerType,
    owner_id: uuid.UUID,
    owner_name: str,
    generation: int,
    gpu_ids: list[int],
) -> None:
    """删除数据库记录前，以签名幂等命令确认节点容器已经不存在。

    容器清理成功但 HTTP 响应丢失时也不能猜测成功；保留数据库记录和租约后，管理员
    可安全重试，Agent 对 absent 的 cleanup 会再次返回成功。只有已验签的 ABSENT 才
    允许在同一数据库事务内释放该 owner 的租约。
    """

    if generation < 1:
        raise WorkloadCleanupBlocked("运行代次无效，拒绝清理")
    command = AgentCommand(
        action=AgentAction.STOP,
        owner=AgentOwner(
            type=owner_type,
            id=owner_id,
            name=owner_name,
            generation=generation,
        ),
        resources=AgentResourceRequest(gpu_ids=gpu_ids),
        execution=CLEANUP_EXECUTION,
    )
    try:
        response = await agent.execute(command)
    except NodeAgentError as exc:
        raise WorkloadCleanupUnavailable("node-agent 响应不确定，已保留记录与 GPU 租约") from exc
    if not response.accepted or response.observed_state != AgentWorkloadState.ABSENT:
        detail = response.message or response.error_code or response.observed_state.value
        raise WorkloadCleanupBlocked(f"node-agent 尚未确认容器 absent：{detail}")

    # 同一 owner 的容器名不含 generation；ABSENT 已证明没有任何代次仍占卡。因此清理
    # 全部残留租约比只删当前代更安全，避免历史异常租约在记录删除后永久占用 GPU。
    await session.execute(
        delete(GPULease).where(
            GPULease.owner_type == owner_type,
            GPULease.owner_id == owner_id,
        )
    )
