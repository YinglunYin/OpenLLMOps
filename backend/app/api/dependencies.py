from collections.abc import AsyncIterator

from app.core.config import get_settings
from app.services.node_agent import NodeAgentHTTPClient


async def get_cleanup_node_agent() -> AsyncIterator[NodeAgentHTTPClient | None]:
    """为低频删除操作创建独立 Agent 客户端，避免依赖后台协调器是否启用。"""

    settings = get_settings()
    if not settings.node_agent_token:
        # 是否真的需要 Agent 只能在读取并锁定数据库行后判断；这里不抢先返回
        # 503，确保不存在/非终态记录仍保持准确的 404/409 API 语义。
        yield None
        return
    client = NodeAgentHTTPClient(
        settings.node_agent_url,
        settings.node_agent_token,
        max_clock_skew_seconds=settings.node_agent_clock_skew_seconds,
        timeout_seconds=settings.node_agent_timeout_seconds,
    )
    try:
        yield client
    finally:
        await client.aclose()
