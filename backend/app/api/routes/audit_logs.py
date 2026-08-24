from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models import AuditLog
from app.schemas import AuditLogRead

router = APIRouter(prefix="/audit-logs", tags=["审计日志"])


@router.get("", response_model=list[AuditLogRead], name="audit.list")
async def list_audit_logs(
    request_id: str | None = Query(default=None, max_length=64),
    action: str | None = Query(default=None, max_length=128),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
) -> list[AuditLog]:
    statement = select(AuditLog)
    if request_id:
        statement = statement.where(AuditLog.request_id == request_id)
    if action:
        statement = statement.where(AuditLog.action == action)
    result = await session.scalars(
        statement.order_by(AuditLog.occurred_at.desc()).offset(offset).limit(limit)
    )
    return list(result)
