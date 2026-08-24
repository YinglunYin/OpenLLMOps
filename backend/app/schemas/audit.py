import uuid
from datetime import datetime

from app.schemas.common import ORMModel


class AuditLogRead(ORMModel):
    id: uuid.UUID
    request_id: str
    actor: str
    auth_method: str | None
    action: str
    method: str
    path: str
    status_code: int
    succeeded: bool
    source_ip: str
    occurred_at: datetime
