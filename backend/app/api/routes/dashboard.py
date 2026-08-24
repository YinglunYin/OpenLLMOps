from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.schemas import DashboardSummaryRead
from app.services.dashboard import build_dashboard_summary

router = APIRouter(prefix="/dashboard", tags=["仪表盘"])


@router.get("/summary", response_model=DashboardSummaryRead, name="dashboard.summary")
async def get_dashboard_summary(
    session: AsyncSession = Depends(get_db),
) -> DashboardSummaryRead:
    return await build_dashboard_summary(session, gpu_count=get_settings().gpu_count)
