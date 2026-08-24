from datetime import UTC, datetime

from sqlalchemy import String, cast, func, literal, select, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Deployment,
    EvaluationRun,
    GPULease,
    ModelAsset,
    ModelImportJob,
    TrainingJob,
)
from app.models.enums import AssetStatus, DeploymentState, JobState, ModelImportStatus
from app.schemas.dashboard import (
    DashboardGPUSummary,
    DashboardLeaseRead,
    DashboardModelSummary,
    DashboardQueueSummary,
    DashboardSummaryRead,
    DashboardWorkloadSummary,
    RecentActivityRead,
)

RECENT_ACTIVITY_LIMIT = 12


def _count(model, *criteria):  # type: ignore[no-untyped-def]
    statement = select(func.count()).select_from(model)
    if criteria:
        statement = statement.where(*criteria)
    return statement.scalar_subquery()


def _activity_select(  # type: ignore[no-untyped-def]
    resource_type: str,
    model,
    status_column,
    *criteria,
):
    statement = select(
        literal(resource_type).label("resource_type"),
        cast(model.id, String).label("resource_id"),
        model.name.label("name"),
        cast(status_column, String).label("status"),
        model.updated_at.label("occurred_at"),
    )
    if criteria:
        statement = statement.where(*criteria)
    return statement


async def build_dashboard_summary(
    session: AsyncSession,
    *,
    gpu_count: int,
) -> DashboardSummaryRead:
    """三个数据库往返完成全部摘要，避免逐资源列表后在 Python 中计数。"""

    counts_statement = select(
        _count(ModelAsset, ModelAsset.deleted_at.is_(None)).label("models_total"),
        _count(
            ModelAsset,
            ModelAsset.deleted_at.is_(None),
            ModelAsset.status == AssetStatus.READY,
        ).label("models_ready"),
        _count(
            ModelAsset,
            ModelAsset.deleted_at.is_(None),
            ModelAsset.status == AssetStatus.IMPORTING,
        ).label("models_importing"),
        _count(
            ModelAsset,
            ModelAsset.deleted_at.is_(None),
            ModelAsset.status == AssetStatus.FAILED,
        ).label("models_failed"),
        _count(Deployment).label("deployments_total"),
        _count(Deployment, Deployment.actual_state == DeploymentState.RUNNING).label("deployments_running"),
        _count(Deployment, Deployment.actual_state == DeploymentState.QUEUED).label("deployments_queued"),
        _count(Deployment, Deployment.actual_state == DeploymentState.FAILED).label("deployments_failed"),
        _count(TrainingJob).label("training_total"),
        _count(TrainingJob, TrainingJob.actual_state == JobState.RUNNING).label("training_running"),
        _count(TrainingJob, TrainingJob.actual_state == JobState.QUEUED).label("training_queued"),
        _count(TrainingJob, TrainingJob.actual_state == JobState.FAILED).label("training_failed"),
        _count(EvaluationRun).label("evaluations_total"),
        _count(EvaluationRun, EvaluationRun.actual_state == JobState.RUNNING).label("evaluations_running"),
        _count(EvaluationRun, EvaluationRun.actual_state == JobState.QUEUED).label("evaluations_queued"),
        _count(EvaluationRun, EvaluationRun.actual_state == JobState.FAILED).label("evaluations_failed"),
        _count(ModelImportJob, ModelImportJob.status == ModelImportStatus.PENDING).label("imports_queued"),
    )
    counts = (await session.execute(counts_statement)).mappings().one()

    lease_rows = list(await session.scalars(select(GPULease).order_by(GPULease.gpu_index)))

    activities = union_all(
        _activity_select(
            "model_asset",
            ModelAsset,
            ModelAsset.status,
            ModelAsset.deleted_at.is_(None),
        ),
        _activity_select("model_import", ModelImportJob, ModelImportJob.status),
        _activity_select("deployment", Deployment, Deployment.actual_state),
        _activity_select("training_job", TrainingJob, TrainingJob.actual_state),
        _activity_select("evaluation_run", EvaluationRun, EvaluationRun.actual_state),
    ).subquery()
    activity_rows = (
        await session.execute(
            select(activities)
            .order_by(
                activities.c.occurred_at.desc(),
                activities.c.resource_type,
                activities.c.resource_id,
            )
            .limit(RECENT_ACTIVITY_LIMIT)
        )
    ).mappings()

    deployment_queue = int(counts["deployments_queued"])
    training_queue = int(counts["training_queued"])
    evaluation_queue = int(counts["evaluations_queued"])
    import_queue = int(counts["imports_queued"])
    leased = len(lease_rows)
    return DashboardSummaryRead(
        generated_at=datetime.now(UTC),
        models=DashboardModelSummary(
            total=counts["models_total"],
            ready=counts["models_ready"],
            importing=counts["models_importing"],
            failed=counts["models_failed"],
        ),
        deployments=DashboardWorkloadSummary(
            total=counts["deployments_total"],
            running=counts["deployments_running"],
            queued=deployment_queue,
            failed=counts["deployments_failed"],
        ),
        training_jobs=DashboardWorkloadSummary(
            total=counts["training_total"],
            running=counts["training_running"],
            queued=training_queue,
            failed=counts["training_failed"],
        ),
        evaluation_runs=DashboardWorkloadSummary(
            total=counts["evaluations_total"],
            running=counts["evaluations_running"],
            queued=evaluation_queue,
            failed=counts["evaluations_failed"],
        ),
        queue=DashboardQueueSummary(
            total=deployment_queue + training_queue + evaluation_queue + import_queue,
            deployments=deployment_queue,
            training_jobs=training_queue,
            evaluation_runs=evaluation_queue,
            model_imports=import_queue,
        ),
        gpus=DashboardGPUSummary(
            total=gpu_count,
            leased=leased,
            free=max(0, gpu_count - leased),
            leases=[
                DashboardLeaseRead(
                    gpu_index=lease.gpu_index,
                    owner_type=lease.owner_type,
                    owner_id=lease.owner_id,
                    owner_name=lease.owner_name,
                    expires_at=lease.expires_at,
                )
                for lease in lease_rows
            ],
        ),
        recent_activity=[RecentActivityRead.model_validate(row) for row in activity_rows],
    )
