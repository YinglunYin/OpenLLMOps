import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models import Dataset, Deployment, EvaluationRun, GPULease, ModelAsset, ModelImportJob, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    DeploymentState,
    DeploymentTaskType,
    DesiredServiceState,
    EvaluationTemplate,
    JobState,
    LeaseOwnerType,
    ModelImportSource,
    ModelImportStatus,
    ModelKind,
    ModelSourceType,
    TrainingAlgorithm,
    TrainingStage,
)
from app.services.dashboard import build_dashboard_summary
from app.services.gpu_monitoring import (
    DCGM_FB_FREE,
    DCGM_FB_RESERVED,
    DCGM_FB_USED,
    DCGM_GPU_TEMP,
    DCGM_GPU_UTIL,
    DCGM_POWER_USAGE,
    get_gpu_statuses,
)
from app.services.prometheus import (
    PrometheusClient,
    PrometheusRangeSeries,
    PrometheusResponseError,
    PrometheusSample,
    PrometheusUnavailableError,
    get_prometheus_client,
)


def _vector_payload(*items: tuple[str, str, str]) -> dict:
    timestamp = datetime.now(UTC).timestamp()
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {
                        "__name__": metric,
                        "gpu": gpu,
                        "modelName": "NVIDIA RTX 4090 D",
                    },
                    "value": [timestamp, value],
                }
                for metric, gpu, value in items
            ],
        },
    }


async def test_prometheus_client_strictly_parses_vector_and_rejects_nonfinite() -> None:
    def valid_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prometheus/api/v1/query"
        assert parse_qs(request.url.query.decode())["query"] == ["DCGM_FI_DEV_GPU_UTIL"]
        return httpx.Response(200, json=_vector_payload((DCGM_GPU_UTIL, "0", "42.5")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(valid_handler)) as http_client:
        prometheus = PrometheusClient(
            "http://prometheus:9090/prometheus",
            1,
            http_client=http_client,
        )
        series = await prometheus.query(DCGM_GPU_UTIL)
    assert len(series) == 1
    assert series[0].labels["gpu"] == "0"
    assert series[0].sample.value == 42.5

    def invalid_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_vector_payload((DCGM_GPU_UTIL, "0", "NaN")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(invalid_handler)) as http_client:
        prometheus = PrometheusClient("http://prometheus:9090", 1, http_client=http_client)
        with pytest.raises(PrometheusResponseError, match="结构无效"):
            await prometheus.query(DCGM_GPU_UTIL)


async def test_gpu_status_preserves_missing_values_and_merges_lease(
    isolated_session_factory,
) -> None:
    now = datetime.now(UTC)
    async with isolated_session_factory() as session, session.begin():
        lease = GPULease(
            gpu_index=0,
            lease_group_id=uuid.uuid4(),
            owner_type=LeaseOwnerType.TRAINING,
            owner_id=uuid.uuid4(),
            owner_name="domain-sft",
            generation=1,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=30),
        )
        session.add(lease)

    def handler(_request: httpx.Request) -> httpx.Response:
        # 只返回 used，验证其余字段不会被伪造成 0。
        return httpx.Response(200, json=_vector_payload((DCGM_FB_USED, "0", "1234")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        prometheus = PrometheusClient("http://prometheus:9090", 1, http_client=http_client)
        async with isolated_session_factory() as session:
            statuses = await get_gpu_statuses(session, prometheus, gpu_count=2)

    assert statuses[0].telemetry_available is True
    assert statuses[0].memory_used_mib == 1234
    assert statuses[0].memory_total_mib is None
    assert statuses[0].utilization_percent is None
    assert statuses[0].degraded_reason is not None and "指标不完整" in statuses[0].degraded_reason
    assert statuses[0].owner_type == LeaseOwnerType.TRAINING
    assert statuses[0].owner_name == "domain-sft"
    assert statuses[0].resource_state == "leased"
    assert statuses[1].telemetry_available is False
    assert statuses[1].memory_used_mib is None
    assert statuses[1].resource_state == "unknown"
    assert statuses[1].degraded_reason == "未收到 GPU 1 的 DCGM 指标"


async def test_gpu_status_uses_official_dcgm_units_and_derives_total(
    isolated_session_factory,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_vector_payload(
                (DCGM_FB_USED, "0", "1024"),
                (DCGM_FB_FREE, "0", "22500"),
                (DCGM_FB_RESERVED, "0", "52"),
                (DCGM_GPU_UTIL, "0", "75"),
                (DCGM_GPU_TEMP, "0", "61"),
                (DCGM_POWER_USAGE, "0", "312.5"),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        prometheus = PrometheusClient("http://prometheus:9090", 1, http_client=http_client)
        async with isolated_session_factory() as session:
            status = (await get_gpu_statuses(session, prometheus, gpu_count=1))[0]
    assert status.name == "NVIDIA RTX 4090 D"
    assert status.memory_total_mib == 23576
    assert status.memory_used_mib == 1024
    assert status.memory_free_mib == 22500
    assert status.utilization_percent == 75
    assert status.temperature_celsius == 61
    assert status.power_watts == 312.5
    assert status.telemetry_available is True
    # 没有控制面租约但存在明显显存/计算活动，必须保守标为未纳管，不能误报空闲。
    assert status.resource_state == "unmanaged"
    assert status.degraded_reason is None


async def test_gpu_status_only_reports_idle_with_healthy_low_activity_telemetry(
    isolated_session_factory,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_vector_payload(
                (DCGM_FB_USED, "0", "256"),
                (DCGM_FB_FREE, "0", "23268"),
                (DCGM_FB_RESERVED, "0", "52"),
                (DCGM_GPU_UTIL, "0", "0"),
                (DCGM_GPU_TEMP, "0", "42"),
                (DCGM_POWER_USAGE, "0", "26"),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        prometheus = PrometheusClient("http://prometheus:9090", 1, http_client=http_client)
        async with isolated_session_factory() as session:
            status = (await get_gpu_statuses(session, prometheus, gpu_count=1))[0]

    assert status.telemetry_available is True
    assert status.resource_state == "idle"


async def test_prometheus_timeout_becomes_explicit_degraded_gpu_data(
    isolated_session_factory,
) -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("test timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as http_client:
        prometheus = PrometheusClient("http://prometheus:9090", 0.1, http_client=http_client)
        with pytest.raises(PrometheusUnavailableError, match="查询超时"):
            await prometheus.query(DCGM_GPU_UTIL)
        async with isolated_session_factory() as session:
            statuses = await get_gpu_statuses(session, prometheus, gpu_count=2)
    assert all(not item.telemetry_available for item in statuses)
    assert all(item.memory_used_mib is None for item in statuses)
    assert all(item.resource_state == "unknown" for item in statuses)
    assert {item.degraded_reason for item in statuses} == {"Prometheus 查询超时"}


def test_gpu_api_without_prometheus_is_degraded_not_zero(client) -> None:  # type: ignore[no-untyped-def]
    response = client.get("/api/v1/system/gpus")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(item["telemetry_available"] is False for item in body)
    assert all(item["memory_used_mib"] is None for item in body)
    assert all(item["resource_state"] == "unknown" for item in body)
    assert {item["degraded_reason"] for item in body} == {"Prometheus 未配置"}


class RecordingHistoryPrometheus:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def query_range(self, promql: str, **_kwargs):  # type: ignore[no-untyped-def]
        self.queries.append(promql)
        now = datetime.now(UTC).timestamp()
        return [
            PrometheusRangeSeries(
                labels={"__name__": DCGM_GPU_UTIL, "gpu": "0"},
                samples=(
                    PrometheusSample(timestamp=now - 30, value=20),
                    PrometheusSample(timestamp=now, value=40),
                ),
            )
        ]


def test_history_whitelist_and_query_limits_block_promql_injection(client) -> None:  # type: ignore[no-untyped-def]
    prometheus = RecordingHistoryPrometheus()
    app.dependency_overrides[get_prometheus_client] = lambda: prometheus
    now = datetime.now(UTC)
    try:
        malicious = client.get(
            "/api/v1/system/gpus/0/history",
            params={
                "metric": "utilization} or on() vector(1) #",
                "start": (now - timedelta(hours=1)).isoformat(),
                "end": now.isoformat(),
                "step_seconds": 30,
            },
        )
        assert malicious.status_code == 422
        assert prometheus.queries == []

        excessive = client.get(
            "/api/v1/system/gpus/0/history",
            params={
                "metric": "utilization",
                "start": (now - timedelta(days=7)).isoformat(),
                "end": now.isoformat(),
                "step_seconds": 5,
            },
        )
        assert excessive.status_code == 422
        assert prometheus.queries == []

        valid = client.get(
            "/api/v1/system/gpus/0/history",
            params={
                "metric": "utilization",
                "start": (now - timedelta(minutes=1)).isoformat(),
                "end": now.isoformat(),
                "step_seconds": 30,
            },
        )
        assert valid.status_code == 200, valid.text
        assert valid.json()["telemetry_available"] is True
        assert len(valid.json()["points"]) == 2
        assert prometheus.queries == ['DCGM_FI_DEV_GPU_UTIL{gpu="0"}']
    finally:
        app.dependency_overrides.pop(get_prometheus_client, None)


async def _seed_dashboard(session: AsyncSession) -> None:
    now = datetime.now(UTC)
    asset = ModelAsset(
        name="dashboard-model",
        source_type=ModelSourceType.MANUAL,
        local_path=f"/models/{uuid.uuid4()}",
        model_kind=ModelKind.BASE,
        status=AssetStatus.READY,
    )
    dataset = Dataset(
        name="dashboard-dataset",
        dataset_type=DatasetType.SFT,
        status=DatasetStatus.READY,
        file_name="dashboard.jsonl",
        local_path=f"/datasets/{uuid.uuid4()}.jsonl",
    )
    session.add_all([asset, dataset])
    await session.flush()
    deployment = Deployment(
        name="dashboard-deployment",
        served_model_name=f"served-{uuid.uuid4()}",
        model_asset_id=asset.id,
        task_type=DeploymentTaskType.GENERATE,
        desired_state=DesiredServiceState.RUNNING,
        actual_state=DeploymentState.QUEUED,
        gpu_ids=[0],
    )
    training = TrainingJob(
        name="dashboard-training",
        model_asset_id=asset.id,
        dataset_id=dataset.id,
        stage=TrainingStage.SFT,
        algorithm=TrainingAlgorithm.LORA,
        actual_state=JobState.RUNNING,
        gpu_ids=[0],
        output_dir="/checkpoints/dashboard-training",
    )
    evaluation = EvaluationRun(
        name="dashboard-evaluation",
        base_model_asset_id=asset.id,
        candidate_model_asset_id=asset.id,
        builtin_datasets=["ceval"],
        base_template=EvaluationTemplate.BASE,
        candidate_template=EvaluationTemplate.BASE,
        output_dir=f"/srv/openllmops/evaluation-output/{uuid.uuid4()}",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        concurrency=4,
        max_tokens=32,
        actual_state=JobState.QUEUED,
        gpu_ids=[1],
    )
    model_import = ModelImportJob(
        name="dashboard-import",
        source=ModelImportSource.HUGGINGFACE,
        repository="example/model",
        model_kind=ModelKind.BASE,
        status=ModelImportStatus.PENDING,
        progress_completed=0,
    )
    session.add_all([deployment, training, evaluation, model_import])
    await session.flush()
    session.add(
        GPULease(
            gpu_index=0,
            lease_group_id=uuid.uuid4(),
            owner_type=LeaseOwnerType.TRAINING,
            owner_id=training.id,
            owner_name=training.name,
            generation=1,
            acquired_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=30),
        )
    )


async def test_dashboard_summary_aggregates_counts_leases_and_recent_activity(
    isolated_session_factory,
) -> None:
    async with isolated_session_factory() as session, session.begin():
        await _seed_dashboard(session)
    async with isolated_session_factory() as session:
        summary = await build_dashboard_summary(session, gpu_count=2)

    assert summary.models.total == 1 == summary.models.ready
    assert summary.deployments.queued == 1
    assert summary.training_jobs.running == 1
    assert summary.evaluation_runs.queued == 1
    assert summary.queue.total == 3
    assert summary.queue.model_imports == 1
    assert summary.gpus.leased == 1 and summary.gpus.free == 1
    assert summary.gpus.leases[0].owner_name == "dashboard-training"
    assert {item.resource_type for item in summary.recent_activity} >= {
        "model_asset",
        "model_import",
        "deployment",
        "training_job",
        "evaluation_run",
    }


def test_dashboard_summary_api_contract(client) -> None:  # type: ignore[no-untyped-def]
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "generated_at",
        "models",
        "deployments",
        "training_jobs",
        "evaluation_runs",
        "queue",
        "gpus",
        "recent_activity",
    }
    assert body["queue"]["total"] == sum(
        body["queue"][field] for field in ("deployments", "training_jobs", "evaluation_runs", "model_imports")
    )
