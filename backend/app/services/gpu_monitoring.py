import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import GPULease
from app.schemas.monitoring import (
    GPUHistoryMetric,
    GPUHistoryPoint,
    GPUHistoryRead,
    GPUStatusRead,
)
from app.services.prometheus import (
    PrometheusClient,
    PrometheusError,
    PrometheusInstantSeries,
    PrometheusResponseError,
)

MAX_HISTORY_SPAN_SECONDS = 7 * 24 * 60 * 60
MIN_HISTORY_STEP_SECONDS = 5
MAX_HISTORY_STEP_SECONDS = 60 * 60
MAX_HISTORY_POINTS = 2000

DCGM_FB_TOTAL = "DCGM_FI_DEV_FB_TOTAL"
DCGM_FB_USED = "DCGM_FI_DEV_FB_USED"
DCGM_FB_FREE = "DCGM_FI_DEV_FB_FREE"
DCGM_FB_RESERVED = "DCGM_FI_DEV_FB_RESERVED"
DCGM_GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
DCGM_GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
DCGM_POWER_USAGE = "DCGM_FI_DEV_POWER_USAGE"

SUPPORTED_INSTANT_METRICS = {
    DCGM_FB_TOTAL,
    DCGM_FB_USED,
    DCGM_FB_FREE,
    DCGM_FB_RESERVED,
    DCGM_GPU_UTIL,
    DCGM_GPU_TEMP,
    DCGM_POWER_USAGE,
}
DCGM_INSTANT_QUERY = (
    '{__name__=~"DCGM_FI_DEV_(FB_TOTAL|FB_USED|FB_FREE|FB_RESERVED|GPU_UTIL|GPU_TEMP|POWER_USAGE)"}'
)
HISTORY_METRICS: dict[GPUHistoryMetric, tuple[str, str]] = {
    GPUHistoryMetric.UTILIZATION: (DCGM_GPU_UTIL, "%"),
    GPUHistoryMetric.MEMORY_USED_MIB: (DCGM_FB_USED, "MiB"),
    GPUHistoryMetric.MEMORY_FREE_MIB: (DCGM_FB_FREE, "MiB"),
    GPUHistoryMetric.TEMPERATURE_CELSIUS: (DCGM_GPU_TEMP, "°C"),
    GPUHistoryMetric.POWER_WATTS: (DCGM_POWER_USAGE, "W"),
}
GPU_INDEX_PATTERN = re.compile(r"^[0-9]+$")


def _parse_gpu_index(labels: dict[str, str]) -> int:
    raw_index = labels.get("gpu")
    if raw_index is None or not GPU_INDEX_PATTERN.fullmatch(raw_index):
        raise PrometheusResponseError("Prometheus DCGM 指标缺少有效 gpu 标签")
    return int(raw_index)


def _validate_dcgm_value(metric_name: str, value: float) -> None:
    if metric_name in {DCGM_FB_TOTAL, DCGM_FB_USED, DCGM_FB_FREE, DCGM_FB_RESERVED}:
        valid = 0 <= value <= 1024 * 1024 * 1024
    elif metric_name == DCGM_GPU_UTIL:
        valid = 0 <= value <= 100
    elif metric_name == DCGM_GPU_TEMP:
        valid = -100 <= value <= 300
    elif metric_name == DCGM_POWER_USAGE:
        valid = 0 <= value <= 1_000_000
    else:
        valid = False
    if not valid:
        raise PrometheusResponseError("Prometheus DCGM 指标值超出有效范围")


def _index_instant_series(
    series: list[PrometheusInstantSeries],
    gpu_count: int,
) -> tuple[dict[int, dict[str, float]], dict[int, str]]:
    values: dict[int, dict[str, float]] = {index: {} for index in range(gpu_count)}
    names: dict[int, str] = {}
    for item in series:
        metric_name = item.labels.get("__name__")
        if metric_name not in SUPPORTED_INSTANT_METRICS:
            raise PrometheusResponseError("Prometheus 返回了非预期 DCGM 指标")
        gpu_index = _parse_gpu_index(item.labels)
        # GPU_COUNT 是本机调度边界；同一 Prometheus 中其他主机/GPU 不进入本机响应。
        if gpu_index >= gpu_count:
            continue
        if metric_name in values[gpu_index]:
            raise PrometheusResponseError("Prometheus 返回了重复 DCGM GPU 时序")
        _validate_dcgm_value(metric_name, item.sample.value)
        values[gpu_index][metric_name] = item.sample.value
        model_name = item.labels.get("modelName")
        if model_name:
            if len(model_name) > 256:
                raise PrometheusResponseError("Prometheus GPU 名称标签过长")
            previous_name = names.get(gpu_index)
            if previous_name is not None and previous_name != model_name:
                raise PrometheusResponseError("Prometheus 返回了冲突的 GPU 名称")
            names[gpu_index] = model_name
    return values, names


def _degraded_statuses(
    gpu_count: int,
    leases: dict[int, GPULease],
    reason: str,
) -> list[GPUStatusRead]:
    return [
        _build_gpu_status(index, {}, None, leases.get(index), unavailable_reason=reason)
        for index in range(gpu_count)
    ]


def _build_gpu_status(
    gpu_index: int,
    values: dict[str, float],
    name: str | None,
    lease: GPULease | None,
    *,
    unavailable_reason: str | None = None,
) -> GPUStatusRead:
    total = values.get(DCGM_FB_TOTAL)
    used = values.get(DCGM_FB_USED)
    free = values.get(DCGM_FB_FREE)
    reserved = values.get(DCGM_FB_RESERVED)
    # 默认 DCGM Exporter 提供 used/free/reserved；三者齐全时可精确还原硬件总显存。
    if total is None and used is not None and free is not None and reserved is not None:
        total = used + free + reserved
    if total is not None and (
        (used is not None and used > total + 1)
        or (free is not None and free > total + 1)
        or (used is not None and free is not None and used + free > total + 1)
    ):
        raise PrometheusResponseError("Prometheus DCGM 显存指标彼此矛盾")

    required = {
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_percent": values.get(DCGM_GPU_UTIL),
        "temperature_celsius": values.get(DCGM_GPU_TEMP),
        "power_watts": values.get(DCGM_POWER_USAGE),
    }
    telemetry_available = bool(values) and unavailable_reason is None
    degraded_reason = unavailable_reason
    if telemetry_available:
        missing = [field for field, value in required.items() if value is None]
        if missing:
            degraded_reason = f"DCGM 指标不完整，缺少：{', '.join(missing)}"
    elif degraded_reason is None:
        degraded_reason = f"未收到 GPU {gpu_index} 的 DCGM 指标"

    return GPUStatusRead(
        index=gpu_index,
        name=name,
        memory_total_mib=total,
        memory_used_mib=used,
        memory_free_mib=free,
        utilization_percent=values.get(DCGM_GPU_UTIL),
        temperature_celsius=values.get(DCGM_GPU_TEMP),
        power_watts=values.get(DCGM_POWER_USAGE),
        telemetry_available=telemetry_available,
        degraded_reason=degraded_reason,
        owner_type=lease.owner_type if lease else None,
        owner_id=lease.owner_id if lease else None,
        owner_name=lease.owner_name if lease else None,
        lease_expires_at=lease.expires_at if lease else None,
    )


async def get_gpu_statuses(
    session: AsyncSession,
    prometheus: PrometheusClient | None,
    gpu_count: int,
) -> list[GPUStatusRead]:
    lease_rows = await session.scalars(select(GPULease).order_by(GPULease.gpu_index))
    leases = {lease.gpu_index: lease for lease in lease_rows}
    if prometheus is None:
        return _degraded_statuses(gpu_count, leases, "Prometheus 未配置")
    try:
        series = await prometheus.query(DCGM_INSTANT_QUERY)
        values, names = _index_instant_series(series, gpu_count)
        return [
            _build_gpu_status(index, values[index], names.get(index), leases.get(index))
            for index in range(gpu_count)
        ]
    except PrometheusError as exc:
        return _degraded_statuses(gpu_count, leases, str(exc))


async def get_gpu_history(
    prometheus: PrometheusClient | None,
    *,
    gpu_index: int,
    metric: GPUHistoryMetric,
    start: datetime,
    end: datetime,
    step_seconds: int,
    max_points: int,
) -> GPUHistoryRead:
    metric_name, unit = HISTORY_METRICS[metric]
    common = {
        "gpu_index": gpu_index,
        "metric": metric,
        "unit": unit,
        "start": start,
        "end": end,
        "step_seconds": step_seconds,
    }
    if prometheus is None:
        return GPUHistoryRead(
            **common,
            telemetry_available=False,
            degraded_reason="Prometheus 未配置",
            points=[],
        )
    # PromQL 只由服务端白名单和已验证整数拼装，不插入任何用户提供的表达式。
    promql = f'{metric_name}{{gpu="{gpu_index}"}}'
    try:
        series = await prometheus.query_range(
            promql,
            start=start.timestamp(),
            end=end.timestamp(),
            step_seconds=step_seconds,
            max_samples=max_points,
        )
        if not series:
            return GPUHistoryRead(
                **common,
                telemetry_available=False,
                degraded_reason="Prometheus 未返回该 GPU 指标",
                points=[],
            )
        if len(series) != 1:
            raise PrometheusResponseError("Prometheus 返回了重复 DCGM GPU 时序")
        item = series[0]
        if item.labels.get("__name__") != metric_name or _parse_gpu_index(item.labels) != gpu_index:
            raise PrometheusResponseError("Prometheus 返回了不匹配的 DCGM GPU 时序")
        points: list[GPUHistoryPoint] = []
        for sample in item.samples:
            _validate_dcgm_value(metric_name, sample.value)
            try:
                timestamp = datetime.fromtimestamp(sample.timestamp, tz=UTC)
            except (OverflowError, OSError, ValueError) as exc:
                raise PrometheusResponseError("Prometheus 时间戳超出有效范围") from exc
            points.append(GPUHistoryPoint(timestamp=timestamp, value=sample.value))
        return GPUHistoryRead(
            **common,
            telemetry_available=bool(points),
            degraded_reason=None if points else "Prometheus 未返回该时间范围的数据点",
            points=points,
        )
    except PrometheusError as exc:
        return GPUHistoryRead(
            **common,
            telemetry_available=False,
            degraded_reason=str(exc),
            points=[],
        )
