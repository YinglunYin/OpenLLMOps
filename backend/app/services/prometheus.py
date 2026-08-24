import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import httpx

from app.core.config import get_settings

MAX_PROMETHEUS_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_PROMETHEUS_SERIES = 1024
MAX_LABELS_PER_SERIES = 128
MAX_LABEL_LENGTH = 1024


class PrometheusError(RuntimeError):
    """不会携带上游响应正文或 URL 的监控查询错误。"""


class PrometheusUnavailableError(PrometheusError):
    """网络、超时或 HTTP 状态导致 Prometheus 不可用。"""


class PrometheusResponseError(PrometheusError):
    """Prometheus 返回了不符合官方 API 合同的数据。"""


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    timestamp: float
    value: float


@dataclass(frozen=True, slots=True)
class PrometheusInstantSeries:
    labels: dict[str, str]
    sample: PrometheusSample


@dataclass(frozen=True, slots=True)
class PrometheusRangeSeries:
    labels: dict[str, str]
    samples: tuple[PrometheusSample, ...]


def _invalid_response() -> PrometheusResponseError:
    return PrometheusResponseError("Prometheus 响应结构无效")


def _parse_labels(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > MAX_LABELS_PER_SERIES:
        raise _invalid_response()
    labels: dict[str, str] = {}
    for key, label_value in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(label_value, str)
            or not key
            or len(key) > MAX_LABEL_LENGTH
            or len(label_value) > MAX_LABEL_LENGTH
        ):
            raise _invalid_response()
        labels[key] = label_value
    return labels


def _parse_sample(value: object) -> PrometheusSample:
    if not isinstance(value, list) or len(value) != 2:
        raise _invalid_response()
    raw_timestamp, raw_value = value
    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, (int, float)):
        raise _invalid_response()
    if not isinstance(raw_value, str) or len(raw_value) > 128:
        raise _invalid_response()
    try:
        timestamp = float(raw_timestamp)
        sample_value = float(raw_value)
    except ValueError as exc:
        raise _invalid_response() from exc
    # Prometheus 会把 NaN/Inf 编码成字符串；控制面明确拒绝这些不可展示值。
    if not math.isfinite(timestamp) or not math.isfinite(sample_value):
        raise _invalid_response()
    return PrometheusSample(timestamp=timestamp, value=sample_value)


def _extract_result(payload: object, expected_type: str) -> list[object]:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise _invalid_response()
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("resultType") != expected_type:
        raise _invalid_response()
    result = data.get("result")
    if not isinstance(result, list) or len(result) > MAX_PROMETHEUS_SERIES:
        raise _invalid_response()
    return result


class PrometheusClient:
    """最小 Prometheus HTTP API 客户端，允许注入 httpx client 做无网络测试。"""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client

    async def _request(self, path: str, params: dict[str, str | int | float]) -> Any:
        url = f"{self.base_url}{path}"

        async def send(client: httpx.AsyncClient) -> httpx.Response:
            return await client.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
                follow_redirects=False,
            )

        try:
            if self.http_client is not None:
                response = await send(self.http_client)
            else:
                # 内网监控地址不应受宿主机 HTTP_PROXY 等环境变量重定向。
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await send(client)
        except httpx.TimeoutException as exc:
            raise PrometheusUnavailableError("Prometheus 查询超时") from exc
        except httpx.HTTPError as exc:
            raise PrometheusUnavailableError("Prometheus HTTP 请求失败") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise PrometheusUnavailableError(f"Prometheus 返回 HTTP {response.status_code}")
        if len(response.content) > MAX_PROMETHEUS_RESPONSE_BYTES:
            raise PrometheusResponseError("Prometheus 响应超过大小限制")
        try:
            return response.json()
        except ValueError as exc:
            raise _invalid_response() from exc

    async def query(self, promql: str) -> list[PrometheusInstantSeries]:
        payload = await self._request("/api/v1/query", {"query": promql})
        result = _extract_result(payload, "vector")
        series: list[PrometheusInstantSeries] = []
        for item in result:
            if not isinstance(item, dict) or "histogram" in item:
                raise _invalid_response()
            labels = _parse_labels(item.get("metric"))
            sample = _parse_sample(item.get("value"))
            series.append(PrometheusInstantSeries(labels=labels, sample=sample))
        return series

    async def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step_seconds: int,
        max_samples: int,
    ) -> list[PrometheusRangeSeries]:
        payload = await self._request(
            "/api/v1/query_range",
            {
                "query": promql,
                "start": start,
                "end": end,
                "step": step_seconds,
            },
        )
        result = _extract_result(payload, "matrix")
        series: list[PrometheusRangeSeries] = []
        total_samples = 0
        for item in result:
            if not isinstance(item, dict) or "histograms" in item:
                raise _invalid_response()
            labels = _parse_labels(item.get("metric"))
            raw_samples = item.get("values")
            if not isinstance(raw_samples, list):
                raise _invalid_response()
            samples = tuple(_parse_sample(sample) for sample in raw_samples)
            total_samples += len(samples)
            if total_samples > max_samples:
                raise PrometheusResponseError("Prometheus 返回的数据点超过请求限制")
            if any(current.timestamp <= previous.timestamp for previous, current in pairwise(samples)):
                raise _invalid_response()
            series.append(PrometheusRangeSeries(labels=labels, samples=samples))
        return series


def get_prometheus_client() -> PrometheusClient | None:
    settings = get_settings()
    if settings.prometheus_url is None:
        return None
    return PrometheusClient(settings.prometheus_url, settings.prometheus_timeout_seconds)
