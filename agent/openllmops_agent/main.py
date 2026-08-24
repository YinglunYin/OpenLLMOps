from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from docker.errors import DockerException
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .agent_contract import (
    AgentCommand,
    AgentCommandResponse,
    AgentWorkloadState,
    HMACSigner,
    NonceReplayCache,
    SignatureVerificationError,
    canonical_json,
)
from .command_service import CommandProcessor
from .config import Settings
from .docker_runner import (
    DockerRunner,
    InvalidWorkload,
    RunnerError,
    WorkloadConflict,
    WorkloadNotFound,
    docker_error_message,
)
from .gpu import read_gpu_inventory
from .schemas import (
    GPUInventory,
    HealthResponse,
    WorkloadInfo,
)

MAX_COMMAND_BODY_BYTES = 2 * 1024 * 1024
NULL_REQUEST_ID = UUID(int=0)
logger = logging.getLogger(__name__)

settings = Settings()
action_counter = Counter(
    "openllmops_agent_actions_total",
    "node-agent 工作负载操作次数",
    ("action", "result"),
)
workload_gauge = Gauge("openllmops_agent_workloads", "node-agent 管理的容器数量", ("kind", "status"))


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings.ensure_layout()
    runner = DockerRunner(settings)
    runner.initialize()
    application.state.runner = runner
    application.state.command_processor = CommandProcessor(settings, runner)
    try:
        yield
    finally:
        runner.close()


app = FastAPI(
    title="OpenLLMOps Node Agent",
    version="0.1.0",
    docs_url="/internal/docs",
    openapi_url="/internal/openapi.json",
    lifespan=lifespan,
)
app.state.hmac_signer = HMACSigner(
    settings.node_agent_token.get_secret_value(), settings.node_agent_clock_skew_seconds
)
app.state.nonce_cache = NonceReplayCache(settings.node_agent_clock_skew_seconds)


def require_agent_token(
    token: Annotated[str | None, Header(alias="X-Node-Agent-Token")] = None,
) -> None:
    expected = settings.node_agent_token.get_secret_value()
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="node-agent token 无效")


def get_runner(request: Request) -> DockerRunner:
    return request.app.state.runner


def get_command_processor(request: Request) -> CommandProcessor:
    return request.app.state.command_processor


AgentAuth = Annotated[None, Depends(require_agent_token)]
Runner = Annotated[DockerRunner, Depends(get_runner)]
Processor = Annotated[CommandProcessor, Depends(get_command_processor)]


@app.exception_handler(InvalidWorkload)
async def invalid_handler(_: Request, exc: InvalidWorkload) -> Response:
    return _json_error(422, str(exc))


@app.exception_handler(WorkloadConflict)
async def conflict_handler(_: Request, exc: WorkloadConflict) -> Response:
    return _json_error(status.HTTP_409_CONFLICT, str(exc))


@app.exception_handler(WorkloadNotFound)
async def not_found_handler(_: Request, exc: WorkloadNotFound) -> Response:
    return _json_error(status.HTTP_404_NOT_FOUND, str(exc))


@app.exception_handler(RunnerError)
async def runner_handler(_: Request, exc: RunnerError) -> Response:
    return _json_error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.exception_handler(DockerException)
async def docker_handler(_: Request, exc: DockerException) -> Response:
    return _json_error(status.HTTP_503_SERVICE_UNAVAILABLE, docker_error_message(exc))


def _json_error(code: int, detail: str) -> Response:
    import json

    return Response(
        content=json.dumps({"detail": detail}, ensure_ascii=False),
        status_code=code,
        media_type="application/json",
    )


def _signed_contract_response(request: Request, response: AgentCommandResponse, status_code: int) -> Response:
    body = canonical_json(response)
    headers = request.app.state.hmac_signer.sign(body)
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json",
        headers=headers,
    )


def _signed_contract_error(
    request: Request,
    *,
    request_id: UUID,
    status_code: int,
    message: str,
    error_code: str,
) -> Response:
    return _signed_contract_response(
        request,
        AgentCommandResponse(
            request_id=request_id,
            accepted=False,
            observed_state=AgentWorkloadState.FAILED,
            observed_at=datetime.now(UTC),
            message=message,
            error_code=error_code,
        ),
        status_code,
    )


@app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
def health(runner: Runner) -> HealthResponse:
    runner.client.ping()
    return HealthResponse(status="ok", docker_connected=True, runtime_network=settings.runtime_network)


@app.get("/metrics", include_in_schema=False)
def metrics(runner: Runner) -> Response:
    workload_gauge.clear()
    for item in runner.list_workloads():
        workload_gauge.labels(kind=item.kind, status=item.status).inc()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/gpus", response_model=GPUInventory, dependencies=[Depends(require_agent_token)])
def gpus(runner: Runner) -> GPUInventory:
    return read_gpu_inventory(settings.gpu_count, runner.gpu_allocations())


@app.get(
    "/v1/workloads",
    response_model=list[WorkloadInfo],
    dependencies=[Depends(require_agent_token)],
)
def list_workloads(runner: Runner) -> list[WorkloadInfo]:
    return runner.list_workloads()


@app.get(
    "/v1/workloads/{name}",
    response_model=WorkloadInfo,
    dependencies=[Depends(require_agent_token)],
)
def get_workload(name: str, runner: Runner) -> WorkloadInfo:
    return runner.get_workload(name)


@app.get(
    "/v1/workloads/{name}/logs",
    response_class=Response,
    dependencies=[Depends(require_agent_token)],
)
def workload_logs(name: str, runner: Runner, tail: int = Query(default=500, ge=1, le=5000)) -> Response:
    return Response(runner.logs(name, tail), media_type="text/plain; charset=utf-8")


@app.post(
    "/v1/workloads/commands",
    response_class=Response,
)
async def execute_command(request: Request, processor: Processor) -> Response:
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_COMMAND_BODY_BYTES:
        return _signed_contract_error(
            request,
            request_id=NULL_REQUEST_ID,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            message="AgentCommand 请求体超过 2 MiB 限制",
            error_code="body_too_large",
        )
    body = await request.body()
    if len(body) > MAX_COMMAND_BODY_BYTES:
        return _signed_contract_error(
            request,
            request_id=NULL_REQUEST_ID,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            message="AgentCommand 请求体超过 2 MiB 限制",
            error_code="body_too_large",
        )
    try:
        request.app.state.nonce_cache.verify(request.app.state.hmac_signer, body, request.headers)
    except SignatureVerificationError as exc:
        action_counter.labels(action="contract", result="auth_failure").inc()
        return _signed_contract_error(
            request,
            request_id=NULL_REQUEST_ID,
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=str(exc),
            error_code="invalid_signature",
        )
    try:
        command = AgentCommand.model_validate_json(body)
    except ValidationError:
        action_counter.labels(action="contract", result="validation_failure").inc()
        return _signed_contract_error(
            request,
            request_id=NULL_REQUEST_ID,
            status_code=422,
            message="请求体不是有效的 AgentCommand",
            error_code="invalid_command",
        )
    try:
        normalized_body = canonical_json(command)
    except (TypeError, ValueError):
        return _signed_contract_error(
            request,
            request_id=command.request_id,
            status_code=422,
            message="AgentCommand 包含非有限数值或不可编码字段",
            error_code="invalid_command",
        )
    if normalized_body != body:
        return _signed_contract_error(
            request,
            request_id=command.request_id,
            status_code=422,
            message="AgentCommand 必须使用规范化 JSON 原始字节",
            error_code="non_canonical_json",
        )
    try:
        result = await run_in_threadpool(processor.execute, command)
    except (RunnerError, DockerException, OSError, RuntimeError) as exc:
        action_counter.labels(action=command.action.value, result="failure").inc()
        message = (
            docker_error_message(exc) if isinstance(exc, DockerException) else "node-agent 执行器暂时不可用"
        )
        return _signed_contract_error(
            request,
            request_id=command.request_id,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
            error_code="runner_unavailable",
        )
    except Exception:
        # 合同边界必须对 5xx 原始响应也签名；否则控制面会把未签名的网关/框架错误
        # 与 agent 的真实观察混淆。未知异常只写服务端日志，不把内部细节回传。
        logger.exception("node-agent 执行 AgentCommand 时发生未处理异常")
        action_counter.labels(action=command.action.value, result="failure").inc()
        return _signed_contract_error(
            request,
            request_id=command.request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="node-agent 内部错误",
            error_code="internal_error",
        )
    action_counter.labels(
        action=command.action.value,
        result="success" if result.response.accepted else "rejected",
    ).inc()
    return _signed_contract_response(request, result.response, result.status_code)
