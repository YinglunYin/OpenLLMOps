import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException

from app.api.router import root_router
from app.core.config import get_settings
from app.core.database import AsyncSessionFactory, create_all_tables, dispose_engine, engine
from app.core.early_auth import early_large_upload_auth_middleware
from app.core.errors import (
    openllmops_http_exception_handler,
    openllmops_validation_exception_handler,
)
from app.core.metrics import metrics_middleware
from app.core.request_context import request_context_and_audit_middleware
from app.services.dataset_files import cleanup_stale_upload_parts
from app.services.gpu_scheduler import GPULeaseManager
from app.services.model_import_coordinator import (
    ModelImportCoordinator,
    build_model_importer,
)
from app.services.node_agent import NodeAgentHTTPClient
from app.services.reconciler import StateReconciler

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if settings.auto_create_tables:
        await create_all_tables()
    # SIGKILL 无法执行请求 finally；启动时只回收超过安全时龄的本系统上传临时文件。
    await anyio.to_thread.run_sync(cleanup_stale_upload_parts, settings.dataset_root)
    stop_event: asyncio.Event | None = None
    reconciler_task: asyncio.Task[None] | None = None
    import_stop_event: asyncio.Event | None = None
    import_coordinator_task: asyncio.Task[None] | None = None
    agent_client: NodeAgentHTTPClient | None = None
    if settings.reconciler_enabled:
        if not settings.node_agent_token:
            raise RuntimeError("启用 reconciler 时必须配置 NODE_AGENT_TOKEN 作为 HMAC 共享密钥")
        agent_client = NodeAgentHTTPClient(
            settings.node_agent_url,
            settings.node_agent_token,
            max_clock_skew_seconds=settings.node_agent_clock_skew_seconds,
            timeout_seconds=settings.node_agent_timeout_seconds,
        )
        reconciler = StateReconciler(
            AsyncSessionFactory,
            agent_client,
            GPULeaseManager(settings.gpu_lease_ttl_seconds),
            interval_seconds=settings.reconciler_interval_seconds,
            settings=settings,
        )
        stop_event = asyncio.Event()
        reconciler_task = asyncio.create_task(
            reconciler.run_forever(stop_event),
            name="openllmops-state-reconciler",
        )
    if settings.model_import_coordinator_enabled:
        importer = build_model_importer(
            settings.model_inbox_root,
            settings.model_staging_root,
            settings.model_root,
        )
        import_coordinator = ModelImportCoordinator(
            AsyncSessionFactory,
            importer,
            inbox_root=settings.model_inbox_root,
            staging_root=settings.model_staging_root,
            store_root=settings.model_root,
            huggingface_token_file=settings.huggingface_token_file,
            modelscope_token_file=settings.modelscope_token_file,
            poll_interval_seconds=settings.model_import_poll_interval_seconds,
            concurrency=settings.model_import_concurrency,
            claim_timeout_seconds=settings.model_import_claim_timeout_seconds,
            lock_engine=engine,
        )
        import_stop_event = asyncio.Event()
        import_coordinator_task = asyncio.create_task(
            import_coordinator.run_forever(import_stop_event),
            name="openllmops-model-import-coordinator",
        )
    try:
        yield
    finally:
        if stop_event is not None and reconciler_task is not None:
            stop_event.set()
            await reconciler_task
        if import_stop_event is not None and import_coordinator_task is not None:
            import_stop_event.set()
            await import_coordinator_task
        if agent_client is not None:
            await agent_client.aclose()
        await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)
app.add_exception_handler(HTTPException, openllmops_http_exception_handler)
app.add_exception_handler(RequestValidationError, openllmops_validation_exception_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Authorization",
        "Content-Type",
        settings.api_key_header,
        settings.csrf_header,
        settings.request_id_header,
    ],
    expose_headers=[settings.request_id_header],
)
app.middleware("http")(early_large_upload_auth_middleware)
app.middleware("http")(metrics_middleware)
app.middleware("http")(request_context_and_audit_middleware)
app.include_router(root_router)
app.mount("/metrics", make_asgi_app())
