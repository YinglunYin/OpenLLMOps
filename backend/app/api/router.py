from fastapi import APIRouter, Depends

from app.api.routes import (
    api_keys,
    audit_logs,
    auth,
    datasets,
    deployments,
    evaluations,
    health,
    model_assets,
    model_imports,
    openai_gateway,
    system,
    training_jobs,
)
from app.core.config import get_settings
from app.core.security import require_admin_auth, require_api_key

root_router = APIRouter()
root_router.include_router(health.router)
root_router.include_router(auth.router, prefix=get_settings().api_prefix)

protected_control = APIRouter(dependencies=[Depends(require_admin_auth)])
protected_control.include_router(model_assets.router)
protected_control.include_router(model_imports.router)
protected_control.include_router(model_imports.inbox_router)
protected_control.include_router(datasets.router)
protected_control.include_router(deployments.router)
protected_control.include_router(training_jobs.router)
protected_control.include_router(evaluations.router)
protected_control.include_router(api_keys.router)
protected_control.include_router(audit_logs.router)
protected_control.include_router(system.router)
root_router.include_router(protected_control, prefix=get_settings().api_prefix)

# OpenAI 兼容接口刻意不读取管理员 Cookie，只接受请求头中的 API Key。
protected_openai = APIRouter(dependencies=[Depends(require_api_key)])
protected_openai.include_router(openai_gateway.router)
root_router.include_router(protected_openai)
