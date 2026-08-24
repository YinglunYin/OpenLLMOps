from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import GPULease
from app.schemas import GPULeaseRead

router = APIRouter(prefix="/system", tags=["系统"])


@router.get("/capabilities", response_model=dict[str, Any])
async def get_capabilities() -> dict[str, Any]:
    settings = get_settings()
    return {
        "gpu_count": settings.gpu_count,
        "gpu_policy": "exclusive_non_preemptive",
        "model_format": ["safetensors"],
        "trust_remote_code": False,
        "training": {
            "cpt": ["lora"],
            "sft": ["freeze", "lora", "qlora"],
        },
        "evaluation": ["ceval", "cmmlu", "custom_jsonl"],
        "openai_endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/embeddings",
        ],
    }


@router.get("/gpu-leases", response_model=list[GPULeaseRead])
async def list_gpu_leases(
    session: AsyncSession = Depends(get_db),
) -> list[GPULease]:
    result = await session.scalars(select(GPULease).order_by(GPULease.gpu_index))
    return list(result)
