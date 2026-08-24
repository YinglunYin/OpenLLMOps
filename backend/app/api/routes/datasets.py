import json
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models import Dataset
from app.models.enums import DatasetStatus, DatasetType
from app.schemas import DatasetRead, DatasetUpdate
from app.services.crud import commit_or_conflict, get_or_404
from app.services.dataset_files import (
    ensure_path_within,
    preview_jsonl,
    validate_and_store_jsonl,
)

router = APIRouter(prefix="/datasets", tags=["训练与评测数据集"])


def _validate_upload_filename(filename: str | None) -> str:
    if not filename or not filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=422, detail="仅支持 .jsonl 数据集")
    if (
        len(filename) > 255
        or "/" in filename
        or "\\" in filename
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
    ):
        raise HTTPException(status_code=422, detail="文件名必须是不超过 255 字符的安全普通文件名")
    return filename


@router.post("/upload", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(min_length=1, max_length=128),
    dataset_type: DatasetType = Form(),
    version: str = Form(
        default="v1.0.0",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    ),
    description: str | None = Form(default=None),
    file: UploadFile = File(),
    session: AsyncSession = Depends(get_db),
) -> Dataset:
    original_filename = _validate_upload_filename(file.filename)

    settings = get_settings()
    storage_name = f"{uuid.uuid4()}.jsonl"
    final_path = settings.dataset_root / storage_name
    temporary_path = settings.dataset_root / f".{storage_name}.part"
    dataset_id = uuid.uuid4()
    registered = False
    file_closed = False
    try:
        try:
            result = await anyio.to_thread.run_sync(
                validate_and_store_jsonl,
                file.file,
                temporary_path,
                final_path,
                dataset_type,
            )
        except ValueError as exc:
            try:
                detail: Any = json.loads(str(exc))
            except json.JSONDecodeError:
                detail = str(exc)
            raise HTTPException(status_code=422, detail=detail) from exc

        # close 也放在受控事务范围内：若框架临时文件关闭或任务取消失败，已经
        # os.replace 的最终文件仍会走下方孤儿回收。
        await file.close()
        file_closed = True
        record_count, size_bytes, sha256, errors, schema_summary = result
        # 首版把版本作为不可变上传记录的受控元数据；新版本必须重新上传并获得新 UUID。
        schema_summary = {**schema_summary, "version": version}
        dataset = Dataset(
            id=dataset_id,
            name=name,
            dataset_type=dataset_type,
            status=DatasetStatus.READY,
            file_name=original_filename,
            local_path=str(final_path),
            record_count=record_count,
            size_bytes=size_bytes,
            sha256=sha256,
            schema_summary=schema_summary,
            validation_errors=errors,
            description=description,
        )
        session.add(dataset)
        await commit_or_conflict(session, "数据集登记失败")
        registered = True
        await session.refresh(dataset)
        return dataset
    except BaseException:
        if not registered and final_path.exists():
            # 文件已经原子落盘，数据库登记失败时需要回收；若 commit 结果不确定，则先
            # 回查固定 UUID，宁可保留可审计的孤儿文件，也不能删除已提交记录引用的文件。
            with anyio.CancelScope(shield=True):
                should_remove = False
                try:
                    await session.rollback()
                    should_remove = await session.get(Dataset, dataset_id) is None
                except Exception:
                    should_remove = False
                if should_remove:
                    try:
                        controlled = ensure_path_within(final_path, settings.dataset_root)
                        await anyio.to_thread.run_sync(controlled.unlink, True)
                    except (ValueError, OSError):
                        pass
        raise
    finally:
        if not file_closed:
            with anyio.CancelScope(shield=True):
                # 若 close 本身是原始失败，上方 except 已保留它；清理阶段不能覆盖。
                with suppress(BaseException):
                    await file.close()


@router.get("", response_model=list[DatasetRead])
async def list_datasets(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> list[Dataset]:
    result = await session.scalars(
        select(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result)


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Dataset:
    return await get_or_404(session, Dataset, dataset_id, "数据集")


@router.get("/{dataset_id}/preview", response_model=list[dict[str, Any]])
async def preview_dataset(
    dataset_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    dataset = await get_or_404(session, Dataset, dataset_id, "数据集")
    settings = get_settings()
    try:
        path = ensure_path_within(Path(dataset.local_path), settings.dataset_root)
        return await anyio.to_thread.run_sync(preview_jsonl, path, limit)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"数据集无法预览：{exc}") from exc


@router.patch("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(
    dataset_id: uuid.UUID,
    payload: DatasetUpdate,
    session: AsyncSession = Depends(get_db),
) -> Dataset:
    dataset = await get_or_404(session, Dataset, dataset_id, "数据集")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(dataset, field, value)
    await commit_or_conflict(session, "数据集更新冲突")
    await session.refresh(dataset)
    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Response:
    dataset = await get_or_404(session, Dataset, dataset_id, "数据集")
    await session.delete(dataset)
    await commit_or_conflict(session, "数据集正在被训练或评测任务引用，不能删除")
    # 先提交数据库删除再删除文件；即使文件删除失败，也不会恢复已删除的业务记录。
    try:
        controlled_path = ensure_path_within(Path(dataset.local_path), get_settings().dataset_root)
        await anyio.to_thread.run_sync(controlled_path.unlink, True)
    except (ValueError, OSError):
        pass
    return Response(status_code=status.HTTP_204_NO_CONTENT)
