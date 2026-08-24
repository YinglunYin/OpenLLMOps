import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.models.enums import ModelImportSource, ModelImportStatus, ModelKind
from app.schemas.common import ORMModel, TimestampedModel

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ModelImportCreate(BaseModel):
    # 凭证等未声明字段必须被拒绝，避免调用方误把 secret 放入请求与数据库。
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    source: ModelImportSource
    repository: str | None = Field(default=None, min_length=3, max_length=512)
    revision: str | None = Field(default=None, min_length=1, max_length=255)
    source_directory: str | None = Field(default=None, min_length=1, max_length=255)
    model_kind: ModelKind

    @model_validator(mode="after")
    def validate_source_fields(self) -> "ModelImportCreate":
        if self.source == ModelImportSource.CONTROLLED_DIRECTORY:
            if self.repository is not None or self.revision is not None:
                raise ValueError("受控目录导入不能填写 repository 或 revision")
            if self.source_directory is None:
                raise ValueError("受控目录导入必须选择 source_directory")
            if (
                self.source_directory in {".", ".."}
                or "/" in self.source_directory
                or "\\" in self.source_directory
                or self.source_directory.startswith(".")
            ):
                raise ValueError("source_directory 必须是 inbox 下的一层普通目录名")
        else:
            if self.source_directory is not None:
                raise ValueError("在线导入不能填写 source_directory")
            if self.repository is None or not REPOSITORY_PATTERN.fullmatch(self.repository):
                raise ValueError("repository 必须使用 namespace/model 格式")
        return self


class ModelImportRead(TimestampedModel):
    name: str
    source: ModelImportSource
    repository: str | None
    revision: str | None
    source_directory: str | None
    model_kind: ModelKind
    status: ModelImportStatus
    progress_completed: int
    progress_total: int | None
    started_at: datetime | None
    finished_at: datetime | None
    result_asset_id: uuid.UUID | None
    manifest_json: dict[str, Any] | None
    error_message: str | None

    @computed_field
    @property
    def progress_percent(self) -> float | None:
        if not self.progress_total:
            return None
        return round(min(100.0, self.progress_completed * 100 / self.progress_total), 2)


class InboxCandidateRead(ORMModel):
    name: str
    path: str
    file_count: int
    size_bytes: int
    ready_for_import: bool
    reason: str | None = None
