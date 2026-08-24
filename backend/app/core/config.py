import json
from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from argon2 import extract_parameters
from argon2.exceptions import InvalidHashError
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """统一读取环境变量。

    路径默认值面向容器部署；测试可通过环境变量切换到临时目录。敏感字段不提供
    可用于生产的默认值，避免误把示例密钥带入内网环境。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "OpenLLMOps API"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./openllmops.db"
    auto_create_tables: bool = False

    auth_enabled: bool = True
    admin_username: str = "admin"
    admin_password_hash: str | None = None
    admin_api_key: str | None = None
    api_key_header: str = "X-API-Key"
    api_key_pepper: str = ""
    session_signing_key: str | None = None
    session_cookie_name: str = "openllmops_admin_session"
    session_cookie_secure: bool = True
    session_ttl_seconds: int = Field(default=8 * 60 * 60, ge=300, le=24 * 60 * 60)
    csrf_header: str = "X-CSRF-Token"
    request_id_header: str = "X-Request-ID"
    trusted_proxy_cidrs: Annotated[list[str], NoDecode] = Field(default_factory=list)

    node_agent_url: str = "http://node-agent:9000"
    # 首版沿用 token 环境变量名，但其内容实际作为双向 HMAC 共享密钥使用。
    node_agent_token: str | None = None
    reconciler_enabled: bool = False
    reconciler_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    gpu_lease_ttl_seconds: int = Field(default=30, ge=10, le=600)
    node_agent_clock_skew_seconds: int = Field(default=30, ge=5, le=300)
    node_agent_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    vllm_internal_api_key: str | None = None

    prometheus_url: str | None = None
    prometheus_timeout_seconds: float = Field(default=3.0, gt=0, le=30)

    model_root: Path = Path("/srv/openllmops/models")
    model_inbox_root: Path = Path("/srv/openllmops/inbox")
    model_staging_root: Path = Path("/srv/openllmops/model-staging")
    model_import_coordinator_enabled: bool = False
    model_import_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    model_import_concurrency: int = Field(default=1, ge=1, le=4)
    model_import_claim_timeout_seconds: int = Field(default=120, ge=30, le=3600)
    huggingface_token_file: Path | None = None
    modelscope_token_file: Path | None = None
    dataset_root: Path = Path("/srv/openllmops/datasets")
    checkpoint_root: Path = Path("/srv/openllmops/checkpoints")
    training_artifact_max_files: int = Field(default=100_000, ge=1, le=1_000_000)
    training_artifact_max_bytes: int = Field(
        default=500 * 1024 * 1024 * 1024,
        ge=1,
        le=2 * 1024 * 1024 * 1024 * 1024,
    )
    evaluation_dataset_root: Path = Path("/srv/openllmops/evaluation-datasets")
    evaluation_output_root: Path = Path("/srv/openllmops/evaluation-output")
    node_agent_runtime_root: Path = Path("/srv/openllmops/runtime")
    evaluation_gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=0.95)
    evaluation_concurrency: int = Field(default=4, ge=1, le=32)
    evaluation_max_tokens: int = Field(default=32, ge=1, le=512)
    evaluation_allow_partial_builtins: bool = False
    gpu_count: int = Field(default=2, ge=1, le=32)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:5173"])
    proxy_timeout_seconds: float = Field(default=600.0, gt=0)

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith("/"):
            value = f"/{value}"
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_string_list(cls, value: object) -> object:
        # Compose 中通常以逗号分隔传入，配置文件也可直接使用 JSON 数组。
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("trusted_proxy_cidrs", mode="before")
    @classmethod
    def parse_trusted_proxies(cls, value: object) -> object:
        return cls.parse_string_list(value)

    @field_validator("trusted_proxy_cidrs")
    @classmethod
    def validate_trusted_proxies(cls, values: list[str]) -> list[str]:
        for value in values:
            ip_network(value, strict=False)
        return values

    @field_validator("huggingface_token_file", "modelscope_token_file", mode="before")
    @classmethod
    def normalize_optional_secret_path(cls, value: object) -> object:
        # Compose 对未配置的可选变量会传空字符串；不能把它解释为当前目录 Path('.')。
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("prometheus_url", mode="before")
    @classmethod
    def normalize_prometheus_url(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("PROMETHEUS_URL 端口无效") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("PROMETHEUS_URL 必须是不含凭证、查询参数或片段的 HTTP(S) 地址")
        return normalized

    @field_validator(
        "evaluation_dataset_root",
        "evaluation_output_root",
        "node_agent_runtime_root",
    )
    @classmethod
    def require_absolute_evaluation_roots(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("评测受控根目录必须使用绝对路径")
        return value

    @model_validator(mode="after")
    def validate_production_auth(self) -> "Settings":
        if self.environment != "production":
            return self
        if not self.auth_enabled:
            raise ValueError("生产环境禁止关闭鉴权")
        if not self.admin_password_hash or not self.admin_password_hash.startswith("$argon2"):
            raise ValueError("生产环境必须配置 Argon2 ADMIN_PASSWORD_HASH")
        try:
            extract_parameters(self.admin_password_hash)
        except InvalidHashError as exc:
            raise ValueError("生产环境 ADMIN_PASSWORD_HASH 不是有效的 Argon2 哈希") from exc
        if not self.session_signing_key or len(self.session_signing_key) < 32:
            raise ValueError("生产环境 SESSION_SIGNING_KEY 至少需要 32 个字符")
        if not self.session_cookie_secure:
            raise ValueError("生产环境会话 Cookie 必须启用 Secure")
        if not self.cors_origins or "*" in self.cors_origins:
            raise ValueError("生产环境 CORS_ORIGINS 必须是明确的可信来源")
        for token_file in (self.huggingface_token_file, self.modelscope_token_file):
            if token_file is not None and not token_file.is_absolute():
                raise ValueError("生产环境模型仓库 token file 必须使用绝对路径")
        if self.evaluation_allow_partial_builtins:
            raise ValueError("生产环境禁止启用 EVALUATION_ALLOW_PARTIAL_BUILTINS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
