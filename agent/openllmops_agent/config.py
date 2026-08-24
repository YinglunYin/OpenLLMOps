from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .image_policy import HARDENED_LLAMAFACTORY_IMAGE, validate_training_image_list


class Settings(BaseSettings):
    """仅从环境变量读取节点配置，避免高权限代理接受运行时改写。"""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    node_agent_token: SecretStr
    node_agent_clock_skew_seconds: int = Field(default=30, ge=5, le=300)
    docker_host: str = "tcp://docker-socket-proxy:2375"
    gpu_count: int = Field(default=4, ge=1, le=16)

    model_root: Path = Path("/srv/openllmops/models")
    dataset_root: Path = Path("/srv/openllmops/datasets")
    checkpoint_root: Path = Path("/srv/openllmops/checkpoints")
    training_config_root: Path = Path("/srv/openllmops/training-configs")
    runtime_root: Path = Path("/srv/openllmops/runtime")
    runtime_network: str = "openllmops-runtime"

    vllm_allowed_images: str = "vllm/vllm-openai:v0.10.2"
    llamafactory_allowed_images: str = HARDENED_LLAMAFACTORY_IMAGE
    workload_uid: int = Field(default=1000, ge=1)
    workload_gid: int = Field(default=1000, ge=1)
    workload_shm_size: str = "16g"
    enforce_nvml_process_check: bool = True
    log_level: str = "INFO"

    @field_validator("node_agent_token")
    @classmethod
    def validate_token(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("NODE_AGENT_TOKEN 至少需要 32 个字符")
        return value

    @field_validator("runtime_network")
    @classmethod
    def validate_network_name(cls, value: str) -> str:
        if not value or len(value) > 63 or not all(char.isalnum() or char in "_.-" for char in value):
            raise ValueError("RUNTIME_NETWORK 不是安全的 Docker 网络名称")
        return value

    @field_validator("workload_shm_size")
    @classmethod
    def validate_shm_size(cls, value: str) -> str:
        suffix = value[-1:].lower()
        number = value[:-1] if suffix in {"k", "m", "g"} else value
        if not number.isdigit() or int(number) <= 0:
            raise ValueError("WORKLOAD_SHM_SIZE 应为正整数，可带 k/m/g 后缀")
        return value.lower()

    @field_validator("llamafactory_allowed_images")
    @classmethod
    def validate_llamafactory_images(cls, value: str) -> str:
        return validate_training_image_list(value)

    @staticmethod
    def _image_set(raw: str) -> frozenset[str]:
        return frozenset(item.strip() for item in raw.split(",") if item.strip())

    @property
    def vllm_images(self) -> frozenset[str]:
        return self._image_set(self.vllm_allowed_images)

    @property
    def vllm_runtime_image(self) -> str:
        images = [item.strip() for item in self.vllm_allowed_images.split(",") if item.strip()]
        if not images:
            raise ValueError("VLLM_ALLOWED_IMAGES 至少包含一个镜像")
        return images[0]

    @property
    def llamafactory_images(self) -> frozenset[str]:
        return self._image_set(self.llamafactory_allowed_images)

    @property
    def llamafactory_runtime_image(self) -> str:
        return self.llamafactory_allowed_images.split(",", maxsplit=1)[0].strip()

    def ensure_layout(self) -> None:
        """创建受控目录；后续所有挂载都必须位于这些根目录内。"""

        for directory in (
            self.model_root,
            self.dataset_root,
            self.checkpoint_root,
            self.training_config_root,
            self.runtime_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
