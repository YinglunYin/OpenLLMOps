"""在线模型仓库下载适配器；SDK 仅在对应来源被使用时加载。"""

from __future__ import annotations

from pathlib import Path


class DownloaderUnavailableError(RuntimeError):
    """运行镜像未安装对应在线来源依赖。"""


def download_huggingface(
    repository: str, revision: str | None, destination: Path, token: str | None
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DownloaderUnavailableError("执行器未安装 huggingface 可选依赖") from exc

    # allow_patterns 明确排除仓库中的 Python 代码和 pickle 权重，下载后仍会全量复检。
    snapshot_download(
        repo_id=repository,
        revision=revision,
        token=token,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.tiktoken",
            "tokenizer.*",
            "vocab.*",
            "merges.txt",
        ],
    )


def download_modelscope(
    repository: str, revision: str | None, destination: Path, token: str | None
) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise DownloaderUnavailableError("执行器未安装 modelscope 可选依赖") from exc

    # ModelScope SDK 不同小版本对 token 参数的支持不一致，凭证由其标准环境变量读取。
    if token:
        raise ValueError("ModelScope 凭证请通过容器 secret 映射到 SDK 标准环境变量")
    snapshot_download(
        model_id=repository,
        revision=revision,
        local_dir=str(destination),
    )

