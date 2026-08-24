ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.27.1
FROM ${VLLM_BASE_IMAGE}
ARG UV_VERSION=0.12.1

LABEL com.openllmops.runner="evaluation-pair-v1" \
      com.openllmops.security.trust-remote-code="disabled" \
      com.openllmops.base.vllm="v0.27.1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/openllmops-evaluation
COPY pyproject.toml ./
COPY uv.lock ./
COPY src ./src
# 即使管理员覆盖了基础镜像引用，版本漂移也会在构建时立即失败；运行依赖
# 严格由 uv.lock 解析，避免镜像重建时静默获得另一组 httpx/anyio 版本。
RUN python -c "import vllm; assert vllm.__version__ == '0.27.1', vllm.__version__" \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && uv sync --frozen --no-dev

ENV PATH="/opt/openllmops-evaluation/.venv/bin:${PATH}"

# 动态任务会覆盖参数；镜像默认只展示帮助，避免误启动无配置评测。
ENTRYPOINT ["openllmops-eval"]
CMD ["--help"]
