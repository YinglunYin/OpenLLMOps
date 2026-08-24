ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.27.1
FROM ${VLLM_BASE_IMAGE}

LABEL com.openllmops.runner="evaluation-pair-v1" \
      com.openllmops.security.trust-remote-code="disabled" \
      com.openllmops.base.vllm="v0.27.1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/openllmops-evaluation
COPY pyproject.toml ./
COPY src ./src
# 即使管理员覆盖了基础镜像引用，版本漂移也会在构建时立即失败。
RUN python -c "import vllm; assert vllm.__version__ == '0.27.1', vllm.__version__" \
    && pip install --no-cache-dir .

# 动态任务会覆盖参数；镜像默认只展示帮助，避免误启动无配置评测。
ENTRYPOINT ["openllmops-eval"]
CMD ["--help"]
