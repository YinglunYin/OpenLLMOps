FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src

COPY workers/training_config/pyproject.toml ./workers/training_config/pyproject.toml
COPY workers/training_config/src ./workers/training_config/src
COPY workers/training_runtime/pyproject.toml ./workers/training_runtime/pyproject.toml
COPY workers/training_runtime/src ./workers/training_runtime/src
COPY agent/pyproject.toml ./agent/pyproject.toml
COPY agent/openllmops_agent ./agent/openllmops_agent
RUN pip install ./workers/training_config ./workers/training_runtime ./agent \
    && groupadd --gid 1000 openllmops \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin openllmops

WORKDIR /app

USER 1000:1000
EXPOSE 9000

# 必须保持单 worker：GPU 冲突检查与容器创建通过进程内锁形成一个原子区段。
CMD ["uvicorn", "openllmops_agent.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
