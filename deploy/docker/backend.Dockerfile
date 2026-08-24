FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 构建工具只保留在单层中，降低最终镜像体积和攻击面。
RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# backend 的 pyproject 声明了本地 model-importer 依赖；先安装工作区包及其在线源 extras，
# 后续安装 backend 时 pip 会按已安装的同名版本解析，不会错误访问公共索引查找私有包。
COPY workers/model_importer/pyproject.toml /opt/model-importer/pyproject.toml
COPY workers/model_importer/src /opt/model-importer/src
RUN pip install "/opt/model-importer[huggingface,modelscope]"

COPY backend/pyproject.toml ./pyproject.toml
COPY backend/README.md ./README.md
COPY backend/alembic.ini ./alembic.ini
COPY backend/app ./app
COPY backend/migrations ./migrations
RUN pip install .

COPY deploy/scripts/api-entrypoint.sh /usr/local/bin/api-entrypoint
RUN chmod 0555 /usr/local/bin/api-entrypoint

EXPOSE 8000
ENTRYPOINT ["api-entrypoint"]
