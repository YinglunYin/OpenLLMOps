FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/openllmops-evaluation
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# 动态任务会覆盖参数；镜像默认只展示帮助，避免误启动无配置评测。
ENTRYPOINT ["openllmops-eval"]
CMD ["--help"]
