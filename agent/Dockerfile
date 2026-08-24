FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY openllmops_agent ./openllmops_agent
RUN pip install . \
    && groupadd --gid 1000 openllmops \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin openllmops

USER 1000:1000
EXPOSE 9000

# 必须保持单 worker：GPU 冲突检查与容器创建通过进程内锁形成一个原子区段。
CMD ["uvicorn", "openllmops_agent.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
