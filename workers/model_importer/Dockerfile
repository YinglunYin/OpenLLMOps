FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/model-importer
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[huggingface,modelscope]"

USER 65532:65532
ENTRYPOINT ["python", "-m", "openllmops_model_importer"]
