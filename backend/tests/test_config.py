import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_cors_origins_accepts_compose_comma_syntax(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CORS_ORIGINS", "http://one.local,http://two.local")
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["http://one.local", "http://two.local"]


def test_cors_origins_accepts_json_array(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CORS_ORIGINS", '["https://console.local"]')
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["https://console.local"]


def test_optional_model_token_file_accepts_compose_empty_value(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HUGGINGFACE_TOKEN_FILE", "")
    monkeypatch.setenv("MODELSCOPE_TOKEN_FILE", "   ")
    settings = Settings(_env_file=None)
    assert settings.huggingface_token_file is None
    assert settings.modelscope_token_file is None


def test_prometheus_url_normalization_and_validation() -> None:
    settings = Settings(_env_file=None, prometheus_url=" http://prometheus:9090/prometheus/ ")
    assert settings.prometheus_url == "http://prometheus:9090/prometheus"
    assert Settings(_env_file=None, prometheus_url="").prometheus_url is None
    with pytest.raises(ValidationError, match="不含凭证"):
        Settings(_env_file=None, prometheus_url="http://admin:secret@prometheus:9090")
    with pytest.raises(ValidationError, match="端口无效"):
        Settings(_env_file=None, prometheus_url="http://prometheus:not-a-port")


def test_evaluation_roots_require_absolute_paths() -> None:
    with pytest.raises(ValidationError, match="评测受控根目录必须使用绝对路径"):
        Settings(_env_file=None, evaluation_dataset_root="relative/evaluation-datasets")
