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
