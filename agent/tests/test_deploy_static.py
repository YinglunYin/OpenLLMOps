from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


def test_backend_dockerfile_installs_workspace_importer_before_backend() -> None:
    dockerfile = (DEPLOY_ROOT / "docker/backend.Dockerfile").read_text(encoding="utf-8")
    importer_copy = dockerfile.index("COPY workers/model_importer/pyproject.toml")
    importer_install = dockerfile.index('pip install "/opt/model-importer[huggingface,modelscope]"')
    backend_copy = dockerfile.index("COPY backend/pyproject.toml")
    backend_install = dockerfile.index("RUN pip install .")

    assert importer_copy < importer_install < backend_copy < backend_install
    # Alembic 的 script_location 是 backend/migrations；静态断言避免无 Docker 的开发机
    # 把一个必然在远端构建失败的旧目录名提交出去。
    assert "COPY backend/migrations ./migrations" in dockerfile
    assert (PROJECT_ROOT / "backend/migrations").is_dir()


def test_production_auth_hmac_and_proxy_settings_are_wired() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    api_environment = compose["services"]["api"]["environment"]
    agent_environment = compose["services"]["node-agent"]["environment"]
    required = {
        "ADMIN_PASSWORD_HASH",
        "SESSION_SIGNING_KEY",
        "SESSION_COOKIE_SECURE",
        "TRUSTED_PROXY_CIDRS",
        "NODE_AGENT_CLOCK_SKEW_SECONDS",
        "RECONCILER_ENABLED",
    }

    assert required <= api_environment.keys()
    assert (
        api_environment["NODE_AGENT_CLOCK_SKEW_SECONDS"] == agent_environment["NODE_AGENT_CLOCK_SKEW_SECONDS"]
    )
    assert compose["networks"]["control"]["ipam"]["config"][0]["subnet"]
    assert compose["services"]["web"]["networks"]["control"]["ipv4_address"]
    entrypoint = (DEPLOY_ROOT / "scripts/api-entrypoint.sh").read_text(encoding="utf-8")
    assert '--forwarded-allow-ips="*"' not in entrypoint
    assert '--forwarded-allow-ips="$trusted_proxy_cidrs"' in entrypoint


def test_example_argon_hash_guidance_and_cors_are_safe() -> None:
    example = (DEPLOY_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "ADMIN_PASSWORD_HASH='$argon2id$" in example
    assert "\nADMIN_PASSWORD_HASH=\n" in example
    assert "SESSION_SIGNING_KEY=" in example
    assert "CORS_ORIGINS=*" not in example
    assert "SESSION_COOKIE_SECURE=true" in example


def test_model_import_staging_and_secret_files_are_wired() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    environment = api["environment"]

    assert environment["MODEL_STAGING_ROOT"] == "/srv/openllmops/model-staging"
    assert environment["MODEL_IMPORT_COORDINATOR_ENABLED"]
    assert "HUGGINGFACE_TOKEN_FILE" in environment
    assert "MODELSCOPE_TOKEN_FILE" in environment
    assert any(
        volume.get("target") == "/run/secrets/model-sources" and volume.get("read_only") is True
        for volume in api["volumes"]
        if isinstance(volume, dict)
    )
