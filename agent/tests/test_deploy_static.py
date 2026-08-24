import json
import os
import subprocess
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, SequenceNode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


def _assert_no_duplicate_yaml_keys(node: Node) -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            key = key_node.value
            assert key not in seen, f"Compose 包含重复 YAML key: {key}"
            seen.add(key)
            _assert_no_duplicate_yaml_keys(value_node)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            _assert_no_duplicate_yaml_keys(item)


def test_compose_contains_no_duplicate_mapping_keys() -> None:
    root = yaml.compose((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert root is not None
    _assert_no_duplicate_yaml_keys(root)


def test_backend_dockerfile_installs_workspace_importer_before_backend() -> None:
    dockerfile = (DEPLOY_ROOT / "docker/backend.Dockerfile").read_text(encoding="utf-8")
    importer_copy = dockerfile.index("COPY workers/model_importer/pyproject.toml")
    importer_install = dockerfile.index('pip install "/opt/model-importer[huggingface,modelscope]"')
    evaluation_copy = dockerfile.index("COPY evaluation/pyproject.toml")
    evaluation_install = dockerfile.index("pip install /opt/evaluation")
    backend_copy = dockerfile.index("COPY backend/pyproject.toml")
    backend_install = dockerfile.index("RUN pip install .")

    assert (
        importer_copy
        < importer_install
        < evaluation_copy
        < evaluation_install
        < backend_copy
        < backend_install
    )
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
    assert 'CORS_ORIGINS=["https://openllmops.local"]' in example
    assert "SESSION_COOKIE_SECURE=true" in example
    assert "CUDA_VARIANT=cu130" in example


def test_model_import_staging_and_secret_files_are_wired() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    environment = api["environment"]

    assert environment["MODEL_STAGING_ROOT"] == "/srv/openllmops/model-staging"
    assert environment["TMPDIR"] == "/srv/openllmops/upload-tmp"
    assert environment["MODEL_IMPORT_COORDINATOR_ENABLED"]
    assert "HUGGINGFACE_TOKEN_FILE" in environment
    assert "MODELSCOPE_TOKEN_FILE" in environment
    assert any(
        volume.get("target") == "/run/secrets/model-sources" and volume.get("read_only") is True
        for volume in api["volumes"]
        if isinstance(volume, dict)
    )
    storage_check = (DEPLOY_ROOT / "scripts/check-storage-permissions.py").read_text(encoding="utf-8")
    assert '"upload-tmp"' in storage_check


def test_evaluation_runtime_and_controlled_paths_are_wired() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    api_environment = compose["services"]["api"]["environment"]
    agent_environment = compose["services"]["node-agent"]["environment"]
    runtime_image = compose["services"]["evaluation-runtime-image"]

    assert api_environment["EVALUATION_DATASET_ROOT"] == "/srv/openllmops/evaluation-datasets"
    assert api_environment["EVALUATION_OUTPUT_ROOT"] == "/srv/openllmops/evaluation-output"
    assert api_environment["NODE_AGENT_RUNTIME_ROOT"] == "/srv/openllmops/runtime"
    assert api_environment["PROMETHEUS_TIMEOUT_SECONDS"]
    assert api_environment["MODEL_IMPORT_CLAIM_TIMEOUT_SECONDS"]
    assert agent_environment["EVALUATION_DATASET_ROOT"] == api_environment["EVALUATION_DATASET_ROOT"]
    assert agent_environment["EVALUATION_OUTPUT_ROOT"] == api_environment["EVALUATION_OUTPUT_ROOT"]
    assert "v0.27.1" in agent_environment["VLLM_ALLOWED_IMAGES"]
    assert "v0.10.2" not in agent_environment["VLLM_ALLOWED_IMAGES"]
    assert runtime_image["profiles"] == ["runtime-image"]
    assert runtime_image["build"]["context"] == "../evaluation"
    assert runtime_image["network_mode"] == "none"
    dockerfile = (PROJECT_ROOT / "evaluation/Dockerfile").read_text(encoding="utf-8")
    assert "vllm.__version__ == '0.27.1'" in dockerfile
    assert 'com.openllmops.security.trust-remote-code="disabled"' in dockerfile


def test_training_wrapper_fixed_build_context_and_preflight_guards_are_wired() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    agent_build = compose["services"]["node-agent"]["build"]
    training_image = compose["services"]["llamafactory-secure-image"]
    dockerfile = (DEPLOY_ROOT / "docker/llamafactory-secure.Dockerfile").read_text(encoding="utf-8")
    preflight = (DEPLOY_ROOT / "scripts/preflight.sh").read_text(encoding="utf-8")

    assert agent_build == {"context": "..", "dockerfile": "agent/Dockerfile"}
    assert training_image["build"]["context"] == ".."
    assert "COPY workers/training_runtime" in dockerfile
    assert 'ENTRYPOINT ["openllmops-training-runtime"]' in dockerfile
    assert 'com.openllmops.runner="training-wrapper-v1"' in dockerfile
    assert 'com.openllmops.artifacts="safetensors-validated-v1"' in dockerfile
    agent_ignore = (PROJECT_ROOT / "agent/Dockerfile.dockerignore").read_text(encoding="utf-8")
    training_ignore = (DEPLOY_ROOT / "docker/llamafactory-secure.Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    )
    assert agent_ignore.startswith("**\n") and "!workers/training_runtime/src/**" in agent_ignore
    assert training_ignore.startswith("**\n") and "!workers/training_config/src/**" in training_ignore
    assert 'require_production_digest "$environment" "LLAMAFACTORY_ALLOWED_IMAGES"' in preflight
    assert 'require_production_digest "$environment" "VLLM_ALLOWED_IMAGES"' in preflight
    assert 'require_production_digest "$environment" "EVALUATION_ALLOWED_IMAGES"' in preflight
    assert 'require_production_digest "$environment" "EVALUATION_VLLM_BASE_IMAGE"' in preflight
    assert "com.openllmops.security.ghsa-mwc7-mf87-v3mf" in preflight
    assert "com.openllmops.security.trust-remote-code" in preflight
    assert "minimum_driver_version=580.95.05" in preflight
    assert "minimum_driver_version=575.57.08" in preflight
    assert "--query-gpu=index,driver_version" in preflight
    assert 'validate_image_cuda_version "$vllm_image"' in preflight
    assert 'validate_image_cuda_version "$evaluation_vllm_base_image"' in preflight
    assert 'validate_image_cuda_version "$evaluation_image"' in preflight
    assert preflight.index("validate-secrets.py") < preflight.index("docker info")
    assert preflight.index("validate-web-config.py") < preflight.index("docker info")
    assert preflight.index("reject-compose-overrides.py") < preflight.index("docker info")
    permission_check = DEPLOY_ROOT / "scripts/check-storage-permissions.py"
    permission_source = permission_check.read_text(encoding="utf-8")
    assert "os.setuid(uid)" in permission_source and "os.access(target" in permission_source
    subprocess.run(
        ["sh", "-n", str(DEPLOY_ROOT / "scripts/preflight.sh")],
        check=True,
    )
    subprocess.run(
        ["sh", "-n", str(DEPLOY_ROOT / "scripts/image-reference-policy.sh")],
        check=True,
    )
    subprocess.run(
        ["sh", "-n", str(DEPLOY_ROOT / "nginx/10-ensure-tls.sh")],
        check=True,
    )


def test_preflight_checks_every_gpu_driver_against_explicit_cuda_variant(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/bin/sh\necho OPENLLMOPS_TEST_DOCKER_REACHED >&2\nexit 91\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    nvidia_smi = fake_bin / "nvidia-smi"
    tls_dir = tmp_path / "tls"
    tls_dir.mkdir(mode=0o700)
    env_file = tmp_path / ".env"

    base_lines = (
        "POSTGRES_PASSWORD=database-" + "d" * 32,
        "SESSION_SIGNING_KEY=session-" + "s" * 40,
        "ADMIN_API_KEY=admin-" + "a" * 40,
        "API_KEY_PEPPER=pepper-" + "p" * 40,
        "NODE_AGENT_TOKEN=agent-" + "n" * 40,
        "TLS_COMMON_NAME=openllmops.local",
        'CORS_ORIGINS=["https://openllmops.local"]',
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=true",
        "GPU_COUNT=2",
    )
    clean_environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    def run(variant: str, rows: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        # 测试数据为固定数字/逗号，逐项加引号避免测试 shell 再解析。
        output_arguments = " ".join(f"'{row}'" for row in rows)
        nvidia_smi.write_text(
            f"#!/bin/sh\nprintf '%s\\n' {output_arguments}\n",
            encoding="utf-8",
        )
        nvidia_smi.chmod(0o755)
        env_file.write_text(
            "\n".join((*base_lines, f"CUDA_VARIANT={variant}")) + "\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        return subprocess.run(
            ["sh", str(DEPLOY_ROOT / "scripts/preflight.sh"), str(env_file)],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )

    # 边界值通过 GPU 阶段后才会命中故意失败的 Docker 桩。
    cuda_130_boundary = run("cu130", ("0, 580.95.05", "1, 580.95.05"))
    assert cuda_130_boundary.returncode == 91
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" in cuda_130_boundary.stderr

    cuda_129_boundary = run("cu129", ("0, 575.57.08", "1, 575.57.08"))
    assert cuda_129_boundary.returncode == 91
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" in cuda_129_boundary.stderr

    # 第一张卡合格不能掩盖第二张卡过旧。
    stale_second_gpu = run("cu130", ("0, 580.95.05", "1, 580.95.04"))
    assert stale_second_gpu.returncode != 0
    assert "GPU 1" in stale_second_gpu.stderr and "580.95.05" in stale_second_gpu.stderr
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" not in stale_second_gpu.stderr

    stale_cu129 = run("cu129", ("0, 575.57.08", "1, 575.57.07"))
    assert stale_cu129.returncode != 0 and "GPU 1" in stale_cu129.stderr
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" not in stale_cu129.stderr

    malformed = run("cu130", ("0, 580.95.05", "1, 580.95.beta"))
    assert malformed.returncode != 0 and "格式异常" in malformed.stderr
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" not in malformed.stderr

    internal_whitespace = run("cu130", ("0, 580.95.05", "1, 580. 95.05"))
    assert internal_whitespace.returncode != 0 and "格式异常" in internal_whitespace.stderr
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" not in internal_whitespace.stderr

    unsupported_variant = run("auto", ("0, 999.99.99", "1, 999.99.99"))
    assert unsupported_variant.returncode != 0 and "cu130 或 cu129" in unsupported_variant.stderr
    assert "OPENLLMOPS_TEST_DOCKER_REACHED" not in unsupported_variant.stderr


def test_web_config_validator_enforces_origin_and_tls_contract(tmp_path: Path) -> None:
    validator = DEPLOY_ROOT / "scripts/validate-web-config.py"
    tls_dir = tmp_path / "tls"
    tls_dir.mkdir(mode=0o755)
    env_file = tmp_path / ".env"

    def validate(*lines: str) -> subprocess.CompletedProcess[str]:
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.run(
            ["python3", str(validator), str(env_file)],
            check=False,
            capture_output=True,
            text=True,
        )

    mismatch = validate(
        "TLS_COMMON_NAME=openllmops.local",
        'CORS_ORIGINS=["https://openllmops.example.internal"]',
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=true",
    )
    assert mismatch.returncode != 0 and "必须包含" in mismatch.stderr

    matched = validate(
        "TLS_COMMON_NAME=openllmops.local",
        'CORS_ORIGINS=["https://openllmops.local"]',
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=true",
    )
    assert matched.returncode == 0, matched.stderr

    cert = tls_dir / "tls.crt"
    key = tls_dir / "tls.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=console.internal",
            "-addext",
            "subjectAltName=DNS:console.internal",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    cert.chmod(0o644)
    key.chmod(0o600)
    provided = validate(
        "TLS_COMMON_NAME=console.internal",
        'CORS_ORIGINS=["https://console.internal"]',
        "HTTPS_PORT=443",
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=false",
    )
    assert provided.returncode == 0, provided.stderr

    key.chmod(0o644)
    unsafe_key = validate(
        "TLS_COMMON_NAME=console.internal",
        'CORS_ORIGINS=["https://console.internal"]',
        "HTTPS_PORT=443",
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=false",
    )
    assert unsafe_key.returncode != 0 and "chmod 0600" in unsafe_key.stderr

    subprocess.run(
        [
            "openssl",
            "genrsa",
            "-aes256",
            "-passout",
            "pass:test-only-password",
            "-out",
            str(key),
            "2048",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    encrypted_key = validate(
        "TLS_COMMON_NAME=console.internal",
        'CORS_ORIGINS=["https://console.internal"]',
        f"TLS_DIR={tls_dir}",
        "TLS_AUTO_GENERATE=false",
    )
    assert encrypted_key.returncode != 0 and "OpenSSL" in encrypted_key.stderr

    uppercase_boolean = validate(
        "TLS_COMMON_NAME=openllmops.local",
        'CORS_ORIGINS=["https://openllmops.local"]',
        f"TLS_DIR={tmp_path / 'empty-tls'}",
        "TLS_AUTO_GENERATE=TRUE",
    )
    assert uppercase_boolean.returncode != 0 and "true 或 false" in uppercase_boolean.stderr


def test_web_service_has_only_tls_mount_and_required_dac_capability() -> None:
    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    web = compose["services"]["web"]
    assert "DAC_OVERRIDE" in web["cap_add"]
    assert "KILL" in web["cap_add"]
    assert len(web["volumes"]) == 1
    assert "/etc/nginx/tls" in web["volumes"][0]
    nginx = (DEPLOY_ROOT / "nginx/nginx.conf").read_text(encoding="utf-8")
    assert "client_max_body_size 16m;" in nginx
    upload_location = nginx.index("location = /api/v1/datasets/upload")
    upload_limit = nginx.index("client_max_body_size 5121m;", upload_location)
    generic_location = nginx.index("location ~ ^/(api|v1)(/|$)")
    assert upload_location < upload_limit < generic_location


def test_deploy_secret_validator_rejects_public_placeholders_and_duplicates(
    tmp_path: Path,
) -> None:
    validator = DEPLOY_ROOT / "scripts/validate-secrets.py"
    example = subprocess.run(
        ["python3", str(validator), str(DEPLOY_ROOT / ".env.example")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert example.returncode != 0
    assert "ADMIN_API_KEY" in example.stderr or "POSTGRES_PASSWORD" in example.stderr

    valid_values = {
        "POSTGRES_PASSWORD": "database-" + "d" * 32,
        "SESSION_SIGNING_KEY": "session-" + "s" * 40,
        "ADMIN_API_KEY": "admin-" + "a" * 40,
        "API_KEY_PEPPER": "pepper-" + "p" * 40,
        "NODE_AGENT_TOKEN": "agent-" + "n" * 40,
    }
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in valid_values.items()) + "\n",
        encoding="utf-8",
    )
    valid = subprocess.run(
        ["python3", str(validator), str(env_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr

    valid_values["NODE_AGENT_TOKEN"] = valid_values["ADMIN_API_KEY"]
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in valid_values.items()) + "\n",
        encoding="utf-8",
    )
    duplicate = subprocess.run(
        ["python3", str(validator), str(env_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0 and "相同密钥" in duplicate.stderr


def test_compose_shell_overrides_are_rejected_without_exposing_values(tmp_path: Path) -> None:
    validator = DEPLOY_ROOT / "scripts/reject-compose-overrides.py"
    secret_value = "do-not-print-this-secret"
    environment = os.environ.copy()
    environment.update({"ENVIRONMENT": "development", "AUTH_ENABLED": "false"})
    env_file = tmp_path / ".env"
    env_file.write_text(f"ADMIN_API_KEY={secret_value}\n", encoding="utf-8")
    env_file.chmod(0o600)

    rejected = subprocess.run(
        ["python3", str(validator), str(env_file), str(DEPLOY_ROOT / "compose.yaml")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert rejected.returncode != 0
    assert "ENVIRONMENT" in rejected.stderr and "AUTH_ENABLED" in rejected.stderr
    assert secret_value not in rejected.stderr

    environment.pop("ENVIRONMENT")
    environment.pop("AUTH_ENABLED")
    env_file.chmod(0o644)
    unsafe_mode = subprocess.run(
        ["python3", str(validator), str(env_file), str(DEPLOY_ROOT / "compose.yaml")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert unsafe_mode.returncode != 0 and "chmod 0600" in unsafe_mode.stderr


def test_rendered_compose_matches_preflighted_critical_values(tmp_path: Path) -> None:
    validator = DEPLOY_ROOT / "scripts/validate-rendered-config.py"
    tls_dir = tmp_path / "tls"
    storage = tmp_path / "storage"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "TLS_COMMON_NAME=openllmops.local",
                "TLS_AUTO_GENERATE=true",
                f"TLS_DIR={tls_dir}",
                'CORS_ORIGINS=["https://openllmops.local"]',
                f"OPENLLMOPS_STORAGE_ROOT={storage}",
                "APP_UID=1000",
                "APP_GID=1000",
                "HTTPS_PORT=443",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = {
        "services": {
            "web": {
                "environment": {
                    "TLS_COMMON_NAME": "openllmops.local",
                    "TLS_AUTO_GENERATE": "true",
                    "TLS_OWNER_UID": "1000",
                    "TLS_OWNER_GID": "1000",
                },
                "volumes": [{"type": "bind", "source": str(tls_dir), "target": "/etc/nginx/tls"}],
                "ports": [{"target": 8443, "published": "443"}],
            },
            "api": {
                "environment": {"CORS_ORIGINS": '["https://openllmops.local"]'},
                "user": "1000:1000",
                "volumes": [{"type": "bind", "source": str(storage), "target": "/srv/openllmops"}],
            },
            "node-agent": {
                "user": "1000:1000",
                "volumes": [{"type": "bind", "source": str(storage), "target": "/srv/openllmops"}],
            },
        }
    }

    def validate(value: dict[str, object]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(validator), str(env_file)],
            input=json.dumps(value),
            check=False,
            capture_output=True,
            text=True,
        )

    assert validate(rendered).returncode == 0
    rendered["services"]["web"]["environment"]["TLS_COMMON_NAME"] = "overridden.internal"  # type: ignore[index]
    rejected = validate(rendered)
    assert rejected.returncode != 0 and "TLS_COMMON_NAME" in rejected.stderr


def test_production_digest_policy_is_exact_and_development_allows_fixed_tags() -> None:
    policy = DEPLOY_ROOT / "scripts/image-reference-policy.sh"

    def validate(environment: str, reference: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "sh",
                "-c",
                '. "$1"; require_production_digest "$2" TEST "$3"',
                "openllmops-image-policy-test",
                str(policy),
                environment,
                reference,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    valid = "registry.internal/openllmops/runtime@sha256:" + "a" * 64
    assert validate("production", valid).returncode == 0
    assert validate("production", "repo@sha256:ab").returncode != 0
    assert validate("production", "repo@sha256:" + "g" * 64).returncode != 0
    assert validate("production", "@sha256:" + "a" * 64).returncode != 0
    assert validate("development", "vllm/vllm-openai:v0.27.1").returncode == 0
    hardened_training_tag = "openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1"
    assert validate("production", hardened_training_tag).returncode != 0
    assert validate("development", hardened_training_tag).returncode == 0


def test_preflight_dotenv_reader_normalizes_quotes_comments_and_rejects_indirection(
    tmp_path: Path,
) -> None:
    reader = DEPLOY_ROOT / "scripts/read-env-value.py"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENVIRONMENT='production'\nGPU_COUNT=4 # 正式机\nADMIN_PASSWORD_HASH='$argon2id$literal'\n",
        encoding="utf-8",
    )

    def read(key: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(reader), str(env_file), key],
            check=False,
            capture_output=True,
            text=True,
        )

    assert read("ENVIRONMENT").stdout.strip() == "production"
    assert read("GPU_COUNT").stdout.strip() == "4"
    assert read("ADMIN_PASSWORD_HASH").stdout.strip() == "$argon2id$literal"

    env_file.write_text("ENVIRONMENT=${OPENLLMOPS_MODE:-development}\n", encoding="utf-8")
    rejected = read("ENVIRONMENT")
    assert rejected.returncode != 0 and "禁止" in rejected.stderr


def test_storage_permission_check_runs_as_configured_non_root_identity(tmp_path: Path) -> None:
    checker = DEPLOY_ROOT / "scripts/check-storage-permissions.py"
    children = (
        "models",
        "inbox",
        "model-staging",
        "datasets",
        "evaluation-datasets",
        "evaluation-output",
        "checkpoints",
        "training-configs",
        "runtime",
        "upload-tmp",
    )
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        uid = gid = 65534
        os.chown(tmp_path, uid, gid)
    for child in children:
        target = tmp_path / child
        target.mkdir(mode=0o700)
        if os.getuid() == 0:
            os.chown(target, uid, gid)

    passed = subprocess.run(
        ["python3", str(checker), str(tmp_path), str(uid), str(gid)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stderr

    blocked = tmp_path / "checkpoints"
    blocked.chmod(0o500)
    failed = subprocess.run(
        ["python3", str(checker), str(tmp_path), str(uid), str(gid)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "不可读写进入" in failed.stderr
