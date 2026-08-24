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
    assert 'require_production_digest "$environment" "VLLM_ALLOWED_IMAGES"' in preflight
    assert 'require_production_digest "$environment" "EVALUATION_ALLOWED_IMAGES"' in preflight
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
