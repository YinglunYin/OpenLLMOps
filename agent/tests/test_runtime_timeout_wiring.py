from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


def test_inference_timeout_controls_are_exposed_to_node_agent() -> None:
    """避免代码支持超时、但 Compose 丢失配置，导致环境无法按模型规模调节窗口。"""

    compose = yaml.safe_load((DEPLOY_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    environment = compose["services"]["node-agent"]["environment"]
    expected = {
        "INFERENCE_STARTUP_TIMEOUT_SECONDS",
        "INFERENCE_UNHEALTHY_TIMEOUT_SECONDS",
        "INFERENCE_FAILURE_STOP_TIMEOUT_SECONDS",
    }

    assert expected <= environment.keys()

    example = (DEPLOY_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "INFERENCE_STARTUP_TIMEOUT_SECONDS=1800" in example
    assert "INFERENCE_UNHEALTHY_TIMEOUT_SECONDS=60" in example
    assert "INFERENCE_FAILURE_STOP_TIMEOUT_SECONDS=30" in example
