import time
from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from openllmops_agent.agent_contract import (
    AgentCommand,
    AgentCommandResponse,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
    HMACSigner,
    NonceReplayCache,
    canonical_json,
)
from openllmops_agent.command_service import CommandResult
from openllmops_agent.main import app

SECRET = "agent-contract-test-secret-at-least-32-bytes"


class StubProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, command: AgentCommand) -> CommandResult:
        self.calls += 1
        return CommandResult(
            status_code=200,
            response=AgentCommandResponse(
                request_id=command.request_id,
                accepted=True,
                observed_state=AgentWorkloadState.RUNNING,
                observed_at=datetime.now(UTC),
                metadata={"endpoint": "http://runtime:8000"},
            ),
        )


class BrokenProcessor:
    def execute(self, command: AgentCommand) -> CommandResult:
        raise AssertionError(f"unexpected {command.request_id}")


def status_command() -> AgentCommand:
    return AgentCommand(
        action="status",
        owner=AgentOwner(type="deployment", id=uuid4(), name="chat", generation=1),
        resources=AgentResourceRequest(gpu_ids=[0]),
        execution={},
    )


def configure_contract_state() -> tuple[HMACSigner, StubProcessor]:
    signer = HMACSigner(SECRET, max_clock_skew_seconds=30)
    processor = StubProcessor()
    app.state.hmac_signer = signer
    app.state.nonce_cache = NonceReplayCache(30)
    app.state.command_processor = processor
    return signer, processor


def test_command_endpoint_verifies_request_and_signs_canonical_response() -> None:
    signer, processor = configure_contract_state()
    command = status_command()
    body = canonical_json(command)
    headers = {
        "Content-Type": "application/json",
        **signer.sign(body, nonce="endpoint-valid-nonce"),
    }

    response = TestClient(app).post("/v1/workloads/commands", content=body, headers=headers)

    assert response.status_code == 200
    signer.verify(response.content, response.headers, now=int(time.time()))
    parsed = AgentCommandResponse.model_validate_json(response.content)
    assert response.content == canonical_json(parsed)
    assert parsed.request_id == command.request_id
    assert processor.calls == 1


def test_command_endpoint_rejects_replay_and_noncanonical_json_with_signed_errors() -> None:
    signer, processor = configure_contract_state()
    command = status_command()
    body = canonical_json(command)
    headers = signer.sign(body, nonce="endpoint-replay-nonce")
    client = TestClient(app)
    assert client.post("/v1/workloads/commands", content=body, headers=headers).status_code == 200

    replay = client.post("/v1/workloads/commands", content=body, headers=headers)
    assert replay.status_code == 401
    signer.verify(replay.content, replay.headers)
    replay_error = AgentCommandResponse.model_validate_json(replay.content)
    assert replay_error.error_code == "invalid_signature"

    noncanonical = body + b"\n"
    noncanonical_headers = signer.sign(noncanonical, nonce="endpoint-noncanonical-nonce")
    rejected = client.post(
        "/v1/workloads/commands",
        content=noncanonical,
        headers=noncanonical_headers,
    )
    assert rejected.status_code == 422
    signer.verify(rejected.content, rejected.headers)
    assert AgentCommandResponse.model_validate_json(rejected.content).error_code == "non_canonical_json"
    assert processor.calls == 1


def test_legacy_token_write_routes_are_removed() -> None:
    client = TestClient(app)
    token = {"X-Node-Agent-Token": SECRET}
    assert client.post("/v1/inference", json={}, headers=token).status_code in {
        404,
        405,
    }
    assert client.post("/v1/training", json={}, headers=token).status_code in {404, 405}
    assert client.post("/v1/workloads/demo/start", headers=token).status_code in {
        404,
        405,
    }
    assert client.post("/v1/workloads/demo/stop", json={}, headers=token).status_code in {404, 405}
    assert client.delete("/v1/workloads/demo", headers=token).status_code in {404, 405}


def test_unexpected_contract_failure_is_sanitized_and_signed() -> None:
    signer, _ = configure_contract_state()
    app.state.command_processor = BrokenProcessor()
    command = status_command()
    body = canonical_json(command)

    response = TestClient(app).post(
        "/v1/workloads/commands",
        content=body,
        headers=signer.sign(body, nonce="endpoint-internal-error-nonce"),
    )

    assert response.status_code == 500
    signer.verify(response.content, response.headers)
    error = AgentCommandResponse.model_validate_json(response.content)
    assert error.request_id == command.request_id
    assert error.error_code == "internal_error"
    assert "unexpected" not in error.message
