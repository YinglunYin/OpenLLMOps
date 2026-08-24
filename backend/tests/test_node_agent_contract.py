import uuid
from datetime import UTC, datetime

import httpx
import pytest

from app.models.enums import LeaseOwnerType
from app.schemas.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
)
from app.services.node_agent import (
    HMACSigner,
    NodeAgentError,
    NodeAgentHTTPClient,
    SignatureVerificationError,
    canonical_json,
)

SECRET = "unit-test-shared-secret-at-least-32-bytes"


def _command() -> AgentCommand:
    return AgentCommand(
        action=AgentAction.START,
        owner=AgentOwner(
            type=LeaseOwnerType.DEPLOYMENT,
            id=uuid.uuid4(),
            name="chat",
            generation=4,
        ),
        resources=AgentResourceRequest(gpu_ids=[0, 1]),
        execution={"runner": "vllm", "service_type": "generate"},
    )


def test_hmac_rejects_tampering_expiry_and_replay() -> None:
    signer = HMACSigner(SECRET, max_clock_skew_seconds=30)
    body = canonical_json(_command())
    seen_nonces: set[str] = set()
    headers = signer.sign(body, timestamp=1_000, nonce="nonce-1")
    signer.verify(body, headers, now=1_010, seen_nonces=seen_nonces)

    with pytest.raises(SignatureVerificationError, match="nonce"):
        signer.verify(body, headers, now=1_010, seen_nonces=seen_nonces)
    tampered_headers = signer.sign(body, timestamp=1_000, nonce="nonce-2")
    with pytest.raises(SignatureVerificationError, match="不匹配"):
        signer.verify(body + b" ", tampered_headers, now=1_010)
    expired_headers = signer.sign(body, timestamp=1_000, nonce="nonce-3")
    with pytest.raises(SignatureVerificationError, match="过期"):
        signer.verify(body, expired_headers, now=1_031)


async def test_http_client_and_fake_agent_use_signed_structured_contract() -> None:
    signer = HMACSigner(SECRET)
    seen_nonces: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/workloads/commands"
        signer.verify(request.content, request.headers, seen_nonces=seen_nonces)
        command = AgentCommand.model_validate_json(request.content)
        response = AgentCommandResponse(
            request_id=command.request_id,
            accepted=True,
            observed_state=AgentWorkloadState.RUNNING,
            observed_at=datetime.now(UTC),
            metadata={
                "endpoint": "http://127.0.0.1:18000/v1",
                "service_type": command.execution["service_type"],
            },
        )
        body = canonical_json(response)
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json", **signer.sign(body)},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://agent") as http_client:
        client = NodeAgentHTTPClient(
            "http://agent",
            SECRET,
            client=http_client,
        )
        result = await client.execute(_command())

    assert result.accepted
    assert result.observed_state == AgentWorkloadState.RUNNING
    assert len(seen_nonces) == 1


async def test_http_client_distinguishes_explicit_4xx_from_indeterminate_failures() -> None:
    signer = HMACSigner(SECRET)
    command = _command()

    def signed_response(
        request: httpx.Request,
        *,
        status_code: int,
        accepted: bool,
        request_id: uuid.UUID | None = None,
    ) -> httpx.Response:
        response = AgentCommandResponse(
            request_id=request_id or command.request_id,
            accepted=accepted,
            observed_state=AgentWorkloadState.FAILED,
            observed_at=datetime.now(UTC),
            message="节点拒绝",
            error_code="invalid_workload",
        )
        body = canonical_json(response)
        return httpx.Response(status_code, content=body, headers=signer.sign(body), request=request)

    async def explicit_rejection(request: httpx.Request) -> httpx.Response:
        return signed_response(request, status_code=422, accepted=False)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(explicit_rejection),
        base_url="http://agent",
    ) as http_client:
        client = NodeAgentHTTPClient("http://agent", SECRET, client=http_client)
        rejection = await client.execute(command)
    assert not rejection.accepted and rejection.error_code == "invalid_workload"

    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("响应超时", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(timeout),
        base_url="http://agent",
    ) as http_client:
        client = NodeAgentHTTPClient("http://agent", SECRET, client=http_client)
        with pytest.raises(NodeAgentError, match="调用失败"):
            await client.execute(command)

    async def server_error(request: httpx.Request) -> httpx.Response:
        return signed_response(request, status_code=503, accepted=False)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server_error),
        base_url="http://agent",
    ) as http_client:
        client = NodeAgentHTTPClient("http://agent", SECRET, client=http_client)
        with pytest.raises(NodeAgentError, match="503"):
            await client.execute(command)

    async def protocol_error(request: httpx.Request) -> httpx.Response:
        return signed_response(
            request,
            status_code=200,
            accepted=True,
            request_id=uuid.uuid4(),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(protocol_error),
        base_url="http://agent",
    ) as http_client:
        client = NodeAgentHTTPClient("http://agent", SECRET, client=http_client)
        with pytest.raises(NodeAgentError, match="request_id"):
            await client.execute(command)
