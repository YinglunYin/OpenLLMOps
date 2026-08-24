import time
from datetime import UTC, datetime
from uuid import UUID

import pytest

from openllmops_agent.agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentOwner,
    AgentResourceRequest,
    AgentWorkloadState,
    HMACSigner,
    NonceReplayCache,
    SignatureVerificationError,
    canonical_json,
)

SECRET = "unit-test-shared-secret-at-least-32-bytes"
REQUEST_BODY = (
    '{"action":"start","contract_version":"1","execution":{"model_path":"/srv/'
    'openllmops/models/模型","runner":"vllm","service_type":"generate"},"owner":'
    '{"generation":4,"id":"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","name":"中文模型",'
    '"type":"deployment"},"request_id":"11111111-2222-3333-4444-555555555555",'
    '"resources":{"gpu_ids":[0,1]}}'
).encode()
REQUEST_SIGNATURE = "v1=b9bb1090d49483a030a8ab71b137ef4d68af3ee299a012c459bbc5a7eeb038d6"
RESPONSE_BODY = (
    '{"accepted":true,"contract_version":"1","error_code":null,"message":null,'
    '"metadata":{"endpoint":"http://模型:8000","port":8000,"service_type":"generate"},'
    '"observed_at":"2033-05-18T03:33:20Z","observed_state":"running",'
    '"request_id":"11111111-2222-3333-4444-555555555555"}'
).encode()
RESPONSE_SIGNATURE = "v1=b6afc81f0ac1426ed779d4831ba4859fb3da70051c414784943ae93ed191a650"


def test_cross_stack_fixed_request_and_response_vectors() -> None:
    command = AgentCommand(
        contract_version="1",
        request_id=UUID("11111111-2222-3333-4444-555555555555"),
        action=AgentAction.START,
        owner=AgentOwner(
            type="deployment",
            id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            name="中文模型",
            generation=4,
        ),
        resources=AgentResourceRequest(gpu_ids=[0, 1]),
        execution={
            "model_path": "/srv/openllmops/models/模型",
            "runner": "vllm",
            "service_type": "generate",
        },
    )
    assert canonical_json(command) == REQUEST_BODY
    request_headers = HMACSigner(SECRET).sign(
        REQUEST_BODY,
        timestamp=2_000_000_000,
        nonce="cross-stack-fixed-nonce",
    )
    assert request_headers["X-OpenLLMOps-Signature"] == REQUEST_SIGNATURE

    response = AgentCommandResponse(
        request_id=command.request_id,
        accepted=True,
        observed_state=AgentWorkloadState.RUNNING,
        observed_at=datetime(2033, 5, 18, 3, 33, 20, tzinfo=UTC),
        metadata={
            "endpoint": "http://模型:8000",
            "port": 8000,
            "service_type": "generate",
        },
    )
    assert canonical_json(response) == RESPONSE_BODY
    response_headers = HMACSigner(SECRET).sign(
        RESPONSE_BODY,
        timestamp=2_000_000_001,
        nonce="cross-stack-response-nonce",
    )
    assert response_headers["X-OpenLLMOps-Signature"] == RESPONSE_SIGNATURE


def test_hmac_rejects_tampering_expiry_and_nonce_replay() -> None:
    signer = HMACSigner(SECRET, max_clock_skew_seconds=30)
    replay_cache = NonceReplayCache(max_clock_skew_seconds=30)
    now = int(time.time())
    headers = signer.sign(REQUEST_BODY, timestamp=now, nonce="nonce-fixed-0001")
    replay_cache.verify(signer, REQUEST_BODY, headers, now=now)

    with pytest.raises(SignatureVerificationError, match="nonce"):
        replay_cache.verify(signer, REQUEST_BODY, headers, now=now)
    tampered = signer.sign(REQUEST_BODY, timestamp=now, nonce="nonce-fixed-0002")
    with pytest.raises(SignatureVerificationError, match="不匹配"):
        replay_cache.verify(signer, REQUEST_BODY + b" ", tampered, now=now)
    expired = signer.sign(REQUEST_BODY, timestamp=now - 31, nonce="nonce-fixed-0003")
    with pytest.raises(SignatureVerificationError, match="过期"):
        replay_cache.verify(signer, REQUEST_BODY, expired, now=now)
