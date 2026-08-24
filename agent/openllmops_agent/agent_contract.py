from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Mapping, MutableSet
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

SIGNATURE_HEADER = "X-OpenLLMOps-Signature"
TIMESTAMP_HEADER = "X-OpenLLMOps-Timestamp"
NONCE_HEADER = "X-OpenLLMOps-Nonce"
SIGNATURE_VERSION = "v1"


class AgentAction(StrEnum):
    START = "start"
    STOP = "stop"
    STATUS = "status"


class AgentWorkloadState(StrEnum):
    ABSENT = "absent"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentOwner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["deployment", "training", "evaluation"]
    id: UUID
    name: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=1)


class AgentResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_ids: list[int] = Field(min_length=1)


class AgentCommand(BaseModel):
    """与 FastAPI 控制面的 AgentCommand 保持逐字段兼容。"""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1"] = "1"
    request_id: UUID = Field(default_factory=uuid4)
    action: AgentAction
    owner: AgentOwner
    resources: AgentResourceRequest
    execution: dict[str, Any] = Field(default_factory=dict)


class AgentCommandResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1"] = "1"
    request_id: UUID
    accepted: bool
    observed_state: AgentWorkloadState
    observed_at: datetime
    message: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignatureVerificationError(ValueError):
    pass


def canonical_json(value: BaseModel | Mapping[str, Any]) -> bytes:
    """生成与 backend.services.node_agent.canonical_json 相同的 UTF-8 字节。"""

    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class HMACSigner:
    def __init__(self, secret: str | bytes, max_clock_skew_seconds: int = 30) -> None:
        secret_bytes = secret.encode() if isinstance(secret, str) else secret
        if len(secret_bytes) < 16:
            raise ValueError("node-agent HMAC 共享密钥至少需要 16 字节")
        self._secret = secret_bytes
        self._max_clock_skew_seconds = max_clock_skew_seconds

    @staticmethod
    def _signing_input(body: bytes, timestamp: str, nonce: str) -> bytes:
        return b"\n".join((SIGNATURE_VERSION.encode(), timestamp.encode(), nonce.encode(), body))

    def sign(
        self,
        body: bytes,
        *,
        timestamp: int | None = None,
        nonce: str | None = None,
    ) -> dict[str, str]:
        timestamp_value = str(int(time.time()) if timestamp is None else timestamp)
        nonce_value = nonce or secrets.token_urlsafe(18)
        digest = hmac.new(
            self._secret,
            self._signing_input(body, timestamp_value, nonce_value),
            hashlib.sha256,
        ).hexdigest()
        return {
            TIMESTAMP_HEADER: timestamp_value,
            NONCE_HEADER: nonce_value,
            SIGNATURE_HEADER: f"{SIGNATURE_VERSION}={digest}",
        }

    def verify(
        self,
        body: bytes,
        headers: Mapping[str, str],
        *,
        now: int | None = None,
        seen_nonces: MutableSet[str] | None = None,
    ) -> None:
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        timestamp = normalized_headers.get(TIMESTAMP_HEADER.lower())
        nonce = normalized_headers.get(NONCE_HEADER.lower())
        supplied_signature = normalized_headers.get(SIGNATURE_HEADER.lower())
        if not timestamp or not nonce or not supplied_signature:
            raise SignatureVerificationError("node-agent HMAC 请求头不完整")
        try:
            signed_at = int(timestamp)
        except ValueError as exc:
            raise SignatureVerificationError("node-agent HMAC 时间戳无效") from exc
        current_time = int(time.time()) if now is None else now
        if abs(current_time - signed_at) > self._max_clock_skew_seconds:
            raise SignatureVerificationError("node-agent HMAC 签名已过期")
        if seen_nonces is not None and nonce in seen_nonces:
            raise SignatureVerificationError("node-agent HMAC nonce 已被使用")

        expected_digest = hmac.new(
            self._secret,
            self._signing_input(body, timestamp, nonce),
            hashlib.sha256,
        ).hexdigest()
        expected_signature = f"{SIGNATURE_VERSION}={expected_digest}"
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise SignatureVerificationError("node-agent HMAC 签名不匹配")
        if seen_nonces is not None:
            seen_nonces.add(nonce)


class NonceReplayCache:
    """单 worker 下的有界时效 nonce 集合；至少覆盖整个签名有效窗口。"""

    def __init__(self, max_clock_skew_seconds: int) -> None:
        self._retention_seconds = max(60, 2 * max_clock_skew_seconds + 1)
        self._expires_at: dict[str, int] = {}
        self._lock = threading.Lock()

    def verify(
        self,
        signer: HMACSigner,
        body: bytes,
        headers: Mapping[str, str],
        *,
        now: int | None = None,
    ) -> None:
        current_time = int(time.time()) if now is None else now
        normalized_headers = {key.lower(): value for key, value in headers.items()}
        nonce = normalized_headers.get(NONCE_HEADER.lower(), "")
        if not 8 <= len(nonce) <= 256:
            raise SignatureVerificationError("node-agent HMAC nonce 长度无效")
        with self._lock:
            self._expires_at = {
                item: expiry for item, expiry in self._expires_at.items() if expiry > current_time
            }
            seen_nonces = set(self._expires_at)
            signer.verify(body, headers, now=current_time, seen_nonces=seen_nonces)
            self._expires_at[nonce] = current_time + self._retention_seconds
