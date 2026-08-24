import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping, MutableSet
from typing import Any

import httpx
from pydantic import BaseModel

from app.schemas.agent_contract import AgentCommand, AgentCommandResponse

SIGNATURE_HEADER = "X-OpenLLMOps-Signature"
TIMESTAMP_HEADER = "X-OpenLLMOps-Timestamp"
NONCE_HEADER = "X-OpenLLMOps-Nonce"
SIGNATURE_VERSION = "v1"


class NodeAgentError(RuntimeError):
    """node-agent 合同、鉴权或网络调用失败。"""


class SignatureVerificationError(NodeAgentError):
    """HMAC 签名无效、过期或发生重放。"""


def canonical_json(value: BaseModel | Mapping[str, Any]) -> bytes:
    """生成稳定 JSON 字节，确保两端对同一业务对象计算出相同签名。"""

    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class HMACSigner:
    """为 node-agent 请求和响应提供防篡改、时效与重放校验。"""

    def __init__(self, secret: str | bytes, max_clock_skew_seconds: int = 30) -> None:
        secret_bytes = secret.encode() if isinstance(secret, str) else secret
        if len(secret_bytes) < 16:
            raise ValueError("node-agent HMAC 共享密钥至少需要 16 字节")
        self._secret = secret_bytes
        self._max_clock_skew_seconds = max_clock_skew_seconds

    @staticmethod
    def _signing_input(body: bytes, timestamp: str, nonce: str) -> bytes:
        # 将版本、时间和随机数纳入签名，避免相同 JSON 在有效期内被原样重放。
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


class NodeAgentHTTPClient:
    """使用双向 HMAC 的 node-agent 客户端。

    node-agent 必须对响应原始字节签名；控制面在解析 JSON 前先验签，防止代理层或
    内网中间节点伪造“容器已停止”等会导致 GPU 租约提前释放的高风险状态。
    """

    def __init__(
        self,
        base_url: str,
        secret: str,
        *,
        max_clock_skew_seconds: int = 30,
        timeout_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._signer = HMACSigner(secret, max_clock_skew_seconds)
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    async def execute(self, command: AgentCommand) -> AgentCommandResponse:
        body = canonical_json(command)
        headers = {
            "Content-Type": "application/json",
            **self._signer.sign(body),
        }
        try:
            response = await self._client.post(
                "/v1/workloads/commands",
                content=body,
                headers=headers,
            )
            # 先验签再解释 HTTP 状态，避免未签名的网关错误伪装成 agent 业务拒绝。
            self._signer.verify(response.content, response.headers)
            result = AgentCommandResponse.model_validate_json(response.content)
            if not (response.is_success or response.status_code in {409, 422}):
                response.raise_for_status()
        except (httpx.HTTPError, ValueError, SignatureVerificationError) as exc:
            raise NodeAgentError(f"node-agent 调用失败：{exc}") from exc
        if result.request_id != command.request_id:
            raise NodeAgentError("node-agent 响应 request_id 与请求不一致")
        return result

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
