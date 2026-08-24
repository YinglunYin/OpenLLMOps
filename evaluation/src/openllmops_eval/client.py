"""OpenAI Compatible 推理客户端。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    latency_ms: float


class CompatibleClient:
    """调用平台统一入口，而不是绕过 Gateway 访问临时容器端口。"""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, *, model: str, prompt: str, template: str, max_tokens: int) -> Generation:
        started = time.perf_counter()
        if template == "instruct":
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        else:
            response = await self._client.post(
                "/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["text"]
        return Generation(text=str(text), latency_ms=(time.perf_counter() - started) * 1000)

