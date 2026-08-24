"""控制面与节点代理一致的 vLLM 高级参数安全合同。"""

from __future__ import annotations

import math
import re
from typing import Any

VLLM_ARGUMENT_ALLOWLIST = frozenset(
    {
        "block-size",
        "cpu-offload-gb",
        "disable-custom-all-reduce",
        "disable-log-stats",
        "distributed-executor-backend",
        "dtype",
        "enable-chunked-prefill",
        "enable-prefix-caching",
        "enforce-eager",
        "gpu-memory-utilization",
        "kv-cache-dtype",
        "max-logprobs",
        "max-model-len",
        "max-num-batched-tokens",
        "max-num-seqs",
        "pipeline-parallel-size",
        "quantization",
        "seed",
        "tokenizer-mode",
    }
)
BOOLEAN_ARGUMENTS = frozenset(
    {
        "disable-custom-all-reduce",
        "disable-log-stats",
        "enable-chunked-prefill",
        "enable-prefix-caching",
        "enforce-eager",
    }
)
POSITIVE_INTEGER_LIMITS = {
    "max-model-len": 131_072,
    "max-num-batched-tokens": 65_536,
    "max-num-seqs": 1_024,
}
NONNEGATIVE_INTEGER_LIMITS = {"max-logprobs": 100, "seed": 2**32 - 1}
ENUM_ARGUMENTS: dict[str, frozenset[str]] = {
    "distributed-executor-backend": frozenset({"mp"}),
    "dtype": frozenset({"auto", "half", "float16", "bfloat16", "float", "float32"}),
    "kv-cache-dtype": frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}),
    "tokenizer-mode": frozenset({"auto", "slow", "mistral"}),
}


def validate_vllm_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if len(arguments) > 128:
        raise ValueError("vLLM 参数数量超过安全上限")
    normalized_keys: set[str] = set()
    for raw_key, value in arguments.items():
        key = raw_key.strip().lower().replace("_", "-")
        if key in normalized_keys:
            raise ValueError(f"vLLM 参数规范化后重复：{key}")
        normalized_keys.add(key)
        if key not in VLLM_ARGUMENT_ALLOWLIST:
            raise ValueError(f"不允许的 vLLM 参数：{raw_key}")
        if isinstance(value, str) and (len(value) > 2048 or any(ord(character) < 32 for character in value)):
            raise ValueError(f"vLLM 参数 {raw_key} 包含非法字符或过长")
        _validate_value(key, value)
    return arguments


def _validate_value(key: str, value: Any) -> None:
    if key in BOOLEAN_ARGUMENTS:
        if not isinstance(value, bool):
            raise ValueError(f"vLLM 参数 {key} 必须是布尔值")
        return
    if key in POSITIVE_INTEGER_LIMITS:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"vLLM 参数 {key} 必须是正整数")
        if value > POSITIVE_INTEGER_LIMITS[key]:
            raise ValueError(f"vLLM 参数 {key} 超出安全上限 {POSITIVE_INTEGER_LIMITS[key]}")
        return
    if key in NONNEGATIVE_INTEGER_LIMITS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"vLLM 参数 {key} 必须是非负整数")
        if value > NONNEGATIVE_INTEGER_LIMITS[key]:
            raise ValueError(f"vLLM 参数 {key} 超出安全上限 {NONNEGATIVE_INTEGER_LIMITS[key]}")
        return
    if key == "cpu-offload-gb":
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 16
        ):
            raise ValueError("vLLM 参数 cpu-offload-gb 必须位于 0..16")
        return
    if key == "block-size":
        if isinstance(value, bool) or not isinstance(value, int) or value not in {8, 16, 32}:
            raise ValueError("vLLM 参数 block-size 仅允许 8、16 或 32")
        return
    if key == "gpu-memory-utilization":
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.1 <= value <= 0.98
        ):
            raise ValueError("vLLM 参数 gpu-memory-utilization 必须位于 0.1..0.98")
        return
    if key == "pipeline-parallel-size":
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise ValueError("首版单机调度固定 pipeline-parallel-size=1")
        return
    allowed_values = ENUM_ARGUMENTS.get(key)
    if allowed_values is not None:
        if not isinstance(value, str) or value.lower() not in allowed_values:
            raise ValueError(f"vLLM 参数 {key} 仅允许：{', '.join(sorted(allowed_values))}")
        return
    if key == "quantization":
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
            raise ValueError("vLLM 参数 quantization 的值格式不安全")
        return
    raise ValueError(f"vLLM 参数 {key} 缺少类型约束")
