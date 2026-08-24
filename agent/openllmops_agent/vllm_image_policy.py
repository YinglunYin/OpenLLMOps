from __future__ import annotations

import argparse
import re

VLLM_RUNTIME_IMAGE = "vllm/vllm-openai:v0.27.1"
VLLM_CU129_RUNTIME_IMAGE = "vllm/vllm-openai:v0.27.1-cu129"
VLLM_AMD64_DIGEST = "sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2"
VLLM_CU129_AMD64_DIGEST = "sha256:6666717cd1cadf9adfff8abec9c3f2eca6e27e742de06fe7d7f129fa3d647732"

_DEVELOPMENT_REFERENCES = frozenset({VLLM_RUNTIME_IMAGE, VLLM_CU129_RUNTIME_IMAGE})
_DIGEST_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
_VERIFIED_AMD64_DIGESTS = frozenset({VLLM_AMD64_DIGEST, VLLM_CU129_AMD64_DIGEST})


class UnsafeVLLMImage(ValueError):
    pass


def validate_vllm_image_reference(reference: str) -> str:
    normalized = reference.strip()
    if not normalized:
        raise UnsafeVLLMImage("推理镜像引用不能为空")
    if normalized.endswith(":latest") or ":latest@" in normalized:
        raise UnsafeVLLMImage("推理镜像禁止使用 latest")
    if normalized in _DEVELOPMENT_REFERENCES:
        return normalized
    if _DIGEST_REFERENCE.fullmatch(normalized):
        digest = normalized.rsplit("@", maxsplit=1)[1]
        if digest in _VERIFIED_AMD64_DIGESTS:
            return normalized
        raise UnsafeVLLMImage("推理镜像 digest 不是已核验的 v0.27.1 amd64 变体")
    raise UnsafeVLLMImage(
        "推理镜像只允许官方 v0.27.1/v0.27.1-cu129 固定标签；"
        "生产镜像必须使用已核验的 registry/repository@sha256 摘要"
    )


def validate_vllm_image_list(raw: str) -> str:
    references = [item.strip() for item in raw.split(",") if item.strip()]
    if not references:
        raise UnsafeVLLMImage("VLLM_ALLOWED_IMAGES 至少包含一个镜像")
    for reference in references:
        validate_vllm_image_reference(reference)
    return ",".join(references)


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 OpenLLMOps vLLM 镜像引用策略")
    parser.add_argument("references", help="逗号分隔的 VLLM_ALLOWED_IMAGES")
    args = parser.parse_args()
    try:
        normalized = validate_vllm_image_list(args.references)
    except UnsafeVLLMImage as exc:
        parser.error(str(exc))
    print(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
