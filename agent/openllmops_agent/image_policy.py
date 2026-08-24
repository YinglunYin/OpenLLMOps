from __future__ import annotations

import argparse
import re
from collections.abc import Mapping

HARDENED_LLAMAFACTORY_IMAGE = "openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1"
HARDENED_UPSTREAM_REVISION = "c4e09c7cbe18844816af9e18a97fe465515edbcd"
GHSA_LABEL = "com.openllmops.security.ghsa-mwc7-mf87-v3mf"
REMOTE_CODE_LABEL = "com.openllmops.security.trust-remote-code"
TRAINING_RUNNER_LABEL = "com.openllmops.runner"
TRAINING_ARTIFACT_LABEL = "com.openllmops.artifacts"
EXPECTED_SECURITY_LABELS = {
    GHSA_LABEL: "mitigated",
    REMOTE_CODE_LABEL: "disabled",
    "org.opencontainers.image.revision": HARDENED_UPSTREAM_REVISION,
    TRAINING_RUNNER_LABEL: "training-wrapper-v1",
    TRAINING_ARTIFACT_LABEL: "safetensors-validated-v1",
}

_DIGEST_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")


class UnsafeTrainingImage(ValueError):
    pass


def validate_training_image_reference(reference: str) -> str:
    """只接受项目固定开发镜像，或生产环境不可变的 sha256 digest。"""

    normalized = reference.strip()
    if not normalized:
        raise UnsafeTrainingImage("训练镜像引用不能为空")
    if normalized == HARDENED_LLAMAFACTORY_IMAGE:
        return normalized
    if normalized.startswith(("hiyouga/llamafactory", "docker.io/hiyouga/llamafactory")):
        raise UnsafeTrainingImage(
            "禁止直接使用上游 LLaMAFactory 镜像；0.9.5 及以前受 GHSA-mwc7-mf87-v3mf "
            "影响，审计时的 0.9.6.dev0 源码仍保留危险 WebUI 路径"
        )
    if ":latest" in normalized or normalized.endswith("/latest"):
        raise UnsafeTrainingImage("训练镜像禁止使用 latest")
    if not _DIGEST_REFERENCE.fullmatch(normalized):
        raise UnsafeTrainingImage("生产训练镜像必须使用 registry/repository@sha256:<64位摘要>")
    return normalized


def validate_training_image_list(raw: str) -> str:
    references = [item.strip() for item in raw.split(",") if item.strip()]
    if not references:
        raise UnsafeTrainingImage("LLAMAFACTORY_ALLOWED_IMAGES 至少包含一个镜像")
    for reference in references:
        validate_training_image_reference(reference)
    return ",".join(references)


def validate_hardening_labels(labels: Mapping[str, str] | None) -> None:
    actual = labels or {}
    missing = {key: value for key, value in EXPECTED_SECURITY_LABELS.items() if actual.get(key) != value}
    if missing:
        names = ", ".join(sorted(missing))
        raise UnsafeTrainingImage(f"训练镜像缺少或伪造了安全构建标签：{names}")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 OpenLLMOps 训练镜像引用策略")
    parser.add_argument("references", help="逗号分隔的 LLAMAFACTORY_ALLOWED_IMAGES")
    args = parser.parse_args()
    try:
        normalized = validate_training_image_list(args.references)
    except UnsafeTrainingImage as exc:
        parser.error(str(exc))
    print(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
