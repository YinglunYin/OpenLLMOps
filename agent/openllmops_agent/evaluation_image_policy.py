from __future__ import annotations

import argparse
import re
from collections.abc import Mapping

EVALUATION_RUNTIME_IMAGE = "openllmops/evaluation:0.1.0-vllm0.27.1"
EVALUATION_RUNNER_LABEL = "com.openllmops.runner"
EVALUATION_REMOTE_CODE_LABEL = "com.openllmops.security.trust-remote-code"
EVALUATION_BASE_LABEL = "com.openllmops.base.vllm"
EXPECTED_EVALUATION_LABELS = {
    EVALUATION_RUNNER_LABEL: "evaluation-pair-v1",
    EVALUATION_REMOTE_CODE_LABEL: "disabled",
    EVALUATION_BASE_LABEL: "v0.27.1",
}

_DIGEST_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")


class UnsafeEvaluationImage(ValueError):
    pass


def validate_evaluation_image_reference(reference: str) -> str:
    normalized = reference.strip()
    if not normalized:
        raise UnsafeEvaluationImage("评测镜像引用不能为空")
    if normalized == EVALUATION_RUNTIME_IMAGE:
        return normalized
    if normalized.endswith(":latest") or ":latest@" in normalized:
        raise UnsafeEvaluationImage("评测镜像禁止使用 latest")
    if not _DIGEST_REFERENCE.fullmatch(normalized):
        raise UnsafeEvaluationImage(
            "评测镜像只允许项目固定本地版本，生产镜像必须使用 registry/repository@sha256 摘要"
        )
    return normalized


def validate_evaluation_image_list(raw: str) -> str:
    references = [item.strip() for item in raw.split(",") if item.strip()]
    if not references:
        raise UnsafeEvaluationImage("EVALUATION_ALLOWED_IMAGES 至少包含一个镜像")
    for reference in references:
        validate_evaluation_image_reference(reference)
    return ",".join(references)


def validate_evaluation_image_labels(labels: Mapping[str, str] | None) -> None:
    actual = labels or {}
    mismatched = {
        key: expected for key, expected in EXPECTED_EVALUATION_LABELS.items() if actual.get(key) != expected
    }
    if mismatched:
        raise UnsafeEvaluationImage("评测镜像缺少或伪造安全构建标签：" + ", ".join(sorted(mismatched)))


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 OpenLLMOps 评测镜像引用策略")
    parser.add_argument("references", help="逗号分隔的 EVALUATION_ALLOWED_IMAGES")
    args = parser.parse_args()
    try:
        normalized = validate_evaluation_image_list(args.references)
    except UnsafeEvaluationImage as exc:
        parser.error(str(exc))
    print(normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
