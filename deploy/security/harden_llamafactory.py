"""对固定上游源码应用 GHSA-mwc7-mf87-v3mf 防护，并在镜像构建期验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

UPSTREAM_REVISION = "c4e09c7cbe18844816af9e18a97fe465515edbcd"
UPSTREAM_VERSION = 'VERSION = "0.9.6.dev0"'
EXPECTED_HASHES = {
    "src/llamafactory/webui/chatter.py": "ab1425f221b0cb5594909cf5a592258634e6039c6c1a77666f9321d3c90bf0ad",
    "src/llamafactory/webui/runner.py": "910551f6f1c45954dc429d4c0f0c2d07950fc2c7575070d7bd53d4e198a10748",
    "src/llamafactory/webui/components/export.py": "4478048c942543cd5cfbd99defca8c21401f3c1b2c634a49602888eff64f7eda",
    "src/llamafactory/hparams/model_args.py": "4a4d0873b1121526ac395ea7b7b4c992f0e793e277e4e22e0952d1e1056b3c39",
    "src/llamafactory/v1/config/model_args.py": "24f9ec0d4cddfd3121671a85c6d183f458ac4b573ffce8f641b2305db3372d1e",
    "src/llamafactory/train/megatron_bridge/config_builder.py": "b4c7b7597cd6b2a8161d5e14ce87d6a73c9c39ce7ee3b8a6c7cae2a5e445d9ca",
    "src/llamafactory/model/loader.py": "ed786dd7ddc947495c847552e666007fdbc28440a816ff19943281f400c5a1ce",
}
POLICY_ERROR = "OpenLLMOps hardened image permanently disables trust_remote_code"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream(root: Path) -> None:
    version_file = root / "src/llamafactory/extras/env.py"
    if UPSTREAM_VERSION not in version_file.read_text(encoding="utf-8"):
        raise RuntimeError("基础镜像不是已审计的 LLaMAFactory 0.9.6.dev0")
    mismatches = {
        relative: digest(root / relative)
        for relative, expected in EXPECTED_HASHES.items()
        if not (root / relative).is_file() or digest(root / relative) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"上游源码哈希不匹配，拒绝在未知版本上套用补丁：{mismatches}"
        )


def replace_exact(path: Path, old: str, new: str, expected_count: int) -> None:
    content = path.read_text(encoding="utf-8")
    actual_count = content.count(old)
    if actual_count != expected_count:
        raise RuntimeError(
            f"{path} 预期匹配 {expected_count} 处，实际 {actual_count} 处；拒绝模糊补丁"
        )
    path.write_text(content.replace(old, new), encoding="utf-8")


def apply_hardening(root: Path, marker: Path) -> None:
    verify_upstream(root)

    webui_files = (
        root / "src/llamafactory/webui/chatter.py",
        root / "src/llamafactory/webui/runner.py",
        root / "src/llamafactory/webui/components/export.py",
    )
    expected_counts = (1, 2, 1)
    for path, count in zip(webui_files, expected_counts):
        replace_exact(path, "trust_remote_code=True", "trust_remote_code=False", count)

    replace_exact(
        root / "src/llamafactory/train/megatron_bridge/config_builder.py",
        '{"trust_remote_code": True}',
        '{"trust_remote_code": False}',
        1,
    )
    replace_exact(
        root / "src/llamafactory/model/loader.py",
        '"trust_remote_code": model_args.trust_remote_code',
        '"trust_remote_code": False',
        1,
    )
    replace_exact(
        root / "src/llamafactory/model/loader.py",
        "trust_remote_code=model_args.trust_remote_code",
        "trust_remote_code=False",
        1,
    )

    replace_exact(
        root / "src/llamafactory/hparams/model_args.py",
        "    def __post_init__(self):\n        if self.model_name_or_path is None:",
        "    def __post_init__(self):\n"
        "        if self.trust_remote_code:\n"
        f'            raise ValueError("{POLICY_ERROR}")\n\n'
        "        if self.model_name_or_path is None:",
        1,
    )
    replace_exact(
        root / "src/llamafactory/v1/config/model_args.py",
        "    def __post_init__(self) -> None:\n        supported_flash_attn =",
        "    def __post_init__(self) -> None:\n"
        "        if self.trust_remote_code:\n"
        f'            raise ValueError("{POLICY_ERROR}")\n\n'
        "        supported_flash_attn =",
        1,
    )

    # 防止基础镜像中可能存在的 unchecked-hash 字节码绕过已修改的 Python 源码。
    for pattern in ("*.pyc", "*.pyo"):
        for compiled_file in (root / "src/llamafactory").rglob(pattern):
            compiled_file.unlink()

    verify_hardening(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "advisory": "GHSA-mwc7-mf87-v3mf",
                "status": "mitigated",
                "trust_remote_code": "disabled",
                "upstream_revision": UPSTREAM_REVISION,
                "upstream_version": "0.9.6.dev0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_hardening(root: Path) -> None:
    source_root = root / "src/llamafactory"
    compiled_files = [
        str(path.relative_to(root))
        for pattern in ("*.pyc", "*.pyo")
        for path in source_root.rglob(pattern)
    ]
    if compiled_files:
        raise RuntimeError(f"仍存在补丁前编译的 Python 字节码：{compiled_files}")

    unsafe_literals: list[str] = []
    for path in source_root.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if (
            "trust_remote_code=True" in content
            or '"trust_remote_code": True' in content
        ):
            unsafe_literals.append(str(path.relative_to(root)))
    if unsafe_literals:
        raise RuntimeError(f"仍存在启用远程代码的字面量：{unsafe_literals}")

    guarded_files = (
        root / "src/llamafactory/hparams/model_args.py",
        root / "src/llamafactory/v1/config/model_args.py",
    )
    for path in guarded_files:
        content = path.read_text(encoding="utf-8")
        if POLICY_ERROR not in content:
            raise RuntimeError(f"模型参数入口缺少永久禁用策略：{path}")

    loader = (root / "src/llamafactory/model/loader.py").read_text(encoding="utf-8")
    if '"trust_remote_code": model_args.trust_remote_code' in loader:
        raise RuntimeError("核心模型加载器仍接受调用方的 trust_remote_code")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("apply", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, required=True)
        subparser.add_argument(
            "--marker",
            type=Path,
            default=Path("/usr/local/share/openllmops/llamafactory-hardening.json"),
        )
    args = parser.parse_args()
    if args.command == "apply":
        apply_hardening(args.root, args.marker)
    else:
        verify_hardening(args.root)
        marker = json.loads(args.marker.read_text(encoding="utf-8"))
        if marker.get("upstream_revision") != UPSTREAM_REVISION:
            raise RuntimeError("安全构建标记与已审计上游 revision 不一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
