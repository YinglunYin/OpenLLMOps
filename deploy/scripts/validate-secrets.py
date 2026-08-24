#!/usr/bin/env python3
"""在启动任何容器前校验生产密钥，错误信息绝不回显密钥内容。"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_LENGTHS = {
    "POSTGRES_PASSWORD": 24,
    "SESSION_SIGNING_KEY": 32,
    "ADMIN_API_KEY": 32,
    "API_KEY_PEPPER": 32,
    "NODE_AGENT_TOKEN": 32,
}
PLACEHOLDER_PREFIXES = ("replace-with-", "example-", "changeme", "change-me")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"环境文件第 {line_number} 行缺少等号")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def validate(values: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for key, minimum in REQUIRED_LENGTHS.items():
        value = values.get(key, "")
        normalized = value.casefold()
        if (
            not value
            or len(value) < minimum
            or normalized.startswith(PLACEHOLDER_PREFIXES)
        ):
            raise ValueError(f"{key} 未替换为至少 {minimum} 字符的随机密钥")
        previous = seen.get(value)
        if previous is not None:
            raise ValueError(f"{key} 不能与 {previous} 使用相同密钥")
        seen[value] = key


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：validate-secrets.py <deploy/.env>")
    try:
        validate(read_env(Path(sys.argv[1])))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"生产密钥校验失败：{exc}") from exc


if __name__ == "__main__":
    main()
