#!/usr/bin/env python3
"""拒绝 shell 环境静默覆盖 Compose 的受控插值变量。"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

VARIABLE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "用法：reject-compose-overrides.py <deploy/.env> <compose.yaml>"
        )
    env_file = Path(sys.argv[1])
    compose_file = Path(sys.argv[2])
    try:
        file_stat = env_file.lstat()
        if env_file.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("deploy/.env 必须是非软链接普通文件")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise ValueError("deploy/.env 权限过宽；请执行 chmod 0600")
        if file_stat.st_uid not in {0, os.geteuid()}:
            raise ValueError("deploy/.env 必须属于当前用户或 root")
        # 同时读取 env 文件，确保路径/编码错误在任何容器操作前暴露；值本身绝不输出。
        env_file.read_text(encoding="utf-8")
        variables = set(
            VARIABLE_PATTERN.findall(compose_file.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"Compose 环境覆盖校验失败：{exc}") from exc
    collisions = sorted(variables.intersection(os.environ))
    if collisions:
        names = ", ".join(collisions[:12])
        suffix = " 等" if len(collisions) > 12 else ""
        raise SystemExit(
            f"调用进程已导出 Compose 插值变量（{names}{suffix}），会覆盖 --env-file；"
            "请在干净 shell 中运行预检"
        )


if __name__ == "__main__":
    main()
