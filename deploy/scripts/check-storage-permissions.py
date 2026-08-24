"""以部署配置的真实身份验证共享存储权限。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CONTROLLED_DIRECTORIES = (
    "models",
    "inbox",
    "model-staging",
    "datasets",
    "evaluation-datasets",
    "evaluation-output",
    "checkpoints",
    "training-configs",
    "runtime",
)


def assume_configured_identity(uid: int, gid: int) -> None:
    if uid <= 0 or gid <= 0:
        raise SystemExit("APP_UID/APP_GID 必须是非 root 正整数")
    if os.geteuid() == 0:
        # 清空 root 的附加组，否则 os.access 可能因为额外组权限给出假阳性。
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    elif (os.geteuid(), os.getegid()) != (uid, gid):
        raise SystemExit("非 root 执行预检时，当前 UID/GID 必须等于 APP_UID/APP_GID")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "用法：check-storage-permissions.py <storage_root> <uid> <gid>"
        )
    root = Path(sys.argv[1])
    try:
        uid = int(sys.argv[2])
        gid = int(sys.argv[3])
    except ValueError as exc:
        raise SystemExit("APP_UID/APP_GID 必须是整数") from exc

    assume_configured_identity(uid, gid)
    for child in CONTROLLED_DIRECTORIES:
        target = root / child
        if not target.is_dir() or not os.access(target, os.R_OK | os.W_OK | os.X_OK):
            raise SystemExit(f"受控目录对 APP_UID/APP_GID 不可读写进入：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
