#!/usr/bin/env python3
"""交互生成 OpenLLMOps 首次部署所需密钥。

脚本只打印到当前终端，不写文件、不上传，也不把密码作为命令行参数。建议把结果
直接保存到密码管理器，再手工填入权限为 0600 的 deploy/.env 或 secret 文件。
"""

from __future__ import annotations

import getpass
import secrets

from argon2 import PasswordHasher


def _read_password() -> str:
    password = getpass.getpass("管理员密码: ")
    repeated = getpass.getpass("再次输入管理员密码: ")
    if password != repeated:
        raise SystemExit("两次密码不一致")
    if len(password) < 12:
        raise SystemExit("管理员密码至少需要 12 个字符")
    return password


def main() -> None:
    password_hash = PasswordHasher().hash(_read_password())
    values = {
        # Argon2 哈希包含 `$`，单引号可避免 Compose .env 插值。
        "ADMIN_PASSWORD_HASH": f"'{password_hash}'",
        "SESSION_SIGNING_KEY": secrets.token_urlsafe(48),
        "ADMIN_API_KEY": f"ollm_admin_{secrets.token_urlsafe(32)}",
        "API_KEY_PEPPER": secrets.token_urlsafe(48),
        "NODE_AGENT_TOKEN": secrets.token_urlsafe(48),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(36),
    }
    print("\n请安全保存以下内容；离开此终端后脚本无法恢复：")
    for key, value in values.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
