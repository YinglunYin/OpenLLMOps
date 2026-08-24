#!/usr/bin/env python3
"""在启动容器前校验 HTTPS、同源与 TLS 挂载配置。"""

from __future__ import annotations

import json
import re
import stat
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

COMMON_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
PLACEHOLDER_HOST_SUFFIXES = (
    ".example.internal",
    ".example.com",
    ".example.net",
    ".example.org",
)


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
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_origins(raw: str) -> list[str]:
    if raw.lstrip().startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise ValueError("CORS_ORIGINS 必须是字符串数组")
        return parsed
    return [item.strip() for item in raw.split(",") if item.strip()]


def _run_openssl(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["openssl", *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("OpenSSL 校验超时，私钥不得依赖交互口令") from exc
    if result.returncode != 0:
        raise ValueError("TLS 证书或私钥无法通过 OpenSSL 校验")
    return result.stdout


def validate_tls_material(cert_file: Path, key_file: Path, common_name: str) -> None:
    _run_openssl("x509", "-in", str(cert_file), "-noout", "-checkend", "86400")
    _run_openssl("x509", "-in", str(cert_file), "-noout", "-checkhost", common_name)
    cert_public_key = _run_openssl("x509", "-in", str(cert_file), "-pubkey", "-noout")
    private_public_key = _run_openssl(
        "pkey",
        "-in",
        str(key_file),
        "-passin",
        "pass:",
        "-pubout",
    )
    if cert_public_key != private_public_key:
        raise ValueError("tls.crt 与 tls.key 不是同一密钥对")


def require_safe_tls_directory(
    path: Path, auto_generate: bool, common_name: str
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"TLS_DIR 必须是已存在的非软链接目录：{path}")
    directory_mode = stat.S_IMODE(path.stat().st_mode)
    if directory_mode & 0o022:
        raise ValueError(
            "TLS_DIR 不能被组或其他用户写入；请执行 chmod 0755 或更严格权限"
        )

    cert_file = path / "tls.crt"
    key_file = path / "tls.key"
    present = [
        candidate.exists() or candidate.is_symlink()
        for candidate in (cert_file, key_file)
    ]
    if any(present) and not all(present):
        raise ValueError("TLS_DIR 中 tls.crt 与 tls.key 必须同时存在或同时不存在")
    if not all(present):
        if not auto_generate:
            raise ValueError("TLS_AUTO_GENERATE=false 时必须提供 tls.crt 与 tls.key")
        return

    for candidate in (cert_file, key_file):
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size == 0
        ):
            raise ValueError(f"TLS 文件必须是非空、非软链接普通文件：{candidate.name}")
    key_mode = stat.S_IMODE(key_file.stat().st_mode)
    if key_mode & 0o077:
        raise ValueError("tls.key 只能由所有者读取/写入；请执行 chmod 0600")
    cert_mode = stat.S_IMODE(cert_file.stat().st_mode)
    if cert_mode & 0o022:
        raise ValueError("tls.crt 不能被组或其他用户写入")
    validate_tls_material(cert_file, key_file, common_name)


def validate(values: dict[str, str], deploy_dir: Path) -> None:
    common_name = values.get("TLS_COMMON_NAME", "openllmops.local").strip()
    if not COMMON_NAME_PATTERN.fullmatch(common_name):
        raise ValueError("TLS_COMMON_NAME 只能是合法 DNS 名称")
    try:
        ip_address(common_name)
    except ValueError:
        pass
    else:
        raise ValueError("TLS_COMMON_NAME 当前只支持 DNS 名称，不支持 IP 地址")
    port_raw = values.get("HTTPS_PORT", "443")
    try:
        https_port = int(port_raw)
    except ValueError as exc:
        raise ValueError("HTTPS_PORT 必须是整数") from exc
    if not 1 <= https_port <= 65535:
        raise ValueError("HTTPS_PORT 必须介于 1 与 65535")
    if https_port != 443:
        raise ValueError("当前 HTTPS 重定向合同要求 HTTPS_PORT=443")

    origins = parse_origins(values.get("CORS_ORIGINS", ""))
    expected_origin = f"https://{common_name}"
    if expected_origin not in origins:
        raise ValueError(f"CORS_ORIGINS 必须包含浏览器入口 {expected_origin}")
    for origin in origins:
        parsed = urlsplit(origin)
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError(f"CORS Origin 端口无效：{origin}") from exc
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"生产 CORS 来源必须是无路径的 HTTPS Origin：{origin}")
        if hostname.endswith(PLACEHOLDER_HOST_SUFFIXES):
            raise ValueError(f"CORS_ORIGINS 仍包含示例域名：{origin}")

    auto_raw = values.get("TLS_AUTO_GENERATE", "true").strip()
    if auto_raw not in {"true", "false"}:
        raise ValueError("TLS_AUTO_GENERATE 只能是 true 或 false")
    tls_raw = values.get("TLS_DIR", "./secrets/tls")
    tls_dir = Path(tls_raw)
    if not tls_dir.is_absolute():
        tls_dir = deploy_dir / tls_dir
    require_safe_tls_directory(tls_dir, auto_raw == "true", common_name)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：validate-web-config.py <deploy/.env>")
    try:
        validate(read_env(Path(sys.argv[1])), Path(__file__).resolve().parents[1])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Web/TLS 配置校验失败：{exc}") from exc


if __name__ == "__main__":
    main()
