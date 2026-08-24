#!/usr/bin/env python3
"""比对 Compose 最终渲染结果，防止调用进程环境覆盖已预检的 .env。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_origins(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise TypeError("渲染后的 CORS_ORIGINS 不是字符串")
    value = raw.strip()
    parsed = json.loads(value) if value.startswith("[") else value.split(",")
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError("渲染后的 CORS_ORIGINS 格式无效")
    return [item.strip() for item in parsed if item.strip()]


def find_bind_source(service: dict[str, object], target: str) -> Path:
    volumes = service.get("volumes")
    if not isinstance(volumes, list):
        raise TypeError(f"服务缺少挂载：{target}")
    for volume in volumes:
        if isinstance(volume, dict) and volume.get("target") == target:
            source = volume.get("source")
            if volume.get("type") != "bind" or not isinstance(source, str):
                raise ValueError(f"{target} 必须是 bind mount")
            return Path(source).resolve()
    raise ValueError(f"服务缺少挂载：{target}")


def validate(
    rendered: dict[str, object], values: dict[str, str], deploy_dir: Path
) -> None:
    services = rendered.get("services")
    if not isinstance(services, dict):
        raise TypeError("Compose 渲染结果缺少 services")
    try:
        web = services["web"]
        api = services["api"]
        agent = services["node-agent"]
    except KeyError as exc:
        raise ValueError(f"Compose 渲染结果缺少服务：{exc.args[0]}") from exc
    if not all(isinstance(service, dict) for service in (web, api, agent)):
        raise TypeError("Compose 服务结构无效")

    web_environment = web.get("environment")
    api_environment = api.get("environment")
    if not isinstance(web_environment, dict) or not isinstance(api_environment, dict):
        raise TypeError("Compose 服务 environment 结构无效")
    expected_common_name = values.get("TLS_COMMON_NAME", "openllmops.local")
    expected_auto = values.get("TLS_AUTO_GENERATE", "true")
    if str(web_environment.get("TLS_COMMON_NAME")) != expected_common_name:
        raise ValueError("调用进程环境覆盖了 TLS_COMMON_NAME")
    if str(web_environment.get("TLS_AUTO_GENERATE")) != expected_auto:
        raise ValueError("调用进程环境覆盖了 TLS_AUTO_GENERATE")
    expected_uid = values.get("APP_UID", "1000")
    expected_gid = values.get("APP_GID", "1000")
    if str(web_environment.get("TLS_OWNER_UID")) != expected_uid:
        raise ValueError("调用进程环境覆盖了 APP_UID")
    if str(web_environment.get("TLS_OWNER_GID")) != expected_gid:
        raise ValueError("调用进程环境覆盖了 APP_GID")
    expected_origins = parse_origins(values.get("CORS_ORIGINS", ""))
    if parse_origins(api_environment.get("CORS_ORIGINS")) != expected_origins:
        raise ValueError("调用进程环境覆盖了 CORS_ORIGINS")

    tls_value = Path(values.get("TLS_DIR", "./secrets/tls"))
    expected_tls = (
        tls_value.resolve()
        if tls_value.is_absolute()
        else (deploy_dir / tls_value).resolve()
    )
    if find_bind_source(web, "/etc/nginx/tls") != expected_tls:
        raise ValueError("调用进程环境覆盖了 TLS_DIR")

    storage_value = Path(values.get("OPENLLMOPS_STORAGE_ROOT", "/srv/openllmops"))
    expected_storage = storage_value.resolve()
    for service_name, service in (("api", api), ("node-agent", agent)):
        if find_bind_source(service, "/srv/openllmops") != expected_storage:
            raise ValueError(
                f"调用进程环境覆盖了 {service_name} 的 OPENLLMOPS_STORAGE_ROOT"
            )
        if str(service.get("user")) != f"{expected_uid}:{expected_gid}":
            raise ValueError(f"调用进程环境覆盖了 {service_name} 的 APP_UID/APP_GID")

    ports = web.get("ports")
    if not isinstance(ports, list) or not any(
        isinstance(port, dict)
        and int(port.get("target", 0)) == 8443
        and str(port.get("published")) == values.get("HTTPS_PORT", "443")
        for port in ports
    ):
        raise ValueError("调用进程环境覆盖了 HTTPS_PORT")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "用法：docker compose ... config --format json | validate-rendered-config.py <.env>"
        )
    try:
        rendered = json.load(sys.stdin)
        validate(
            rendered, read_env(Path(sys.argv[1])), Path(__file__).resolve().parents[1]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Compose 最终配置校验失败：{exc}") from exc


if __name__ == "__main__":
    main()
