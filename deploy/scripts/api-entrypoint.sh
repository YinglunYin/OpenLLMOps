#!/bin/sh
set -eu

# 生产环境优先运行数据库迁移；开发环境可显式用 AUTO_CREATE_TABLES 快速建表。
if [ "${RUN_MIGRATIONS:-true}" = "true" ] && [ -f /app/alembic.ini ]; then
    alembic upgrade head
fi

trusted_proxy_cidrs=${TRUSTED_PROXY_CIDRS:-172.30.10.10/32}
case "$trusted_proxy_cidrs" in
    *"*"* | *";"* | *" "*)
        echo "TRUSTED_PROXY_CIDRS 包含不安全字符" >&2
        exit 1
        ;;
esac

# 只信任与后端安全配置相同的代理网段，禁止任意 runtime 容器伪造转发头。
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 \
    --proxy-headers --forwarded-allow-ips="$trusted_proxy_cidrs"
