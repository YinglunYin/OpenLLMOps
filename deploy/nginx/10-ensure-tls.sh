#!/bin/sh
set -eu

cert_file=/etc/nginx/tls/tls.crt
key_file=/etc/nginx/tls/tls.key

if [ -s "$cert_file" ] && [ -s "$key_file" ]; then
    exit 0
fi

if [ "${TLS_AUTO_GENERATE:-true}" != "true" ]; then
    echo "TLS_AUTO_GENERATE=false，但 /etc/nginx/tls/tls.crt 或 tls.key 不存在" >&2
    exit 1
fi

# 自签名证书只用于首次启动和内网联调；生产证书应由组织 CA 签发并挂载进来。
common_name=${TLS_COMMON_NAME:-openllmops.local}
case "$common_name" in
    *[!A-Za-z0-9.-]*|'')
        echo "TLS_COMMON_NAME 只能包含字母、数字、点和连字符" >&2
        exit 1
        ;;
esac
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 30 \
    -subj "/CN=${common_name}" \
    -addext "subjectAltName=DNS:${common_name},DNS:localhost,IP:127.0.0.1" \
    -keyout "$key_file" \
    -out "$cert_file"
chmod 0600 "$key_file"
chmod 0644 "$cert_file"
