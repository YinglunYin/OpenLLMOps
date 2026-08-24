#!/bin/sh
set -eu
umask 077

cert_file=/etc/nginx/tls/tls.crt
key_file=/etc/nginx/tls/tls.key
owner_uid=${TLS_OWNER_UID:-1000}
owner_gid=${TLS_OWNER_GID:-1000}
case "$owner_uid" in
    *[!0-9]* | 0 | '')
        echo "TLS_OWNER_UID/TLS_OWNER_GID 必须是非 root 正整数" >&2
        exit 1
        ;;
esac
case "$owner_gid" in
    *[!0-9]* | 0 | '')
        echo "TLS_OWNER_UID/TLS_OWNER_GID 必须是非 root 正整数" >&2
        exit 1
        ;;
esac

for tls_file in "$cert_file" "$key_file"; do
    if [ -L "$tls_file" ] || { [ -e "$tls_file" ] && [ ! -f "$tls_file" ]; }; then
        echo "TLS 文件必须是非软链接普通文件：$tls_file" >&2
        exit 1
    fi
done

if [ -s "$cert_file" ] && [ -s "$key_file" ]; then
    # 权限已由宿主机预检确认。容器只有 DAC_OVERRIDE、没有 FOWNER，不能也不应
    # 修改由宿主机管理员拥有的正式证书权限。
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
temporary_key="${key_file}.tmp.$$"
temporary_cert="${cert_file}.tmp.$$"
trap 'rm -f "$temporary_key" "$temporary_cert"' EXIT HUP INT TERM
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 30 \
    -subj "/CN=${common_name}" \
    -addext "subjectAltName=DNS:${common_name},DNS:localhost,IP:127.0.0.1" \
    -keyout "$temporary_key" \
    -out "$temporary_cert"
chmod 0600 "$temporary_key"
chmod 0644 "$temporary_cert"
mv -f "$temporary_key" "$key_file"
mv -f "$temporary_cert" "$cert_file"
chown "$owner_uid:$owner_gid" "$key_file" "$cert_file"
trap - EXIT HUP INT TERM
