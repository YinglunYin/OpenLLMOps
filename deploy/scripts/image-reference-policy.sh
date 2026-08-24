#!/bin/sh

# 该文件同时被生产预检和无 Docker 的静态测试加载；调用方自行启用 ``set -e``。
require_production_digest() {
    policy_environment=$1
    image_kind=$2
    image_reference=$3

    [ "$policy_environment" = "production" ] || return 0

    case "$image_reference" in
        *@sha256:*)
            image_repository=${image_reference%@sha256:*}
            image_digest=${image_reference##*@sha256:}
            ;;
        *)
            image_repository=
            image_digest=
            ;;
    esac
    case "$image_digest" in
        *[!0-9a-f]*) digest_is_hex=false ;;
        *) digest_is_hex=true ;;
    esac
    if [ -z "$image_repository" ] || [ "${#image_digest}" -ne 64 ] || [ "$digest_is_hex" != "true" ]; then
        echo "生产环境的 $image_kind 必须使用 registry/repository@sha256:<64位小写十六进制摘要>：$image_reference" >&2
        return 1
    fi
}
