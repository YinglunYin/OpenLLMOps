#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
deploy_dir=$(dirname -- "$script_dir")
project_root=$(dirname -- "$deploy_dir")
env_file=${1:-"$deploy_dir/.env"}
compose_file="$deploy_dir/compose.yaml"

# 与测试共用同一份 POSIX sh 镜像引用策略，防止静态断言和生产逻辑漂移。
. "$script_dir/image-reference-policy.sh"

if [ ! -f "$env_file" ]; then
    echo "环境文件不存在：$env_file" >&2
    exit 1
fi

for command_name in docker nvidia-smi openssl python3 stat; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "缺少命令：$command_name" >&2
        exit 1
    fi
done

# Compose 的优先级规定 shell 环境覆盖 --env-file。先拒绝所有与 compose 插值同名的
# 已导出变量，避免 ENVIRONMENT/AUTH_ENABLED/密钥/镜像白名单等绕过后续文件校验。
python3 "$script_dir/reject-compose-overrides.py" "$env_file" "$compose_file"
python3 "$script_dir/read-env-value.py" "$env_file" __VALIDATE_ONLY >/dev/null

# 必须先拒绝公开占位值，随后才允许任何 Docker/Compose 操作。
python3 "$script_dir/validate-secrets.py" "$env_file"
python3 "$script_dir/validate-web-config.py" "$env_file"

version_at_least() {
    current=$1
    minimum=$2
    first=$(printf '%s\n%s\n' "$minimum" "$current" | sort -V | head -n 1)
    [ "$first" = "$minimum" ]
}

read_env_value() {
    key=$1
    python3 "$script_dir/read-env-value.py" "$env_file" "$key"
}

# CUDA_VARIANT 不由驱动或镜像名隐式推测：生产 digest 不包含 tag，
# 显式声明才能让审计、预检和评测镜像重建遵守同一契约。下限取自
# NVIDIA 对应 Toolkit 更新版的 Linux x86_64 驱动表；不依赖有功能限制的
# 跨小版本兼容下限，因为 vLLM 会运行 JIT 内核。
cuda_variant=$(read_env_value CUDA_VARIANT)
case "$cuda_variant" in
    cu130)
        expected_cuda_version=13.0.2
        minimum_driver_version=580.95.05
        ;;
    cu129)
        expected_cuda_version=12.9.1
        minimum_driver_version=575.57.08
        ;;
    *)
        echo "CUDA_VARIANT 必须显式设为 cu130 或 cu129" >&2
        exit 1
        ;;
esac

driver_version_has_valid_format() {
    version=$1
    major=${version%%.*}
    remainder=${version#*.}
    [ "$remainder" != "$version" ] || return 1
    minor=${remainder%%.*}
    patch=${remainder#*.}
    [ "$patch" != "$remainder" ] || return 1
    case "$patch" in
        *.*) return 1 ;;
    esac
    for component in "$major" "$minor" "$patch"; do
        case "$component" in
            "" | *[!0-9]*) return 1 ;;
        esac
    done
}

# 一次读取每张卡的索引和驱动版本，逐行严格校验。虽然正常主机的
# 各 GPU 共用同一内核驱动，仍不能只检查第一行后就放行。
if ! gpu_inventory=$(nvidia-smi --query-gpu=index,driver_version --format=csv,noheader,nounits); then
    echo "nvidia-smi 无法读取 GPU 驱动信息" >&2
    exit 1
fi
if [ -z "$gpu_inventory" ]; then
    echo "nvidia-smi 未返回任何 GPU" >&2
    exit 1
fi

detected_gpu_count=0
while IFS= read -r gpu_row; do
    case "$gpu_row" in
        *,*) ;;
        *)
            echo "nvidia-smi 返回了格式异常的 GPU 驱动行" >&2
            exit 1
            ;;
    esac
    gpu_index=${gpu_row%%,*}
    driver_version=${gpu_row#*,}
    # CSV 分隔符周围的空白是 nvidia-smi 的正常输出；中间空白不能
    # 用全局删除“修复”，否则 580. 95.05 之类异常值会被放行。
    gpu_index=$(printf '%s' "$gpu_index" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    driver_version=$(printf '%s' "$driver_version" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    case "$gpu_index" in
        "" | *[!0-9]*)
            echo "nvidia-smi 返回了格式异常的 GPU 索引" >&2
            exit 1
            ;;
    esac
    case "$driver_version" in
        *,*)
            echo "GPU $gpu_index 的驱动版本格式异常" >&2
            exit 1
            ;;
    esac
    if ! driver_version_has_valid_format "$driver_version"; then
        echo "GPU $gpu_index 的驱动版本格式异常：$driver_version" >&2
        exit 1
    fi
    if ! version_at_least "$driver_version" "$minimum_driver_version"; then
        echo "GPU $gpu_index 驱动 $driver_version 低于 CUDA_VARIANT=$cuda_variant 所需的 $minimum_driver_version" >&2
        exit 1
    fi
    detected_gpu_count=$((detected_gpu_count + 1))
done <<EOF
$gpu_inventory
EOF

configured_gpu_count=$(read_env_value GPU_COUNT)
configured_gpu_count=${configured_gpu_count:-4}
case "$configured_gpu_count" in
    "" | *[!0-9]* | 0)
        echo "GPU_COUNT 必须是正整数" >&2
        exit 1
        ;;
esac
if [ "$configured_gpu_count" != "$detected_gpu_count" ]; then
    echo "GPU_COUNT=$configured_gpu_count，但 nvidia-smi 检测到 $detected_gpu_count 张卡" >&2
    exit 1
fi

docker info >/dev/null
compose_version=$(docker compose version --short)
engine_version=$(docker version --format '{{.Server.Version}}')

if ! version_at_least "$compose_version" "2.33.1"; then
    echo "Docker Compose $compose_version 过旧，最低需要 2.33.1" >&2
    exit 1
fi
if ! version_at_least "$engine_version" "28.0.0"; then
    echo "Docker Engine $engine_version 过旧，最低需要 28.0.0" >&2
    exit 1
fi
docker compose --env-file "$env_file" -f "$compose_file" config --quiet
# Compose 规定调用进程环境优先于 --env-file；把含密钥的最终 JSON 只经内存管道
# 交给校验器，确认 TLS/CORS/存储等关键值没有被 shell 环境静默覆盖。
docker compose --env-file "$env_file" -f "$compose_file" config --format json \
    | python3 "$script_dir/validate-rendered-config.py" "$env_file"

environment=$(read_env_value ENVIRONMENT)
environment=${environment:-production}

control_subnet=$(read_env_value CONTROL_SUBNET)
control_subnet=${control_subnet:-172.30.10.0/24}
web_proxy_ip=$(read_env_value WEB_PROXY_IP)
web_proxy_ip=${web_proxy_ip:-172.30.10.10}
trusted_proxy_cidrs=$(read_env_value TRUSTED_PROXY_CIDRS)
trusted_proxy_cidrs=${trusted_proxy_cidrs:-172.30.10.10/32}
python3 - "$control_subnet" "$web_proxy_ip" "$trusted_proxy_cidrs" <<'PY'
from ipaddress import ip_address, ip_network
import sys

try:
    control = ip_network(sys.argv[1], strict=True)
    proxy = ip_address(sys.argv[2])
    trusted = ip_network(sys.argv[3], strict=True)
except ValueError as exc:
    raise SystemExit(f"控制网络或可信代理配置无效：{exc}") from exc
expected = ip_network(f"{proxy}/{proxy.max_prefixlen}", strict=True)
if control.version != 4 or proxy not in control:
    raise SystemExit("WEB_PROXY_IP 必须是 CONTROL_SUBNET 内的 IPv4 地址")
if trusted != expected:
    raise SystemExit("TRUSTED_PROXY_CIDRS 必须只包含 WEB_PROXY_IP 的 /32 地址")
PY

# 先做引用策略校验：拒绝上游受影响版本、latest 和生产环境可变标签。
training_images=$(read_env_value LLAMAFACTORY_ALLOWED_IMAGES)
training_images=${training_images:-openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1}
training_images=$(PYTHONPATH="$project_root/agent" python3 -m openllmops_agent.image_policy "$training_images")

remaining_images=$training_images
while [ -n "$remaining_images" ]; do
    case "$remaining_images" in
        *,*)
            training_image=${remaining_images%%,*}
            remaining_images=${remaining_images#*,}
            ;;
        *)
            training_image=$remaining_images
            remaining_images=
            ;;
    esac
    require_production_digest "$environment" "LLAMAFACTORY_ALLOWED_IMAGES" "$training_image"
    if ! docker image inspect "$training_image" >/dev/null 2>&1; then
        echo "训练镜像尚未构建或拉取：$training_image" >&2
        echo "请先构建安全镜像，或从内部仓库按 digest 拉取已审计镜像" >&2
        exit 1
    fi

    advisory_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.security.ghsa-mwc7-mf87-v3mf" }}' "$training_image")
    remote_code_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.security.trust-remote-code" }}' "$training_image")
    revision_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$training_image")
    runner_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.runner" }}' "$training_image")
    artifact_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.artifacts" }}' "$training_image")
    if [ "$advisory_label" != "mitigated" ] || [ "$remote_code_label" != "disabled" ] || \
        [ "$revision_label" != "c4e09c7cbe18844816af9e18a97fe465515edbcd" ] || \
        [ "$runner_label" != "training-wrapper-v1" ] || \
        [ "$artifact_label" != "safetensors-validated-v1" ]; then
        echo "训练镜像缺少可信安全标签，拒绝启动：$training_image" >&2
        exit 1
    fi
done

# vLLM 仅允许已审计的 0.27.1 固定变体或仓库 digest，且必须由管理员预拉取。
validate_image_cuda_version() {
    image_reference=$1
    image_cuda_version=$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$image_reference" \
        | sed -n 's/^CUDA_VERSION=//p')
    if [ "$image_cuda_version" != "$expected_cuda_version" ]; then
        echo "镜像 $image_reference 的 CUDA_VERSION 与 CUDA_VARIANT=$cuda_variant 不一致；期望 $expected_cuda_version" >&2
        exit 1
    fi
}

vllm_images=$(read_env_value VLLM_ALLOWED_IMAGES)
vllm_images=${vllm_images:-vllm/vllm-openai:v0.27.1}
vllm_images=$(PYTHONPATH="$project_root/agent" python3 -m openllmops_agent.vllm_image_policy "$vllm_images")
remaining_images=$vllm_images
while [ -n "$remaining_images" ]; do
    case "$remaining_images" in
        *,*)
            vllm_image=${remaining_images%%,*}
            remaining_images=${remaining_images#*,}
            ;;
        *)
            vllm_image=$remaining_images
            remaining_images=
            ;;
    esac
    if ! docker image inspect "$vllm_image" >/dev/null 2>&1; then
        echo "vLLM 镜像尚未拉取：$vllm_image" >&2
        echo "请由管理员先按固定版本或 digest 拉取，任务不会隐式拉取镜像" >&2
        exit 1
    fi
    require_production_digest "$environment" "VLLM_ALLOWED_IMAGES" "$vllm_image"
    validate_image_cuda_version "$vllm_image"
done

# 评测镜像的构建基础引用与运行白名单分开配置，因此也要单独
# 核对。否则一次重建可能把 cu129 评测镜像静默换回 cu130。
evaluation_vllm_base_image=$(read_env_value EVALUATION_VLLM_BASE_IMAGE)
evaluation_vllm_base_image=${evaluation_vllm_base_image:-vllm/vllm-openai:v0.27.1}
evaluation_vllm_base_image=$(PYTHONPATH="$project_root/agent" python3 -m openllmops_agent.vllm_image_policy "$evaluation_vllm_base_image")
if ! docker image inspect "$evaluation_vllm_base_image" >/dev/null 2>&1; then
    echo "评测基础 vLLM 镜像尚未拉取：$evaluation_vllm_base_image" >&2
    exit 1
fi
require_production_digest "$environment" "EVALUATION_VLLM_BASE_IMAGE" "$evaluation_vllm_base_image"
validate_image_cuda_version "$evaluation_vllm_base_image"

# 评测镜像除引用白名单外，还要核对构建标签，防止同名镜像替换执行器。
evaluation_images=$(read_env_value EVALUATION_ALLOWED_IMAGES)
evaluation_images=${evaluation_images:-openllmops/evaluation:0.1.0-vllm0.27.1}
evaluation_images=$(PYTHONPATH="$project_root/agent" python3 -m openllmops_agent.evaluation_image_policy "$evaluation_images")
remaining_images=$evaluation_images
while [ -n "$remaining_images" ]; do
    case "$remaining_images" in
        *,*)
            evaluation_image=${remaining_images%%,*}
            remaining_images=${remaining_images#*,}
            ;;
        *)
            evaluation_image=$remaining_images
            remaining_images=
            ;;
    esac
    if ! docker image inspect "$evaluation_image" >/dev/null 2>&1; then
        echo "评测镜像尚未构建或拉取：$evaluation_image" >&2
        exit 1
    fi
    require_production_digest "$environment" "EVALUATION_ALLOWED_IMAGES" "$evaluation_image"
    runner_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.runner" }}' "$evaluation_image")
    remote_code_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.security.trust-remote-code" }}' "$evaluation_image")
    base_label=$(docker image inspect --format '{{ index .Config.Labels "com.openllmops.base.vllm" }}' "$evaluation_image")
    if [ "$runner_label" != "evaluation-pair-v1" ] || [ "$remote_code_label" != "disabled" ] || \
        [ "$base_label" != "v0.27.1" ]; then
        echo "评测镜像缺少可验证的顺序运行/禁用远程代码/vLLM 版本标签：$evaluation_image" >&2
        exit 1
    fi
    validate_image_cuda_version "$evaluation_image"
done

docker_runtimes=$(docker info --format '{{json .Runtimes}}')
case "$docker_runtimes" in
    *nvidia*) ;;
    *)
        echo "Docker 尚未注册 nvidia runtime，请检查 NVIDIA Container Toolkit" >&2
        exit 1
        ;;
esac

storage_root=$(read_env_value OPENLLMOPS_STORAGE_ROOT)
storage_root=${storage_root:-/srv/openllmops}
case "$storage_root" in
    /*) ;;
    *)
        echo "OPENLLMOPS_STORAGE_ROOT 必须是绝对路径" >&2
        exit 1
        ;;
esac
if [ "$storage_root" = "/" ] || [ "$storage_root" = "$HOME" ]; then
    echo "拒绝把系统根目录或用户主目录作为存储根目录" >&2
    exit 1
fi

app_uid=$(read_env_value APP_UID)
app_uid=${app_uid:-1000}
app_gid=$(read_env_value APP_GID)
app_gid=${app_gid:-1000}
workload_uid=$(read_env_value WORKLOAD_UID)
workload_uid=${workload_uid:-1000}
workload_gid=$(read_env_value WORKLOAD_GID)
workload_gid=${workload_gid:-1000}
if [ "$app_uid:$app_gid" != "$workload_uid:$workload_gid" ]; then
    echo "APP_UID:APP_GID 必须与 WORKLOAD_UID:WORKLOAD_GID 一致，否则评测容器无法读取合并数据或写入产物" >&2
    exit 1
fi

python3 "$script_dir/check-storage-permissions.py" "$storage_root" "$app_uid" "$app_gid"

models_device=$(stat -c '%d' "$storage_root/models")
staging_device=$(stat -c '%d' "$storage_root/model-staging")
if [ "$models_device" != "$staging_device" ]; then
    echo "models 与 model-staging 必须位于同一文件系统，才能原子入库" >&2
    exit 1
fi

secret_source=$(read_env_value MODEL_SOURCE_SECRETS_DIR)
secret_source=${secret_source:-./secrets/model-sources}
case "$secret_source" in
    /*) secret_source_dir=$secret_source ;;
    *) secret_source_dir="$deploy_dir/$secret_source" ;;
esac
if [ ! -d "$secret_source_dir" ] || [ -L "$secret_source_dir" ]; then
    echo "模型仓库令牌目录必须是非软链接目录：$secret_source_dir" >&2
    exit 1
fi

validate_model_token_file() {
    variable_name=$1
    container_path=$(read_env_value "$variable_name")
    [ -n "$container_path" ] || return 0
    case "$container_path" in
        /run/secrets/model-sources/*)
            file_name=${container_path#/run/secrets/model-sources/}
            ;;
        *)
            echo "$variable_name 必须位于 /run/secrets/model-sources/" >&2
            exit 1
            ;;
    esac
    case "$file_name" in
        "" | "." | ".." | */*)
            echo "$variable_name 只能指向令牌目录内的直接普通文件" >&2
            exit 1
            ;;
    esac
    host_file="$secret_source_dir/$file_name"
    if [ ! -f "$host_file" ] || [ -L "$host_file" ]; then
        echo "$variable_name 对应的宿主机文件不存在、不是普通文件或是软链接" >&2
        exit 1
    fi
    permissions=$(LC_ALL=C stat -c '%A' "$host_file")
    case "$permissions" in
        *w*)
            echo "$variable_name 对应文件仍有写权限；请执行 chmod 0444 $host_file" >&2
            exit 1
            ;;
    esac
}

validate_model_token_file HUGGINGFACE_TOKEN_FILE
validate_model_token_file MODELSCOPE_TOKEN_FILE

echo "预检通过：GPU=$detected_gpu_count，storage=$storage_root，推理/训练/评测镜像与 Compose 配置有效"
