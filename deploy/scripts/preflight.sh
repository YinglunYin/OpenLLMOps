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

for command_name in docker nvidia-smi python3 stat; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "缺少命令：$command_name" >&2
        exit 1
    fi
done

docker info >/dev/null
compose_version=$(docker compose version --short)
engine_version=$(docker version --format '{{.Server.Version}}')

version_at_least() {
    current=$1
    minimum=$2
    first=$(printf '%s\n%s\n' "$minimum" "$current" | sort -V | head -n 1)
    [ "$first" = "$minimum" ]
}

if ! version_at_least "$compose_version" "2.33.1"; then
    echo "Docker Compose $compose_version 过旧，最低需要 2.33.1" >&2
    exit 1
fi
if ! version_at_least "$engine_version" "28.0.0"; then
    echo "Docker Engine $engine_version 过旧，最低需要 28.0.0" >&2
    exit 1
fi
docker compose --env-file "$env_file" -f "$compose_file" config --quiet

read_env_value() {
    key=$1
    sed -n "s/^${key}=//p" "$env_file" | tail -n 1
}

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
done

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
done

configured_gpu_count=$(read_env_value GPU_COUNT)
configured_gpu_count=${configured_gpu_count:-4}
detected_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [ "$configured_gpu_count" != "$detected_gpu_count" ]; then
    echo "GPU_COUNT=$configured_gpu_count，但 nvidia-smi 检测到 $detected_gpu_count 张卡" >&2
    exit 1
fi

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
