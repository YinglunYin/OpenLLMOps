# OpenLLMOps 单机部署

控制面由 PostgreSQL、FastAPI、Vue/Nginx、Prometheus、DCGM Exporter 和 node-agent 组成。推理与训练工作负载由 node-agent 按需创建，不在 Compose 中预先固化。

## 准备

宿主机需要 Ubuntu 22.04、NVIDIA 驱动、Docker Engine 28+、Docker Compose 2.33.1+ 和 NVIDIA Container Toolkit。Compose 的最低版本用于显式指定 API 容器的外联默认网关。正式机按 4 张 RTX 4090D 配置，开发机可将 `GPU_COUNT` 改为 `2`。

```bash
cp deploy/.env.example deploy/.env
sudo install -d -o 1000 -g 1000 \
  /srv/openllmops/{models,inbox,model-staging,upload-tmp,datasets,evaluation-datasets,evaluation-output,checkpoints,training-configs,runtime}
chmod 0600 deploy/.env
```

在已安装 `argon2-cffi` 的 Python 环境中运行 `python3 scripts/generate_secrets.py`。脚本会在终端交互读取管理员密码，并生成六个互不复用的值；把输出填入 `.env`。其中 `ADMIN_PASSWORD_HASH` 必须保留单引号，否则 Argon2 哈希里的 `$` 会被 Compose 当成变量插值。脚本不写文件，真实密码、哈希和密钥都不要提交到 Git。

生产环境还需要让 `TLS_COMMON_NAME` 与实际 DNS 域名一致，并把 `CORS_ORIGINS` 配成包含同一个浏览器 HTTPS Origin 的明确列表。首版固定对外 HTTPS 端口为 443，预检会拒绝示例域名、IP 形式 CN、非 443 端口和二者不匹配的配置。`WEB_PROXY_IP` 必须位于 `CONTROL_SUBNET`，`TRUSTED_PROXY_CIDRS` 则必须是该地址对应的 `/32`；API 只信任这一台 Nginx 的转发头，不信任控制网络中的其他容器。

## 私有模型仓库凭证

公开 Hugging Face/ModelScope 仓库不需要令牌。需要私有仓库时，把令牌分别写入宿主机普通文件，不要写入 `.env`、接口参数或日志：

```bash
mkdir -p deploy/secrets/model-sources
# 使用本地编辑器创建所需文件，然后移除所有写权限。
chmod 0444 deploy/secrets/model-sources/huggingface.token
chmod 0444 deploy/secrets/model-sources/modelscope.token
```

再在 `.env` 中按需配置容器内的固定只读路径：

```dotenv
HUGGINGFACE_TOKEN_FILE=/run/secrets/model-sources/huggingface.token
MODELSCOPE_TOKEN_FILE=/run/secrets/model-sources/modelscope.token
```

未使用的项保持空值。后端会拒绝软链接、可写文件、超大文件和相对路径；预检脚本也会在启动前检查宿主机映射。人工拷入的离线模型放在 `/srv/openllmops/inbox`，在线下载先进入 `/srv/openllmops/model-staging`，校验成功后再原子移入 models；`model-staging` 和 `models` 必须位于同一文件系统。

浏览器上传的 multipart 数据会先由 Starlette 暂存到 `/srv/openllmops/upload-tmp`，再逐行校验并原子写入 `datasets`。该目录必须位于实际数据盘且由 `APP_UID:APP_GID` 读写，不能映射到内存 `/tmp`；它是临时区，不应纳入备份。

## vLLM 与评测镜像

推理和评测默认使用 vLLM `v0.27.1`。节点不会根据任务请求隐式拉取镜像，首次部署先预拉取推理基础镜像，再构建项目评测镜像：

```bash
docker pull vllm/vllm-openai:v0.27.1
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  --profile runtime-image build evaluation-runtime-image
```

官方 `v0.27.1` amd64 digest 为 `sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2`。正式环境必须在拉取后校验并将 `VLLM_ALLOWED_IMAGES` 与 `EVALUATION_VLLM_BASE_IMAGE` 改为完整 `repository@sha256:...` 引用，评测镜像也必须推送内部仓库并把 `EVALUATION_ALLOWED_IMAGES` 改成 digest；预检会拒绝生产配置中的可变 tag。开发机可显式设置 `ENVIRONMENT=development` 使用固定 tag。

`CUDA_VARIANT` 必须与镜像一起选择，不允许留空或自动猜测：

| `CUDA_VARIANT` | vLLM 容器 CUDA | Linux 宿主驱动下限 | 用途 |
| --- | --- | --- | --- |
| `cu130` | 13.0.2 | `580.95.05` | 默认 |
| `cu129` | 12.9.1 | `575.57.08` | 无法升级到 R580 时的回退 |

下限对应 NVIDIA [CUDA 13.0 Update 2](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html#cuda-driver) 与 [CUDA 12.9 Update 1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver) 的官方 Linux x86_64 驱动表。项目不依赖仅保证部分功能的 CUDA 小版本兼容下限，因为 vLLM/FlashInfer 会使用 JIT 编译内核。预检会读取每张 GPU 的 `driver_version`，任何一张低于下限或版本格式异常都会中止；同时会核对推理、评测基础和已构建评测镜像内的 `CUDA_VERSION`。

使用 cu129 时必须同时修改三项，再重建评测镜像：

```dotenv
CUDA_VARIANT=cu129
VLLM_ALLOWED_IMAGES=vllm/vllm-openai:v0.27.1-cu129
EVALUATION_VLLM_BASE_IMAGE=vllm/vllm-openai:v0.27.1-cu129
```

cu129 官方 amd64 digest 为 `sha256:6666717cd1cadf9adfff8abec9c3f2eca6e27e742de06fe7d7f129fa3d647732`。不要回退到旧 vLLM 版本。宿主只安装 NVIDIA 内核驱动和 NVIDIA Container Toolkit；CUDA 运行时、cuBLAS/NCCL 等用户态库由固定容器镜像提供，宿主无需安装 CUDA Toolkit。

评测镜像使用 `openllmops-eval run-pair`，在同一整卡组上完全停止基线实例后才启动候选实例。构建结果要推送内部仓库时，必须保留 `com.openllmops.runner=evaluation-pair-v1`、`com.openllmops.security.trust-remote-code=disabled` 和 `com.openllmops.base.vllm=v0.27.1` 标签，并把 `EVALUATION_ALLOWED_IMAGES` 改为仓库返回的 digest。

内置数据按准备器的固定布局放置：`evaluation-datasets/ceval/ceval.jsonl` 与 `ceval.manifest.json`，以及 `evaluation-datasets/cmmlu/cmmlu.jsonl` 与 `cmmlu.manifest.json`。两者的输出过程、非商业许可确认和离线准备方式见 `evaluation/README.md`。agent 会对每个来源计算 SHA-256，多数据集按 name/path 排序合并，并将容器报告指纹绑定到调度时生成的 manifest。

## 训练镜像安全基线

截至 2026-08-24，上游[最新正式版 `v0.9.5`](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.9.5) 仍受 [GHSA-mwc7-mf87-v3mf / CVE-2026-58116](https://github.com/advisories/GHSA-mwc7-mf87-v3mf) 影响，没有可直接采用的官方修复版本。上游 [`c4e09c7cbe18844816af9e18a97fe465515edbcd`](https://github.com/hiyouga/LLaMA-Factory/commit/c4e09c7cbe18844816af9e18a97fe465515edbcd) 源码标记为 `0.9.6.dev0`，审计时仍包含公告涉及的硬编码 `trust_remote_code=True` 路径，因此禁止直接使用任何 `hiyouga/llamafactory` 镜像或 `latest`。

项目提供固定安全衍生版 `openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1`。它以官方 OCI index digest 为基础，并在构建期校验七个相关源码文件的 SHA-256、永久禁用远程模型代码，再写入安全标签；任何上游源码漂移或补丁匹配数量变化都会让构建失败。该名称不是上游发行版本。

首次部署先显式构建训练运行时：

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  --profile runtime-image build llamafactory-secure-image evaluation-runtime-image
```

生产环境应将构建结果推送到内部仓库，取得 registry 返回的 manifest digest：

```bash
docker tag openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1 \
  registry.internal/openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1
docker push registry.internal/openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1
# 将下列 64 个 0 替换为 registry 返回的真实摘要。
docker pull registry.internal/openllmops/llamafactory-secure@sha256:0000000000000000000000000000000000000000000000000000000000000000
```

随后把 `LLAMAFACTORY_ALLOWED_IMAGES` 改为上述完整 digest 引用。`LLAMAFACTORY_SECURE_IMAGE` 只是本地构建输出 tag，不要把 tag 当作生产白名单。镜像在内部仓库复制时必须保留以下标签：

- `com.openllmops.security.ghsa-mwc7-mf87-v3mf=mitigated`
- `com.openllmops.security.trust-remote-code=disabled`
- `org.opencontainers.image.revision=c4e09c7cbe18844816af9e18a97fe465515edbcd`
- `com.openllmops.runner=training-wrapper-v1`
- `com.openllmops.artifacts=safetensors-validated-v1`

训练请求仅接受 `template`、`num_train_epochs`、`learning_rate`、`cutoff_len`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`logging_steps`、`save_steps`、`warmup_ratio`、`lora_rank`、`lora_alpha`、`lora_dropout`、`freeze_trainable_layers`、`max_samples` 和 `seed`；未知键、字符串伪装的数字和非有限数会在 API 与节点两侧拒绝。输出固定为 `checkpoints/<训练任务 UUID>`，不能由调用方选择任意目录。

安全镜像入口是 `openllmops-training-runtime`，不会启动 WebUI。它使用参数数组执行 `llamafactory-cli train`，保持 Hugging Face、Datasets 和 Transformers 离线；多卡任务把 `NPROC_PER_NODE` 固定为租约整卡数并使用单机 torchrun。训练成功后会递归移除 Trainer 生成的 pickle optimizer/scheduler/RNG 状态，把 checkpoint 降格为可安全导出但不可恢复优化器的 Safetensors 快照；随后 LoRA/QLoRA 在同一容器中以未量化基础模型合并到 `output/merged`，Freeze 的 `output` 本身即完整模型。只有通过非链接普通文件、Safetensors、模型/adapter 配置与 tokenizer 载荷检查的目录才会上报控制面。容器收到 SIGTERM/SIGINT 时会转发到整个 torchrun 进程组、等待有限时间后强制清理，且不会继续合并或伪报成功。

启动前执行只读预检，确认训练镜像、GPU 数量、NVIDIA Container Toolkit、受控目录对实际 `APP_UID/WORKLOAD_UID` 的读写权限、代理网段、可选令牌文件和 Compose 配置一致：

```bash
sh deploy/scripts/preflight.sh deploy/.env
```

## 启动与检查

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml config
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
curl --insecure https://127.0.0.1/_gateway/health
```

首次联调默认生成 30 天自签名证书，并在生成后把文件所有者设置为 `.env` 的 `APP_UID:APP_GID`，保证后续非 root 预检仍能读取。正式环境应把组织 CA 签发的非软链接普通文件 `tls.crt` 和 `tls.key` 放入 `TLS_DIR`，分别设置为 `0644`、`0600`，再设置 `TLS_AUTO_GENERATE=false` 后重建 Web 容器。TLS 目录不能被组或其他用户写入；Web 容器仅为这个唯一 bind mount 保留 `DAC_OVERRIDE`，从而可读取宿主机管理员拥有的 `0600` 私钥。预检会用 OpenSSL 校验证书至少还有 24 小时有效期、DNS 名称、非交互私钥格式和密钥对匹配关系。若从旧版本升级后 key 仍属于 root，先执行 `sudo chown "$APP_UID:$APP_GID" tls.key tls.crt`（用实际数字替换变量）再预检。

## 安全边界

- Web 是唯一映射到宿主机的业务服务；PostgreSQL、Prometheus、node-agent 和 Docker socket 代理均只在内部网络中可见。
- node-agent 的工作负载命令采用带时间戳和 nonce 的双向 HMAC；请求与响应都签名，并持久化 request_id/generation 水位以拒绝重放和迟到命令。旧的 token 直传写端点已移除。
- 推理、训练和评测镜像都必须预先存在于节点并命中白名单；训练/评测还会校验构建标签。启动时统一转为不可变 image ID，避免校验后的 tag 替换。
- 训练和评测容器均断网；训练模型、数据与配置固定路径只读挂载，训练输出只能写入 `checkpoints/<job UUID>`，评测输出只能写入 `evaluation-output/<run UUID>`。
- 推理参数采用白名单，并永久拒绝 `trust_remote_code`、宿主机网络、特权容器和任意挂载。
- Docker socket 由只开放必要端点的代理隔离；它仍是高权限组件，应限制 Compose 文件和 `.env` 的宿主机访问权限。
- 动态容器以非 root 用户、只读根文件系统、全部 capabilities 丢弃和 `no-new-privileges` 运行。模型/数据只读，checkpoint 与任务缓存按任务单独可写。

## 停止

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml stop
```

`stop` 会保留 runtime 网络和动态工作负载容器。若要彻底执行 `down`，应先在界面停止并删除所有推理/训练/评测容器；否则 Docker 会因动态容器仍连接 runtime 网络而拒绝删除该网络。数据库与 Prometheus 数据卷默认保留，不要在未备份时添加 `--volumes`。
