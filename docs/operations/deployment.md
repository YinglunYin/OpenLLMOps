# 单机部署手册

## 1. 主机前置条件

目标主机为 Ubuntu 22.04、4 × RTX 4090D、128 GB 内存和 2 TB 本地磁盘。开发机可把 `GPU_COUNT` 降为 2，其他镜像与配置保持一致。

最终生产主机的逐项执行与证据要求见[《4 卡裸金属现场验收手册》](bare-metal-acceptance.md)。该清单只有在真实主机执行并签字后才代表通过。

OpenLLMOps 的 node-agent 必须通过真实宿主 Docker daemon 创建、检查和删除动态 GPU 容器，并把宿主受控目录按相同绝对路径挂载给这些容器。因此部署形态是硬边界：

- 支持完整 GPU 虚拟机或裸金属主机。
- 不支持普通 GPU 容器实例；即使实例内 `nvidia-smi` 正常，也不代表能够安全创建同级 GPU 容器。禁止用 Docker-in-Docker、共享平台宿主 Docker socket 或特权容器绕过这一边界。

主机需要：

- 受支持的 NVIDIA 内核驱动；主机不必安装 CUDA Toolkit。CUDA 运行时、cuBLAS/NCCL 等用户态库由固定容器镜像提供，NVIDIA Container Toolkit 把宿主驱动能力安全暴露给容器。
- Docker Engine 与 Compose 2.33.1 或更高版本。
- NVIDIA Container Toolkit，并确认普通测试容器可以枚举全部 GPU。
- 一个只用于 OpenLLMOps 的存储根目录，不能把 `/`、用户主目录或未展开的环境变量直接作为根目录。
- 内网 DNS 名称和由组织 CA 签发的 TLS 证书；自签名证书仅用于开发验证。

正式安装前运行：

```bash
sudo install -d -o 1000 -g 1000 \
  /srv/openllmops/{models,inbox,model-staging,upload-tmp,datasets,evaluation-datasets,evaluation-output,checkpoints,training-configs,runtime}
cp deploy/.env.example deploy/.env
chmod 0600 deploy/.env
```

随后编辑 `deploy/.env`，替换所有占位密钥并确认 `OPENLLMOPS_STORAGE_ROOT=/srv/openllmops`。密钥不应通过聊天、工单或命令行参数传递；优先使用密码管理器生成并通过只读 secret 文件注入。

上述十个子目录是当前版本的 canonical 受控目录。不要另建含义相近的 `staging` 或 `artifacts` 并期待 Compose 使用它们。PostgreSQL 与 Prometheus 使用 Compose named volume；TLS 和可选模型源令牌位于 `deploy/secrets`，不属于 `OPENLLMOPS_STORAGE_ROOT`。

## 2. 镜像与 GPU 版本

- 禁止使用 `latest`。vLLM、LLaMA-Factory、Node Agent、DCGM Exporter 和 PostgreSQL 都应固定版本；正式环境进一步固定镜像 digest。
- 更新 NVIDIA 驱动或 CUDA 基础镜像前，先在 2 卡开发机完成 vLLM 单卡、双卡张量并行、Embedding 和 LLaMA-Factory QLoRA 冒烟测试。
- RTX 4090D 没有 NVLink。多卡任务仍可能通过 PCIe/NCCL 工作，但必须显式测试；遇到 P2P 不支持时使用经验证的 NCCL 设置，不能在未测试的生产环境临时试错。
- LLaMA-Factory 镜像必须通过已知漏洞预检。受影响版本和 `latest` 会被部署脚本拒绝。

当前已审计的 vLLM `v0.27.1` 只有两种允许变体：

| `CUDA_VARIANT` | 容器 CUDA | Linux 宿主驱动下限 | 镜像 tag |
| --- | --- | --- | --- |
| `cu130` | 13.0.2 | `580.95.05` | `vllm/vllm-openai:v0.27.1` |
| `cu129` | 12.9.1 | `575.57.08` | `vllm/vllm-openai:v0.27.1-cu129` |

这两个下限分别来自 NVIDIA [CUDA 13.0 Update 2](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html#cuda-driver) 和 [CUDA 12.9 Update 1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver) 的官方驱动表。预检逐张 GPU 读取 `index,driver_version`，因此不会在第一张卡合格时忽略后续异常行。项目不依赖具有 JIT/新特性限制的 CUDA 小版本兼容下限。

默认使用 `CUDA_VARIANT=cu130`。若机房暂时只有 R575，需在 `.env` 中同时修改下列三项，并用新基础镜像重建评测镜像：

```dotenv
CUDA_VARIANT=cu129
VLLM_ALLOWED_IMAGES=vllm/vllm-openai:v0.27.1-cu129
EVALUATION_VLLM_BASE_IMAGE=vllm/vllm-openai:v0.27.1-cu129
```

生产环境把 tag 替换为文档中已核验的 amd64 digest，但 `CUDA_VARIANT` 仍保留；预检会通过镜像内的 `CUDA_VERSION` 防止 digest、评测基础镜像或已构建评测镜像与声明变体不一致。

## 3. 镜像准备、TLS、预检与首次启动

把正式证书和私钥分别放到 `deploy/secrets/tls/tls.crt` 与 `tls.key`，将私钥权限设为 `0600`，并设置 `TLS_AUTO_GENERATE=false`。

`preflight.sh` 会检查 vLLM、训练和评测镜像的环境变量与安全标签，因此镜像尚未 pull/build 时运行预检必然失败。必须遵守“渲染配置 → pull/build → preflight → `up --no-build`”顺序，避免预检之后又重建出另一份镜像：

```bash
# 1. 先验证 Compose 可以渲染；该步骤不替代完整预检。
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet

# 2. 拉取固定控制面依赖镜像；动态 vLLM 与评测基础镜像也必须按 .env 中的
#    固定 tag（开发）或 repository@sha256（生产）显式 docker pull。
docker compose --env-file deploy/.env -f deploy/compose.yaml pull \
  postgres prometheus dcgm-exporter docker-socket-proxy
docker pull <VLLM_ALLOWED_IMAGES 中的每个引用>
docker pull <EVALUATION_VLLM_BASE_IMAGE>

# 3. 构建项目控制面和两种受控运行时。生产环境应先推送运行时镜像并把
#    白名单改成 registry 返回的 digest，再执行下一步。
docker compose --env-file deploy/.env -f deploy/compose.yaml build api node-agent web
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  --profile runtime-image build llamafactory-secure-image evaluation-runtime-image

# 4. 镜像全部存在后执行完整只读预检。
sh deploy/scripts/preflight.sh deploy/.env

# 5. 预检后禁止隐式重建，分阶段启动。
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --no-build \
  postgres docker-socket-proxy node-agent prometheus dcgm-exporter
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --no-build api web
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
```

尖括号是操作说明，不可原样复制为 shell 参数。若 `VLLM_ALLOWED_IMAGES` 配置多个逗号分隔引用，必须逐个拉取。生产 digest 与构建/推送流程详见 `deploy/README.md`。

先启动依赖，再启动 API 与 Web，便于在迁移失败时停止在明确阶段。首次启动后检查：

- `https://<内网域名>/_gateway/health` 返回健康。
- `/health/ready` 确认数据库迁移已就绪。
- 资源页显示期望数量的 GPU；DCGM 不支持的消费卡指标应显示“不可用”，不能显示伪造的 0。
- 用一个小模型完成导入、单卡部署、流式 Playground 和停止/再启动。

## 4. 1～2 卡开发与冒烟环境

开发机至少覆盖以下差异：

```dotenv
GPU_COUNT=2
VITE_USE_MOCKS=false
TLS_AUTO_GENERATE=true
```

不要把生产任务配置硬编码为 GPU 2、3。调度 API 会校验设备编号，但表单也应按实时设备列表生成选项。

`GPU_COUNT` 必须等于该主机 `nvidia-smi` 实际枚举的总卡数；它不是“本次任务想使用几张卡”。在 4 卡主机做单卡 smoke 时仍设置 `GPU_COUNT=4`，只在创建任务时选择一张卡。

单卡 smoke 只验收宿主容器能力、控制面、单卡 vLLM、OpenAI 端点和任务生命周期。它不能证明 2/4 卡 Tensor Parallel、NCCL/PCIe 拓扑、整组租约原子性或 4 卡训练。2 卡开发机也不能替代最终 4 × RTX 4090D 无 NVLink 主机验收。

## 5. 停止与重启

普通 `docker compose down` 只管理固定控制面，不能假设动态 vLLM/LLaMA-Factory 容器也会被安全处理。

计划停机顺序：

1. 禁止新任务进入队列。
2. 在界面停止推理部署，并终止或等待训练/评测结束。
3. 确认 Node Agent 不再报告活动工作负载和 GPU 进程。
4. 停止 `web`、`api`、监控和 Node Agent，最后停止 PostgreSQL。
5. 完成数据库和文件快照后，才允许移除固定容器。

控制面异常重启后，期望为运行的推理服务会重新排队恢复；训练和评测不会自动重跑。仍存在的容器会按 generation 对账，已丢失任务需由管理员使用原模型、数据集和配置新建任务，首版不能从 checkpoint 恢复优化器状态。

## 6. 上线验收

- 管理接口无管理员会话/API Key 时返回 401/403。
- `/v1/responses` 返回 404；三个承诺端点均通过 API Key 测试。
- `trust_remote_code`、非 Safetensors 权重、越界目录和软链接导入全部被拒绝。
- 同一张 GPU 的并发启动只有一个成功，整组多卡租约不会部分占用。
- 训练排队不会停止已运行推理；释放 GPU 后队首任务自动启动。
- 在最终 4 × RTX 4090D 主机记录 `nvidia-smi topo -m`，并用适合 TP=4 的代表模型完成 2 卡与 4 卡 NCCL/Tensor Parallel 实测；不得从单卡或其他 GPU 型号结果外推。无 NVLink 是已知验收条件，不是可忽略的性能备注。
- Node Agent HMAC 的错误签名、过期时间戳、重复 nonce 和旧 generation 全部被拒绝。
- 备份恢复演练至少成功一次，并记录恢复所需时间。
