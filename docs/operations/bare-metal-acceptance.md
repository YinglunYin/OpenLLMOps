# 4 卡裸金属现场验收手册

> 状态说明：本文是后续在最终生产主机执行的现场验收清单。本轮只编写和审查手册，**尚未在最终 4 卡裸金属主机执行任何步骤**，因此本文中的项目当前均不能视为已通过。

## 1. 范围、环境与通过规则

本手册只适用于下列最终环境：

- Ubuntu 22.04；
- 4 × NVIDIA RTX 4090D，卡间无 NVLink；
- 128 GB 主机内存；
- 2 TB 本地数据盘；
- 真实宿主 Docker daemon、Docker Compose 与 NVIDIA Container Toolkit；
- `ENVIRONMENT=production`、HTTPS、单管理员认证和推理 API Key 均已启用。

普通 GPU 容器、Docker-in-Docker、共享平台宿主 Docker socket 和 1～2 卡开发机均不能代替本次验收。主机不要求安装 CUDA Toolkit；CUDA 用户态运行时来自固定容器镜像，宿主只提供满足要求的 NVIDIA 驱动。

验收遵守以下规则：

1. 每个必测项记录为 `PASS`、`FAIL` 或 `BLOCKED`，不得把“未执行”“没有观察到”写成 `PASS`。
2. 任一 GPU 未枚举、镜像 digest 不一致、整卡重复分配、推理被训练自动停止、训练/评测容器越过隔离边界或承诺端点不可用，均阻断上线。
3. C-Eval、CMMLU 和自定义领域集不设置通过阈值；任务可复现、结果可比较且百分比计算正确即完成功能验收，分数上升或下降作为结果如实记录。
4. RTX 4090D 不支持 NVLink 是既定条件，不是缺陷；TP=2/4 必须在真实 PCIe/NCCL 路径上成功，不能由单卡结果外推。
5. 所有命令都从仓库已审核的 commit 执行。尖括号内容是占位符，不可原样复制。

配套资料：

- [单机部署手册](deployment.md)
- [部署目录说明](../../deploy/README.md)
- [评测集版本、许可与准备流程](../../evaluation/README.md)
- [故障定位](troubleshooting.md)
- [备份恢复](backup-restore.md)

## 2. 验收准备与证据目录

开始前安排一名操作人和一名复核人，冻结代码、镜像及模型/数据集版本，并预留完整维护窗口。验收使用专用的小规模 CPT、SFT 和领域评测数据，不得在故障注入阶段使用唯一一份生产数据。

在不包含密钥的受控位置建立本次证据目录：

```bash
export ACCEPTANCE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export EVIDENCE_ROOT="/srv/openllmops-acceptance/$ACCEPTANCE_ID"
install -d -m 0700 "$EVIDENCE_ROOT"
git rev-parse HEAD | tee "$EVIDENCE_ROOT/git-commit.txt"
date -u +%FT%TZ | tee "$EVIDENCE_ROOT/started-at.txt"
```

证据目录不得保存以下内容：

- `deploy/.env`、完整的 `docker compose config` 输出或容器的完整 `Config.Env`；
- 管理员密码、bootstrap key、推理 API Key、会话 Cookie、CSRF token、模型仓库 token；
- 自定义领域数据的原文或其他不必要的业务敏感内容。

需要调用 OpenAI 兼容接口时，应在管理界面创建仅用于本次验收的 API Key。明文只出现一次，将其放入受限的临时 shell 变量，不写入脚本、命令行历史或证据；验收结束立即撤销。

在验收记录首页固定以下输入：

| 项目 | 必须记录的值 |
| --- | --- |
| 代码 | commit SHA、分支或 release tag、工作树是否干净 |
| 主机 | 资产编号、Ubuntu 版本、内核、内存、数据盘设备与挂载点 |
| GPU | 4 张卡的 index、UUID、型号、显存、驱动版本 |
| 镜像 | repository、tag、`RepoDigest`、本地 image ID、架构、CUDA 变体 |
| 模型 | 来源、requested/resolved revision、Safetensors 校验和、模型类型 |
| 数据集 | 类型、版本、记录数、SHA-256、许可确认 |
| 配置 | `CUDA_VARIANT`、`GPU_COUNT=4`、TP、显存比例及测试用 vLLM 参数 |

## 3. BM-01：宿主机与 4 卡拓扑

执行并保存原始输出：

```bash
cat /etc/os-release | tee "$EVIDENCE_ROOT/os-release.txt"
uname -a | tee "$EVIDENCE_ROOT/uname.txt"
free -h | tee "$EVIDENCE_ROOT/memory.txt"
df -hT / /srv/openllmops | tee "$EVIDENCE_ROOT/disk.txt"
lscpu | tee "$EVIDENCE_ROOT/lscpu.txt"
nvidia-smi -L | tee "$EVIDENCE_ROOT/nvidia-smi-list.txt"
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id \
  --format=csv,noheader,nounits \
  | tee "$EVIDENCE_ROOT/gpu-inventory.csv"
nvidia-smi topo -m | tee "$EVIDENCE_ROOT/gpu-topology.txt"
docker version | tee "$EVIDENCE_ROOT/docker-version.txt"
docker compose version | tee "$EVIDENCE_ROOT/compose-version.txt"
docker info --format '{{json .Runtimes}}' | tee "$EVIDENCE_ROOT/docker-runtimes.json"
```

通过条件：

- `ID=ubuntu` 且 `VERSION_ID=22.04`；内存规格为 128 GB，允许操作系统计量和保留造成的小幅显示差异；2 TB 十进制磁盘通常约显示为 1.8 TiB，挂载点与采购规格一致且有足够剩余空间。
- `nvidia-smi` 精确返回 GPU `0..3` 四行，每行均为采购的 RTX 4090D SKU（驱动可能显示为 `RTX 4090 D`），UUID 唯一，驱动版本一致。
- `nvidia-smi topo -m` 完整返回四卡 PCIe/NUMA 关系；卡间没有 `NV#` 链路符合预期，必须保留实际 `PIX/PXB/PHB/SYS` 关系供 NCCL 故障分析。
- 驱动满足所选镜像变体：`cu130` 不低于 `580.95.05`，`cu129` 不低于 `575.57.08`。
- Docker Engine 不低于 `28.0.0`，Compose 不低于 `2.33.1`，Docker runtimes 中包含 `nvidia`。
- 这是真实宿主或完整 GPU 虚拟机，Docker daemon 可创建同级容器；若只有容器内 `nvidia-smi` 而无可用 daemon/socket，标记 `BLOCKED`。

## 4. BM-02：镜像、digest 与部署配置冻结

生产环境禁止 `latest` 和动态工作负载的可变 tag。先按[单机部署手册](deployment.md)完成构建、推送和显式拉取，再把下列运行时白名单换成内部仓库返回的 `repository@sha256:...`：

- `VLLM_ALLOWED_IMAGES`；
- `LLAMAFACTORY_ALLOWED_IMAGES`；
- `EVALUATION_VLLM_BASE_IMAGE`；
- `EVALUATION_ALLOWED_IMAGES`。

控制面、PostgreSQL、Prometheus、DCGM Exporter 和 Docker socket proxy 也要记录已批准的 digest。若控制面镜像暂时只在本机构建而没有 `RepoDigest`，至少记录不可变 image ID、Dockerfile、基础镜像 digest 和 Git commit；正式发布清单仍应补齐内部仓库 digest。

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/.env -f deploy/compose.yaml config --images \
  | sort -u | tee "$EVIDENCE_ROOT/compose-images.txt"

# 对清单中的每一个批准镜像分别执行；不要把含逗号的白名单整体作为一个参数。
docker image inspect '<repository@sha256:...>' \
  --format 'id={{.Id}} arch={{.Architecture}} digests={{json .RepoDigests}}' \
  | tee -a "$EVIDENCE_ROOT/image-inventory.txt"
```

使用已存在的批准镜像做一次不拉取的 CUDA 容器检查。该命令属于后续现场验收，本轮未执行：

```bash
docker run --rm --pull=never --gpus all --entrypoint python \
  '<批准的 vLLM repository@sha256:...>' \
  -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; print(torch.version.cuda); [print(i, torch.cuda.get_device_name(i)) for i in range(4)]' \
  | tee "$EVIDENCE_ROOT/vllm-cuda-smoke.txt"
```

通过条件：

- Compose 能成功渲染，但完整渲染结果不落盘，因为其中含密钥。
- 所有镜像为 `linux/amd64`，本地 image ID 与批准清单匹配；动态运行时均以 digest 加入白名单。
- 镜像 `CUDA_VERSION` 与 `CUDA_VARIANT` 一致，vLLM 容器内 PyTorch 精确发现四张 4090D。
- 训练镜像保留安全修复、禁用远程代码、受控 runner 和 Safetensors 产物标签；评测镜像保留顺序运行、禁用远程代码及固定 vLLM 基础版本标签。
- 镜像和配置冻结后不再执行隐式 `pull` 或 `build`。任何重新构建都必须回到本节重新记录 digest 并重跑预检。

## 5. BM-03：Compose 预检与分阶段启动

确认 TLS、独立密钥、只读模型源 token 和十个 canonical 数据目录均已按部署手册准备。`GPU_COUNT` 必须是主机总卡数 `4`，而不是某个任务计划使用的卡数。

```bash
sh deploy/scripts/preflight.sh deploy/.env \
  2>&1 | tee "$EVIDENCE_ROOT/preflight.txt"

docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --no-build \
  postgres docker-socket-proxy node-agent prometheus dcgm-exporter
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --no-build api web

docker compose --env-file deploy/.env -f deploy/compose.yaml ps \
  | tee "$EVIDENCE_ROOT/compose-ps.txt"
docker compose --env-file deploy/.env -f deploy/compose.yaml logs --no-color --tail=300 \
  postgres docker-socket-proxy node-agent prometheus dcgm-exporter api web \
  >"$EVIDENCE_ROOT/startup-logs.txt"
```

用组织 CA 验证 HTTPS，不以 `-k/--insecure` 的结果作为生产验收证据：

```bash
curl --fail-with-body --cacert '<组织 CA 文件>' \
  'https://<内网域名>/_gateway/health' \
  | tee "$EVIDENCE_ROOT/gateway-health.txt"
docker compose --env-file deploy/.env -f deploy/compose.yaml exec -T api python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=5).read().decode())" \
  | tee "$EVIDENCE_ROOT/api-ready.json"
```

Nginx 只把 `/api/*` 和 `/v1/*` 转发给 FastAPI；外部 `/health/ready` 会落入前端 SPA，不能作为 API readiness 证据。因此 readiness 必须从 API 容器内部检查，或以 Compose 的 API health 状态为准。

通过条件：

- `preflight.sh` 最终打印通过，迁移成功，固定服务处于预期的 running/healthy 状态，日志中没有循环重启、鉴权秘密或持续错误。
- 未认证访问 `/api/v1/system/capabilities` 返回 401/403；管理员登录、CSRF 防护和登出正常。
- `/api/v1/system/capabilities` 返回 `gpu_count=4`、`exclusive_non_preemptive`、训练算法清单以及三个承诺的 OpenAI 端点。
- Node Agent 能连接 Docker runtime network。仅 `/healthz` 通过还不够，下一节必须继续验证 NVML、DCGM 和真实工作负载。

## 6. BM-04：四卡监控与 Node Agent

在固定服务刚启动的空闲态，以及后续单卡、TP=2、TP=4、训练和评测运行态，分别保存一次监控证据。可从管理界面或使用管理员认证访问：

- `GET /api/v1/system/gpus`；
- `GET /api/v1/system/gpu-leases`；
- `GET /api/v1/system/gpus/{index}/history`。

不要把 Node Agent token 输出到终端或证据文件。需要核验 Agent 原始 GPU 视图时，在容器内部从环境读取 token：

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml exec -T node-agent python - <<'PY' \
  | tee "$EVIDENCE_ROOT/node-agent-gpus.json"
import os
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:9000/v1/gpus",
    headers={"X-Node-Agent-Token": os.environ["NODE_AGENT_TOKEN"]},
)
with urllib.request.urlopen(request, timeout=5) as response:
    print(response.read().decode("utf-8"))
PY
```

通过条件：

- API、Node Agent 与 `nvidia-smi` 均以相同 index/型号看到四张卡；Node Agent 与 `nvidia-smi` 的 UUID 逐卡一致。
- 空闲态没有租约；任务运行时 `allocated_to`、数据库 owner、generation 与实际容器一致。
- 显存、利用率、温度至少能在负载前后变化；功耗等 4090D/DCGM 不支持的单项可显示“不可用”，但缺失值必须是 `null/unknown` 并带 degraded 原因，不能伪造成 0。
- Prometheus 至少经过三个抓取周期后，四张卡的历史接口均返回带时间戳的数据点；默认抓取周期为 10 秒，建议负载稳定运行 30 秒以上再取证。
- 管理界面同时显示四卡当前状态、租约 owner 和历史曲线，截图包含 UTC 时间。

若 DCGM Exporter 不能发现任何 GPU，或显存/利用率始终不可用，本项为 `FAIL`，不能仅凭 Node Agent 的 NVML 输出放行监控功能。

## 7. BM-05：模型与数据集固定

至少准备三类模型资产：

1. 一个能在单张 4090D 上稳定加载的生成模型，用于单卡、FIFO、Playground 和训练冒烟；
2. 一个代表正式使用规模、且明确支持 TP=2/4 的标准架构生成模型，用于真实 PCIe/NCCL 验证；
3. 一个标准架构 Embedding 模型，用于独立 Embedding 部署。

所有模型必须为 Safetensors，不含 pickle 权重、`auto_map` 或需 `trust_remote_code` 的自定义代码。在线导入必须记录 requested revision、解析后的完整 commit 和平台校验和；人工/SFTP 导入只从受控 inbox 的一层普通目录进行，并记录原始来源与外部校验和。

至少上传三类 JSONL：

- CPT：每行且只能有一个非空 `text` 或 `content` 字段；
- SFT：全文件统一使用 `messages`、`conversations` 或 `instruction` + `output` 中的一种格式；
- 自定义评测：使用稳定唯一 `id`，支持 `multiple_choice`、`classification` 或 `short_qa`，并提供 `category` 与标准 `answer/answers`。

自定义选择题示例：

```json
{"id":"domain-001","task_type":"multiple_choice","category":"法规","question":"示例问题","choices":{"A":"选项一","B":"选项二"},"answer":"A"}
```

通过条件：模型和数据集均为 ready/available；界面可预览 JSONL；资产详情记录 revision、大小和 SHA-256；错误类型、混合行格式、重复评测 ID 和不安全模型会被拒绝。正式质量比较使用的领域集不得与训练集样本重叠。

## 8. BM-06：生成与 Embedding 分别部署

生成与 Embedding 必须创建为两个独立部署，不能把同一服务同时声明成两种任务类型：

| 部署 | 模型类型 | GPU | TP | 必测端点 |
| --- | --- | --- | --- | --- |
| `accept-generation-tp1` | Base 或 Instruct 生成模型 | `[0]` | 1 | completions、chat completions、流式 chat、Playground |
| `accept-embedding-tp1` | Embedding 模型 | `[1]` | 1 | embeddings |

初始建议将 `gpu_memory_utilization` 设为经模型验证的保守值，并显式记录 `max_model_len`、`dtype` 和高级 vLLM 参数。`tensor_parallel_size` 必须始终等于所选 GPU 数量。

待两个部署均为 `actual_state=running`、`health_status=healthy` 后，创建专用推理 API Key 并测试三个承诺端点。下例中的 key 只在当前 shell 变量中保存：

```bash
read -r -s ACCEPTANCE_API_KEY
export ACCEPTANCE_API_KEY
export OPENLLMOPS_URL='https://<内网域名>'
export OPENLLMOPS_CA='<组织 CA 文件>'

curl --fail-with-body --cacert "$OPENLLMOPS_CA" \
  -H "X-API-Key: $ACCEPTANCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"accept-generation","prompt":"用一句话介绍北京。","max_tokens":32}' \
  "$OPENLLMOPS_URL/v1/completions" \
  | tee "$EVIDENCE_ROOT/completions.json"

curl --fail-with-body --cacert "$OPENLLMOPS_CA" \
  -H "X-API-Key: $ACCEPTANCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"accept-generation","messages":[{"role":"user","content":"只回答：正常"}],"max_tokens":16}' \
  "$OPENLLMOPS_URL/v1/chat/completions" \
  | tee "$EVIDENCE_ROOT/chat-completions.json"

curl --fail-with-body --no-buffer --cacert "$OPENLLMOPS_CA" \
  -H "X-API-Key: $ACCEPTANCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"accept-generation","messages":[{"role":"user","content":"逐字输出测试"}],"stream":true,"max_tokens":16}' \
  "$OPENLLMOPS_URL/v1/chat/completions" \
  | tee "$EVIDENCE_ROOT/chat-stream.txt"

curl --fail-with-body --cacert "$OPENLLMOPS_CA" \
  -H "X-API-Key: $ACCEPTANCE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"accept-embedding","input":["第一条文本","第二条文本"]}' \
  "$OPENLLMOPS_URL/v1/embeddings" \
  | tee "$EVIDENCE_ROOT/embeddings.json"
```

另行记录 `POST /v1/responses` 的 HTTP 状态码，期望为 404。本版本没有该端点，不能把它加入健康探针或客户端合同。

通过条件：

- completions 和 chat completions 返回正确的对外模型名、非空文本和用量字段；流式响应逐段到达并以 `[DONE]` 收口；Playground 流式显示、停止生成和错误提示正常。
- embeddings 对两个输入返回两个顺序一致、维度一致、只含有限数值的向量。
- 无 key 和错误 key 返回 401；生成模型不能路由到 embeddings，Embedding 模型不能路由到生成端点。
- 两个服务可同时运行，GPU 0/1 各只有一个租约和一个匹配的动态容器，互不覆盖。
- 停止、再次启动、编辑停止态配置、查看详情和停止后删除均符合状态约束。

## 9. BM-07：单卡、TP=2 与 TP=4 vLLM

先停止上一节两个单卡部署并确认租约释放，再使用同一个固定 revision 的代表生成模型依次执行下列矩阵；每次只保留一个待测部署，避免结果受到并发负载干扰。

| 场景 | `gpu_ids` | `tensor_parallel_size` | 必须记录 |
| --- | --- | --- | --- |
| 单卡 | `[0]` | 1 | 启动时间、显存、健康、非流式/流式结果 |
| 双卡 | `[0,1]` | 2 | NCCL 日志、两卡显存、健康、同一请求结果 |
| 四卡 | `[0,1,2,3]` | 4 | NCCL 日志、四卡显存、健康、同一请求结果 |

若代表模型无法在单卡容纳，可用同架构小模型完成三档功能一致性，再用正式规模模型至少完成其实际需要的 TP=2/4 场景；两组结果必须分开记录，不能宣称大模型已通过未实际执行的档位。

每个场景至少：

1. 轮询到 running/healthy，稳定 30 秒；
2. 调用 completions、非流式 chat 和流式 chat；
3. 保存四卡当前指标、历史点、租约、`nvidia-smi` 和 `nvidia-smi pmon -c 1`；
4. 保存动态容器日志中的 vLLM/NCCL 初始化摘要；
5. 用受限格式检查容器，不保存完整环境变量：

```bash
docker ps --filter label=com.openllmops.managed=true \
  --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}} {{.Labels}}' \
  | tee -a "$EVIDENCE_ROOT/managed-workloads.txt"
```

通过条件：TP 值与 GPU 数量完全一致；同一租约组一次取得全部卡；各卡均出现对应显存占用；没有 CUDA OOM、NCCL hang、P2P 假设错误或反复重启；三个生成测试在三档均成功。吞吐和延迟作为基线记录，不在本手册中预设门槛。

## 10. BM-08：整卡独占、全有或全无、严格 FIFO 与非抢占

### 10.1 推理不被训练自动停止，且队列严格 FIFO

1. 在 GPU 0 启动生成部署 `fifo-inference`，确认 healthy 并持续发送低频健康请求。
2. 创建需要 GPU 0 的短 SFT LoRA 任务 `fifo-training-1`，记下 `queued_at`。
3. 在它之后创建只需要空闲 GPU 3 的部署 `fifo-inference-2`。
4. 至少观察三个 reconciler 周期。

预期结果：

- 原推理始终 running/healthy，容器 ID 不变，没有收到自动 stop。
- `fifo-training-1` 因 GPU 0 被占而保持 queued，没有创建训练容器。
- 严格全局 FIFO 会让更晚的 `fifo-inference-2` 暂时保持 queued，即使 GPU 3 空闲；不得绕过队首。
- 管理员手动停止 `fifo-inference` 后，`fifo-training-1` 先取得 GPU 0 并开始；队首成功出队后，`fifo-inference-2` 才可取得 GPU 3。

### 10.2 多卡租约全有或全无

1. 在 GPU 1 保持一个健康推理部署。
2. 创建请求 `[1,2]`、TP=2 的更早任务。
3. 观察租约表，GPU 2 不应被该任务提前占用。
4. 手动释放 GPU 1 后，任务应以同一个 `lease_group_id` 同时取得 GPU 1/2。

### 10.3 并发竞争

由两个独立客户端尽可能同时提交使用同一张卡的启动请求。预期数据库中该 GPU 始终只有一行租约，只有队首进入 starting/running，另一个保持 queued；不得出现两个活动容器或短暂的部分多卡租约。

本项证据必须包含任务创建/排队/启动时间、每次 `/api/v1/system/gpu-leases` 快照、推理连续健康记录、Agent 工作负载列表和对应容器 ID。

## 11. BM-09：训练运行时矩阵

使用同一个固定基础生成模型、经校验的小规模数据版本和可复现 seed，按顺序执行四个任务。为减少共享 CPU、内存和磁盘对结果的干扰，建议一次只运行一个训练 smoke：

| ID | 阶段 | 算法 | 数据集 | 必须产生/验证的结果 |
| --- | --- | --- | --- | --- |
| TR-01 | CPT | LoRA | CPT JSONL | 成功、Adapter、合并模型、checkpoint 清单 |
| TR-02 | SFT | Freeze | SFT JSONL | 成功、完整 Safetensors 模型、checkpoint 清单 |
| TR-03 | SFT | LoRA | SFT JSONL | 成功、Adapter、合并模型、checkpoint 清单 |
| TR-04 | SFT | QLoRA | SFT JSONL | 4-bit 路径成功、Adapter、合并模型、checkpoint 清单 |

Smoke 参数可采用 1 epoch、较小 `max_samples`、`cutoff_len`、batch size 1 和固定 seed，但每次必须记录完整表单值。为了实际产生可导出的 checkpoint，`save_steps` 应设为 1 或不大于预计总步数。SFT 的 template 必须与基础模型对话格式一致；CPT 固定只能选择 LoRA。不得通过高级参数注入任意 LLaMA-Factory、DeepSpeed、shell 或路径参数。

每个任务的通过条件：

- 状态按 queued → starting → running → succeeded 收敛，`progress=100`，`current_step/total_steps` 合理，训练 loss、learning rate、epoch/step 等已报告指标在详情页可见。
- 任务独占所选整卡；训练容器 `network_mode=none`、非 root、只读根文件系统、丢弃 capabilities，并只看到批准的 GPU。
- 训练期间 GPU 指标和历史曲线有真实变化，任务结束后租约与 GPU 进程释放。
- Freeze 产出可部署完整模型；LoRA/QLoRA 产出 Adapter，受控导出后还有合并的完整 Safetensors 模型。
- `GET /api/v1/training-jobs/{id}/artifacts` 只列出受控 `checkpoint/adapter/merged/full` 产物；下载归档名称、文件数、字节数稳定，不包含 `.pt/.pth/.bin/.pkl/.pickle/.joblib/.ckpt` 优化器或 RNG 反序列化状态。
- 任何失败都保存可读的 `error_message` 和容器日志，不把有产物目录解释成成功。

另建一个可在数分钟内保持 running 的训练任务，点击“终止”。它应经过 canceling 并到达 canceled，最终 checkpoint 仅按 Agent 已登记内容展示，GPU 和容器被释放；首版不支持 resume，也不会自动重跑该任务。

## 12. BM-10：训练后模型发布与部署

选择成功的 SFT LoRA 或 QLoRA 任务执行“发布模型”。记录新模型资产 ID、`training://<job-id>` 来源、基础模型 ID、算法、产物类型、大小和校验和。

通过条件：

- 发布操作幂等，多次点击返回同一资产，不覆盖其他目录；新资产为 Instruct、ready 和 Safetensors。
- 新资产位于模型受控根目录，不直接引用 checkpoint 目录，不含 pickle 或远程自定义代码。
- 使用新资产创建独立生成部署，轮询到 healthy；completions、chat completions、流式 Playground 均成功。
- 停止并再次启动后仍使用相同模型资产和对外模型名；保存发布前后资产详情、部署详情、一次请求与 GPU 租约证据。

CPT 发布模型保持基础模型原有 Base/Instruct 类型；SFT 发布模型登记为 Instruct。现场记录应与这一规则一致。

## 13. BM-11：训练前后量化评测

### 13.1 固定内置数据集

按[评测执行器文档](../../evaluation/README.md)准备完整数据集并确认非商业许可：

| 数据集 | 固定来源 | 本次必须使用的可评分 split |
| --- | --- | --- |
| C-Eval | 官方 Hugging Face revision `3923b519fd180e689d0961bf3a032ece929742f3` | `dev` + `val`；该固定快照的 `test` 没有答案，不得伪造 |
| CMMLU | 官方 GitHub commit `d6e7b716d8ac694f38969a6c0407437d1fded799` | `dev` + `test` |

保存各自 manifest 中的来源 SHA-256、规范化内容 SHA-256、输出 SHA-256、split/科目计数、样本总数、许可和 `partial=false`。正式验收禁止使用 `--allow-partial`。

### 13.2 前后模型与执行方式

创建一个评测任务：

- baseline：训练前的固定模型资产；
- candidate：BM-10 发布的训练后模型资产；
- datasets：C-Eval、CMMLU 和一个不与训练集重叠的自定义领域评测集；
- GPU：选择能够容纳两模型中较大者的同一整卡组，TP 等于 GPU 数量。

执行器必须在同一 GPU 组上先完整运行 baseline，停止其 vLLM 并释放显存后，再运行 candidate；禁止两个模型并行抢占显存。Base 与 Instruct 按资产类型使用各自固定模板，模板名必须写入报告。

通过条件：

- 状态按 queued → starting → running → succeeded 收敛，比较结果 `comparable=true`；baseline 和 candidate 的数据集 SHA-256 与样本 ID 集合完全一致。
- 每个数据集和 category/科目均展示 baseline/candidate 的正确数、总数、无效输出数和准确率百分比；模型运行统计分别展示 baseline/candidate 的总体平均延迟。
- 同时展示百分点变化和相对变化；baseline 为 0 时相对变化必须是未定义/`—`，不能显示 `0%`。
- 非零无效输出数始终可见；baseline 或 candidate 全部输出均无效时，还必须显示对应 warning，不能只用最终准确率掩盖解析失败。
- 评测期间只有一个 evaluation 租约组，显存占用应体现 baseline 释放后 candidate 再加载；完成或取消后 GPU 全部释放。

### 13.3 百分比与灾难性遗忘记录

对每个数据集分别按以下公式复核界面和报告：

```text
baseline 百分比 B = baseline_correct / baseline_total × 100%
candidate 百分比 C = candidate_correct / candidate_total × 100%
绝对变化（百分点） = C - B
相对变化（%） = (C - B) / B × 100%，仅当 B != 0
灾难性遗忘百分比 = max(0, (B - C) / B × 100%)，仅当 B != 0
```

- C-Eval 和 CMMLU 用于观察通用能力：逐数据集、逐科目记录 `B`、`C`、百分点变化、相对变化和灾难性遗忘百分比。
- 自定义领域集用于观察领域能力：逐 category 记录同样指标，正的相对变化表示领域提升。
- `B=0` 时相对变化和灾难性遗忘百分比均记为“未定义”，不能以 0% 替代。
- 可以补充按正确数/总数计算的微平均，但不能用它替代每个数据集及科目明细。
- 本版本不设置“不通过阈值”。即使通用能力下降或领域能力没有提升，只要计算和证据正确，本项功能可记 `PASS`，同时必须把实际数值和风险写入验收结论，由业务方另行决定是否采用该训练模型。

## 14. BM-12：受控故障与恢复

每项故障注入前先保存租约、工作负载和 `nvidia-smi`，一次只注入一个故障：

1. **推理进程异常退出**：先按 owner 标签解析并复核唯一的测试容器 ID，再对该精确 ID 执行一次 `docker kill --signal KILL`，禁止使用名称通配符。期望容器按受控 `on-failure` 策略在原 generation 内恢复 healthy，恢复期间租约不释放、同一卡不启动第二个任务；若达到重启上限则应明确失败，不能无限占卡或伪报健康。
2. **Node Agent 暂时不可用**：在测试推理仍运行时短暂停止 `node-agent` 固定容器。期望控制面保留/隔离现有租约，不在结果不确定时向同一卡启动第二个任务；恢复 Agent 后完成对账。禁止手工删除数据库租约。
3. **DCGM Exporter 暂时不可用**：短暂停止 exporter。期望 GPU API/UI 显示 degraded/unknown，而不是伪造 0；重启并经过数个抓取周期后历史恢复。
4. **训练人工终止**：使用 BM-09 的专用长任务。期望容器进程组被回收、状态 canceled、GPU 释放且任务不自动重跑。
5. **鉴权与路由错误**：无 key、错误 key 返回 401；错误服务类型不被路由；`/v1/responses` 返回 404。

故障后必须回到以下稳定态才可继续：固定服务健康、Agent 工作负载与数据库状态一致、无重复租约、四卡均能继续调度、错误日志已归档。若恢复需要改数据库、删除未知容器或重启整机，应记录 `FAIL` 并先做根因分析。

## 15. BM-13：清理与最终签字

按下列顺序清理：

1. 取消或等待评测结束；
2. 终止或等待训练结束，确认已保存所需产物；
3. 在界面停止所有测试推理部署，等待 stopped；
4. 确认 `/api/v1/system/gpu-leases` 为空、Node Agent 无活动 workload、`nvidia-smi` 无测试进程；
5. 撤销并删除验收专用推理 API Key，清除 shell 中的 key 变量；
6. 保存日志、报告和截图的 SHA-256；
7. 如需停栈，先停止 `web/api/监控/node-agent`，最后停止 PostgreSQL。

禁止执行 `docker compose down -v`、`docker system prune`、通配符递归删除或直接删除数据库租约。训练任务记录删除不会自动递归删除 checkpoint；模型、数据集和产物是否删除应按明确 ID、备份状态和保留策略另行审批。

最终证据包至少包含：

- Git commit、主机与四卡 inventory、`nvidia-smi topo -m`、Docker/Compose/runtime 版本；
- 经过批准的镜像 digest 清单和只读 preflight 输出；
- Compose 状态、健康响应和必要的脱敏日志；
- 空闲、单卡、TP=2、TP=4、训练、评测阶段的 GPU 指标、历史曲线、租约与 Agent 工作负载；
- 三个 OpenAI 端点响应、流式 `[DONE]`、`/v1/responses` 的 404 证据；
- FIFO/非抢占/全有或全无时间线；
- 四类训练任务的配置、状态、指标、产物 manifest 与训练后部署证据；
- C-Eval、CMMLU、自定义领域集的 manifest、总体及 category 明细、warnings、灾难性遗忘百分比；
- 故障注入前后状态、清理后的空租约/空进程证据；
- 操作人、复核人、UTC 起止时间、每项 PASS/FAIL/BLOCKED、缺陷编号和最终上线决定。

建议使用下表签字：

| 验收项 | 结果 | 证据路径/编号 | 缺陷或备注 | 操作人 | 复核人 |
| --- | --- | --- | --- | --- | --- |
| BM-01 宿主与拓扑 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-02 镜像与 digest | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-03 Compose 启动 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-04 四卡监控 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-05 模型与数据 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-06 分别部署与端点 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-07 vLLM TP=1/2/4 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-08 FIFO 与非抢占 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-09 训练矩阵 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-10 训练后部署 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-11 前后量化评测 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-12 故障恢复 | 未执行 |  | 本轮仅编写手册 |  |  |
| BM-13 清理与证据 | 未执行 |  | 本轮仅编写手册 |  |  |

只有全部必测项完成、所有阻断缺陷关闭并由操作人和复核人共同签字后，才能把“代码与静态测试已通过”更新为“最终 4 × RTX 4090D 裸金属验收已通过”。
