# OpenLLMOps Node Agent

node-agent 是单机 GPU 节点的最小特权执行面。控制面负责期望状态、整卡租约、非抢占式排队和审计；node-agent 再次校验 GPU 占用，并创建受限的 vLLM/LLaMAFactory 容器。

## 内部 API 与鉴权

工作负载写操作只有一个入口：`POST /v1/workloads/commands`。请求和响应都使用 `NODE_AGENT_TOKEN` 作为共享密钥进行 HMAC-SHA256 签名；旧的 token 直传创建、启动、停止和删除端点已移除，不能绕过 generation 与幂等检查。

| 方法 | 路径 | 鉴权 | 作用 |
| --- | --- | --- | --- |
| `POST` | `/v1/workloads/commands` | 双向 HMAC | 幂等执行 `start`、`stop`、`status` |
| `GET` | `/v1/gpus` | `X-Node-Agent-Token` | NVML 指标与当前整卡分配 |
| `GET` | `/v1/workloads` | `X-Node-Agent-Token` | 列出受管容器 |
| `GET` | `/v1/workloads/{name}` | `X-Node-Agent-Token` | 查看受管容器 |
| `GET` | `/v1/workloads/{name}/logs` | `X-Node-Agent-Token` | 获取末尾日志 |
| `GET` | `/healthz`、`/metrics` | 内部网络隔离 | 健康检查和 Prometheus 指标 |

请求必须是规范 JSON 的原始 UTF-8 字节：`ensure_ascii=false`、`allow_nan=false`、键排序、无多余空格。三个签名头为：

- `X-OpenLLMOps-Timestamp`：Unix 秒；默认允许前后 30 秒时钟偏差。
- `X-OpenLLMOps-Nonce`：每次请求唯一；agent 至少保留 60 秒并拒绝重放。
- `X-OpenLLMOps-Signature`：`v1=<小写十六进制 HMAC-SHA256>`。

签名输入逐字节为 `v1\n<timestamp>\n<nonce>\n<raw_body>`。响应也按同一规则对规范 JSON 原始字节签名，控制面必须先验签再解释状态码或释放 GPU 租约。

`request_id` 用于传输重试去重；`owner.type + owner.id + owner.generation` 用于拒绝迟到命令。水位和有限的请求结果缓存在 `RUNTIME_ROOT/node-agent/command-state.json`，agent 重启后仍然有效。同一 `request_id` 或同一 generation 被绑定到不同参数时返回签名的 `409`，旧 generation 同样返回 `409`。

`start` 的 `execution.runner` 与 owner 类型固定映射：

- `deployment -> vllm`：支持 `service_type=generate|embedding`，返回 `endpoint`、`port` 和 `service_type`。
- `training -> llamafactory`：agent 根据受控 JSONL 数据集生成配置，状态响应尽可能返回 `progress`、步数、metrics、checkpoint、adapter 和合并模型路径。
- `evaluation -> evaluation`：当前合同没有安全的输出目录字段，agent 会返回签名的 `422 unsupported_runner`，不会伪造启动成功。

`stop` 和 `status` 的 `execution` 必须是空对象。停止成功会删除已停止容器并返回 `absent`，控制面只有看到该状态后才能释放整卡租约。

## 运行安全边界

推理详细参数采用节点侧白名单。`--model`、`--host`、`--port`、`--served-model-name`、`--runner`、`--convert`、`--load-format safetensors` 和 `--tensor-parallel-size` 都由 agent 构造，调用方不能覆盖；`trust_remote_code` 永久不在白名单中。embedding 服务固定使用 vLLM pooling/embed 模式。

训练只允许继续预训练 `stage=pt + LoRA`，以及 SFT 的 Freeze、LoRA、4-bit QLoRA。训练容器只执行 `llamafactory-cli train`，不启动 WebUI；模型、数据集和配置只读挂载，checkpoint 与任务缓存按任务单独可写，并固定离线环境和 `trust_remote_code=false`。

动态容器固定为非 root、只读根文件系统、丢弃全部 capabilities、启用 `no-new-privileges`，且不能使用宿主机网络、特权模式或任意挂载。GPU 分配在单 worker 内串行完成，并结合受管容器标签与 NVML 外部进程检查实现整卡独占；任何不确定状态均拒绝启动，不自动抢占运行中的推理服务。

## LLaMAFactory 镜像基线

LLaMAFactory `0.9.5` 及以前受 `GHSA-mwc7-mf87-v3mf` 影响。agent 禁止直接使用上游 `hiyouga/llamafactory`、`latest` 和生产环境可变 tag；默认只接受项目固定安全衍生版 `openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1`。其他引用必须是 `registry/repository@sha256:...`，而且镜像必须预先存在并携带匹配的安全构建标签。

agent 在校验后使用不可变 Docker image ID 启动容器，避免 tag 在检查与启动之间被替换。构建、校验与生产 digest 推广步骤见 `deploy/README.md`。
