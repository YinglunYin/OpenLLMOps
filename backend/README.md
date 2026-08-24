# OpenLLMOps Backend

后端控制面基于 FastAPI、SQLAlchemy 2 异步接口和 PostgreSQL。开发/测试可切换到
SQLite，生产环境必须使用 PostgreSQL。

```bash
uv sync --group dev
AUTO_CREATE_TABLES=true AUTH_ENABLED=false uv run uvicorn app.main:app --reload
uv run pytest
```

接口分为三组：

- `/health/*`：不鉴权的存活与就绪探针；
- `/api/v1/*`：模型资产、数据集、部署、训练、评测和 API Key 控制面；
- `/v1/completions`、`/v1/chat/completions`、`/v1/embeddings`：OpenAI 兼容网关。

本版本明确不注册 `/v1/responses`。

## 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `DATABASE_URL` | 生产使用 `postgresql+asyncpg://...` |
| `AUTO_CREATE_TABLES` | 仅开发/测试为 `true`；生产使用 Alembic |
| `AUTH_ENABLED` / `ADMIN_API_KEY` | 总开关与单管理员引导密钥 |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD_HASH` | 单管理员用户名与 Argon2 密码哈希；不接收明文密码配置 |
| `SESSION_SIGNING_KEY` / `SESSION_TTL_SECONDS` | 会话 HMAC 密钥（至少 32 字符）和有效期 |
| `SESSION_COOKIE_SECURE` | 默认 `true`，生产环境禁止关闭 |
| `TRUSTED_PROXY_CIDRS` | 可转发真实客户端 IP 的代理 CIDR；默认不信任任何代理 |
| `API_KEY_PEPPER` | 数据库 API Key 摘要的额外服务端秘密 |
| `MODEL_ROOT` / `MODEL_INBOX_ROOT` | 模型资产目录与人工拷贝受控入口 |
| `MODEL_STAGING_ROOT` | 导入暂存目录；必须与 `MODEL_ROOT` 位于同一文件系统，才能原子入库 |
| `MODEL_IMPORT_COORDINATOR_ENABLED` | 是否启动数据库轮询导入协调器；生产 API 实例应设为 `true` |
| `MODEL_IMPORT_POLL_INTERVAL_SECONDS` | 导入任务轮询间隔，默认 1 秒 |
| `MODEL_IMPORT_CONCURRENCY` | 单实例并发导入数，默认 1、最大 4 |
| `MODEL_IMPORT_CLAIM_TIMEOUT_SECONDS` | 导入 claim 心跳失效阈值，默认 120 秒；用于硬崩恢复 |
| `HUGGINGFACE_TOKEN_FILE` / `MODELSCOPE_TOKEN_FILE` | 可选的仓库访问令牌只读 secret 文件路径，不接受令牌环境变量或 API 字段 |
| `DATASET_ROOT` / `CHECKPOINT_ROOT` | 数据集和训练产物受控目录 |
| `GPU_COUNT` | 开发机通常为 2，目标生产机为 4 |
| `NODE_AGENT_URL` / `NODE_AGENT_TOKEN` | 受限宿主机执行代理地址与双向 HMAC 共享密钥 |
| `RECONCILER_ENABLED` / `RECONCILER_INTERVAL_SECONDS` | 状态协调器开关与轮询间隔 |
| `GPU_LEASE_TTL_SECONDS` | GPU 租约心跳失效时间；生产默认 30 秒 |
| `NODE_AGENT_CLOCK_SKEW_SECONDS` | HMAC 请求/响应允许的最大时钟偏差 |
| `VLLM_INTERNAL_API_KEY` | OpenAI 网关访问内部 vLLM 实例的密钥 |
| `PROMETHEUS_URL` / `PROMETHEUS_TIMEOUT_SECONDS` | Prometheus HTTP API 地址与查询超时；未配置时 GPU API 明确返回 degraded |

## 状态与调度边界

控制面将部署和任务的 `desired_state` 与 `actual_state` 分开保存。`start` 接口只把部署放入
统一 FIFO 非抢占队列，reconciler 拿到全部指定整卡租约后才调用 node-agent。生产 PostgreSQL
通过事务级 advisory lock、`SELECT ... FOR UPDATE` 和 `gpu_leases.gpu_index` 唯一约束保证
整组 GPU 全有或全无；SQLite 的进程内锁只用于单元测试。

推理、训练和评测共用队列与租约，但使用不同的实际状态收敛规则：推理实例意外消失时可
重新排队，训练和评测则失败并等待人工处理。训练资源不足只会继续排队，绝不会主动停止
已运行的推理服务。租约过期且 node-agent 失联时 GPU 会保持隔离，只有确认工作负载已结束
后才回收，避免同一卡上启动第二个容器。

node-agent 合同为 `POST /v1/workloads/commands`。请求与响应都对规范化 JSON 原始字节执行
HMAC-SHA256 签名，携带时间戳与一次性 nonce；动作由 `start`、`stop`、`status` 组成，
`request_id` 和 `owner.generation` 提供重试幂等与迟到响应隔离。

## 管理员密码与会话

使用交互式输入生成 Argon2 哈希，避免明文密码进入命令历史：

```bash
uv run python -c 'from argon2 import PasswordHasher; from getpass import getpass; print(PasswordHasher().hash(getpass("Admin password: ")))'
uv run python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

第一行结果配置为 `ADMIN_PASSWORD_HASH`，第二行结果配置为 `SESSION_SIGNING_KEY`。若通过
Docker Compose `.env` 传递 Argon2 哈希，应使用单引号包裹，避免其中的 `$` 被变量插值。

浏览器登录返回 HttpOnly、Secure、SameSite=Strict 的签名 Cookie；响应体中的 CSRF token
只需保存在前端内存，并在管理写请求的 `X-CSRF-Token` 头中回传。OpenAI 兼容接口不会读取
该 Cookie，只接受 API Key。生产环境还必须把 `CORS_ORIGINS` 配成明确的 HTTPS 来源。

审计来源 IP 默认使用 TCP 直连地址并忽略转发头。只有直连节点属于
`TRUSTED_PROXY_CIDRS` 时才从右向左解析 `X-Forwarded-For`；同时应确保 Uvicorn 的
`forwarded-allow-ips` 使用相同或更严格的可信代理范围。

## 模型导入

`POST /api/v1/model-imports` 创建 Hugging Face、ModelScope 或受控目录导入任务；
`GET /api/v1/model-imports` 与详情接口读取持久化阶段和进度，`cancel` 接口可幂等取消。
人工/SFTP 拷贝只允许放入 `MODEL_INBOX_ROOT` 的一层普通子目录，前端通过
`GET /api/v1/model-inbox` 展示可导入候选。软链接、目录逃逸、pickle 类权重和要求执行远程
代码的模型都会被 worker 拒绝。

协调器以数据库原子领取任务：PostgreSQL 使用 `FOR UPDATE SKIP LOCKED`，SQLite 开发测试
使用单条条件更新。worker 完成 Safetensors 与配置校验并原子移入 `MODEL_ROOT` 后，控制面
才在同一事务中创建 `ready` 模型资产；失败或取消任务不会产生可部署资产。在线访问令牌
只从只读文件临时读入执行线程，既不写入导入任务/资产记录，也不接受请求体传入。

协调器持续用 `claimed_at` 作为执行心跳。进程或主机硬崩后，超过
`MODEL_IMPORT_CLAIM_TIMEOUT_SECONDS` 的 transferring/validating 任务会清理其固定 UUID
暂存/未发布目录并重新排队，canceling 任务会收敛为 canceled。PostgreSQL advisory lock 与
任务行锁防止多实例并发恢复；已有模型资产关联的路径永远不会被恢复清理。worker 已完成
原子移动后以 ready 为发布提交点，最后检查点之后到达的取消不会制造无记录孤儿目录。

## GPU 监控与仪表盘

`GET /api/v1/system/gpus` 通过 Prometheus 查询 DCGM Exporter 的显存、利用率、温度和功耗，
并按 GPU index 合并数据库整卡租约。缺失指标保持 `null` 并在 `degraded_reason` 说明原因，
Prometheus 超时、不可用或响应非法时不会用 0 冒充真实遥测。

`GET /api/v1/system/gpus/{gpu_index}/history` 仅接受固定指标枚举、最多 7 天跨度、5–3600 秒
步长及最多 2000 点。PromQL 完全由服务端白名单构造，查询参数不能注入表达式。
`GET /api/v1/dashboard/summary` 用三个数据库往返返回模型/部署/训练/评测计数、队列、GPU
租约和最近活动，不依赖逐资源列表扫描。
