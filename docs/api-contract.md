# HTTP API 约定

## 通用规则

- 管理 API 前缀为 `/api/v1`，JSON 字段采用 `snake_case`。
- 列表端点返回 JSON 数组，支持的端点使用 `offset`/`limit` 分页，默认按创建时间倒序。
- 创建资源返回 `201` 和完整资源；启动、停止、取消等动作返回当前期望/实际状态。进入队列不等于执行完成。
- 错误遵循 FastAPI 的 `detail` 结构；每个响应携带 `X-Request-ID`，服务端审计记录同一 ID，但不回显密钥或请求体。
- 管理写操作使用管理员会话时必须携带运行时 CSRF token；OpenAI 兼容入口只接受 API Key，不读取管理员 Cookie。

## 管理接口

| 模块 | 主要端点 |
| --- | --- |
| 会话 | `POST /auth/login`、`POST /auth/logout`、`GET /auth/me` |
| 模型资产 | `GET /model-assets`、`GET/PATCH/DELETE /model-assets/{id}`（创建只能走受控导入或训练发布） |
| 模型导入 | `GET/POST /model-imports`、`GET /model-imports/{id}`、`POST /model-imports/{id}/cancel` |
| 受控目录 | `GET /model-inbox`（每次请求执行只读扫描） |
| 部署 | `GET/POST /deployments`、`GET/PATCH/DELETE /deployments/{id}`、`POST /deployments/{id}/start|stop` |
| 数据集 | `GET /datasets`、`POST /datasets/upload`、`GET/PATCH/DELETE /datasets/{id}`、`GET /datasets/{id}/preview` |
| 训练 | `GET/POST /training-jobs`、`GET/DELETE /training-jobs/{id}`、`POST /training-jobs/{id}/terminate` |
| 训练产物 | `GET /training-jobs/{id}/artifacts`、`GET /training-jobs/{id}/artifacts/{kind}/download`、`POST /training-jobs/{id}/publish-model` |
| 测评 | `GET/POST /evaluation-runs`、`GET/DELETE /evaluation-runs/{id}`、`POST /evaluation-runs/{id}/cancel` |
| 资源 | `GET /system/capabilities`、`GET /system/gpu-leases`、`GET /system/gpus`、`GET /system/gpus/{index}/history` |
| 密钥 | `GET/POST /api-keys`、`POST /api-keys/{id}/revoke`、`DELETE /api-keys/{id}` |
| 审计 | `GET /audit-logs` |
| 总览 | `GET /dashboard/summary` |

首版不提供训练 `resume` 端点。成功 checkpoint 是经过安全清理的 Safetensors 快照，不包含可反序列化的优化器/RNG 状态；继续训练需创建新任务。

## OpenAI 兼容入口

| 端点 | 允许的部署类型 | 流式 |
| --- | --- | --- |
| `POST /v1/completions` | generation | 是 |
| `POST /v1/chat/completions` | generation | 是 |
| `POST /v1/embeddings` | embedding | 否 |

请求中的 `model` 是部署对外公开的唯一别名。Gateway 在数据库/缓存中解析到健康实例；不存在、已停止或类型不匹配时返回明确的 OpenAI 风格错误。首版对 `/v1/responses` 返回 `404`，不做静默兼容。

## 流式响应边界

Playground 直接消费 `/v1/chat/completions` 的 OpenAI Compatible SSE 数据帧，并以
`data: [DONE]` 结束。首版管理面通过轮询读取任务当前状态，不提供管理事件 SSE、事件补发或
`Last-Event-ID` 合同；网络中断后的 Playground 请求需要由用户重新发起。
