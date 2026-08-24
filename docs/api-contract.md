# HTTP API 约定

## 通用规则

- 管理 API 前缀为 `/api/v1`，JSON 字段采用 `snake_case`。
- 列表统一返回 `{items, total, page, page_size}`，默认按创建时间倒序。
- 异步命令成功接受时返回 `202` 和操作/任务 ID；不把“已入队”误报为“已完成”。
- 错误统一返回 `{code, message, request_id, details}`；校验错误中的路径不得包含服务器真实绝对路径。
- 写操作接收 `Idempotency-Key`；状态更新使用 `version` 或 `If-Match` 防止覆盖并发操作。

## 管理接口

| 模块 | 主要端点 |
| --- | --- |
| 会话 | `POST /auth/login`、`POST /auth/logout`、`GET /auth/me` |
| 模型 | `GET/POST /models`、`GET/DELETE /models/{id}`、`POST /model-imports`、`POST /model-imports/{id}/cancel` |
| 受控目录 | `GET /model-inbox`、`POST /model-inbox/scan` |
| 部署 | `GET/POST /deployments`、`GET/PATCH/DELETE /deployments/{id}`、`POST /deployments/{id}/start|stop` |
| 数据集 | `GET/POST /datasets`、`GET/DELETE /datasets/{id}`、`GET /datasets/{id}/preview` |
| 训练 | `GET/POST /training-jobs`、`GET /training-jobs/{id}`、`POST /training-jobs/{id}/cancel|resume` |
| 训练指标 | `GET /training-jobs/{id}/metrics`、`GET /training-jobs/{id}/events` |
| 产物 | `GET /artifacts`、`POST /artifacts/{id}/register-model`、`GET /artifacts/{id}/download` |
| 测评 | `GET/POST /evaluations`、`GET /evaluations/{id}`、`POST /evaluations/{id}/cancel`、`GET /evaluation-comparisons/{id}` |
| 资源 | `GET /resources/gpus`、`GET /resources/summary`、`GET /resources/metrics` |
| 密钥 | `GET/POST /api-keys`、`DELETE /api-keys/{id}` |
| 审计 | `GET /audit-logs` |

## OpenAI 兼容入口

| 端点 | 允许的部署类型 | 流式 |
| --- | --- | --- |
| `POST /v1/completions` | generation | 是 |
| `POST /v1/chat/completions` | generation | 是 |
| `POST /v1/embeddings` | embedding | 否 |

请求中的 `model` 是部署对外公开的唯一别名。Gateway 在数据库/缓存中解析到健康实例；不存在、已停止或类型不匹配时返回明确的 OpenAI 风格错误。首版对 `/v1/responses` 返回 `404`，不做静默兼容。

## SSE 约定

管理事件流与 Playground 使用 `text/event-stream`。每条事件带单调递增的 `id`、事件类型和 JSON `data`。客户端重连可发送 `Last-Event-ID`；服务端只补发保留窗口内事件，窗口外返回快照后继续实时流。
