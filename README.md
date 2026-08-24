# OpenLLMOps

OpenLLMOps 是面向单机多卡 NVIDIA 服务器的 B/S 架构大模型管理平台，覆盖模型资产、vLLM 推理部署、LLaMA-Factory 训练、JSONL 数据集、训练前后量化测评、Playground 与 GPU 资源监控。

目标生产环境为 Ubuntu 22.04、4 × RTX 4090D、128 GB 内存和 2 TB 磁盘；双卡开发机使用同一套代码与容器合同。4090D 之间没有 NVLink，系统因此采用整卡独占、非抢占式严格 FIFO 调度，不把训练任务建立在隐式抢停推理服务之上。

## 架构

- `frontend/`：Vue 3、Element Plus、Pinia、Vue Router、ECharts 和流式 Playground。
- `backend/`：FastAPI、SQLAlchemy async、PostgreSQL、Alembic、管理员会话/API Key、审计与任务状态协调。
- `agent/`：唯一持有受限 Docker 控制能力的节点代理；用双向 canonical JSON + HMAC 接收幂等工作负载命令。
- `workers/`：Safetensors 模型导入、LLaMA-Factory 白名单配置、安全训练/合并运行时和 checkpoint 安全归档。
- `evaluation/`：Base/Instruct 模板、选择题/分类/短问答评分、C-Eval/CMMLU 准备和训练前后顺序评测。
- `deploy/`：PostgreSQL、API、Web/Nginx、Prometheus、DCGM Exporter、Node Agent 与 Docker socket proxy 的单机 Compose 栈。

控制面不直接挂载 Docker socket。推理、训练和评测容器只由节点代理按镜像、参数、路径和 GPU 白名单创建；数据库 GPU 租约与节点 NVML 检查共同防止整卡重复分配。

## 已实现的关键能力

- 从 Hugging Face、ModelScope 或管理员人工复制后的受控目录创建可取消导入任务；拒绝 pickle 权重、软链接、路径逃逸与 `trust_remote_code`，校验成功才发布模型资产。
- 生成与 Embedding 分别部署，提供 `/v1/completions`、`/v1/chat/completions`、`/v1/embeddings`；本版本不实现 `/v1/responses`。
- CPT LoRA 与 SFT Freeze/LoRA/QLoRA 的受限配置生成，训练过程状态、指标和产物路径回传。
- CPT、SFT、Evaluation JSONL 上传、校验和预览。
- C-Eval、CMMLU 固定官方 revision 准备器，以及自定义选择题、分类和短问答的百分比/百分点/相对变化比较。
- 单管理员 HTTPS 管理面、HttpOnly 会话 Cookie、CSRF/Origin 校验、推理 API Key 与无请求体的安全审计日志。
- 推理期望状态自动恢复；训练任务不会自动续跑，且绝不会为训练自动停止推理服务。

## 本地开发

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Node.js 22 和 npm。macOS/无 GPU 环境可以运行控制面与全部非 GPU 测试，但不能验证 NVIDIA 容器。

```bash
make install
make check
```

前端默认连接真实接口；只有显式设置 `VITE_USE_MOCKS=true` 才加载演示数据。后端本地开发变量示例位于根目录 `.env.example`；复制到 `backend/.env` 前先创建 `/tmp/openllmops-dev` 下列出的受控目录。生产部署不要使用该文件。

## 单机生产部署

生产配置的唯一模板是 `deploy/.env.example`：

```bash
cp deploy/.env.example deploy/.env
uv run --with argon2-cffi python scripts/generate_secrets.py
sudo install -d -o 1000 -g 1000 \
  /srv/openllmops/{models,inbox,model-staging,datasets,evaluation-datasets,evaluation-output,checkpoints,training-configs,runtime}
```

将密钥生成器的每个值分别保存到 `deploy/.env`。Hugging Face/ModelScope token 不进入 `.env`，而应由 UID 1000 可读、不可写的独立文件映射到 `/run/secrets/model-sources`。管理员使用 SFTP 时，只负责把模型目录复制到 `/srv/openllmops/inbox`；平台不会保存 SFTP 账号或主动连接远程 SFTP 服务。

首次启动前构建安全训练与评测运行时，并执行只读预检：

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  --profile runtime-image build
sh deploy/scripts/preflight.sh deploy/.env
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
```

正式环境应使用组织 CA 证书，并把动态工作负载镜像白名单改为内部仓库的不可变 `@sha256` 引用。详细安装、备份、恢复和排障步骤见 [`deploy/README.md`](deploy/README.md) 与 [`docs/operations`](docs/operations/deployment.md)。

## 设计与接口资料

- 页面设计图：[`docs/design/mockups`](docs/design/mockups/README.md)
- 产品需求：[`docs/product-requirements.md`](docs/product-requirements.md)
- API 合同：[`docs/api-contract.md`](docs/api-contract.md)
- 架构与状态机：[`docs/architecture`](docs/architecture/overview.md)

## 当前验证边界

所有 Python/前端单元测试、类型检查、静态检查和数据库迁移均可在无 GPU 开发机执行。Compose 与 NVIDIA 运行时仍必须在真实 Ubuntu/NVIDIA 主机完成最终验收；在该验收完成前，不应把“代码测试通过”解释为 4090D 上的模型加载、训练吞吐或多卡 NCCL 已通过。
