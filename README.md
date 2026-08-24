# OpenLLMOps

OpenLLMOps 是面向单机多卡 NVIDIA 服务器的大模型服务综合管理系统。系统采用 B/S 架构，覆盖模型资产导入、vLLM 推理部署、LLaMA-Factory 训练、训练数据集、模型测评、Playground 和 GPU 资源监控。

## 已确认的目标环境

- 生产：Ubuntu 22.04、4 × RTX 4090D、128 GB 内存、2 TB 磁盘、PCIe 互联且无 NVLink。
- 开发：允许缩减为 2 × RTX 4090D，并与生产使用相同的固定版本容器镜像。
- 前端：Vue 3 + Element Plus。
- 后端：FastAPI + PostgreSQL。
- 部署：Docker Compose 管理控制面，受限 Node Agent 动态管理推理、训练和测评容器。

## 当前接口范围

- `/v1/completions`
- `/v1/chat/completions`
- `/v1/embeddings`

当前版本不实现 `/v1/responses`。

## 设计资料

高保真页面设计图和统一设计约束位于 [`docs/design/mockups`](docs/design/mockups/README.md)。

## 开发状态

项目处于初始化与核心功能开发阶段。后续提交将逐步加入前后端工程、数据库迁移、任务状态机、GPU 调度、容器编排与测试。

