# OpenLLMOps 页面设计图

本目录保存开发前的高保真页面设计参考图。图片由内置 `image_gen` 模式生成，统一按 16:9 桌面后台系统设计。

## 统一设计提示

- Vue 3 + Element Plus 风格的可落地企业后台，不使用概念艺术语言。
- 白色顶栏、深海军蓝左侧导航、浅冷灰内容背景、白色卡片。
- 主色 `#2563EB`，成功/警告/异常分别使用绿色、琥珀色和红色。
- 固定采用“顶栏 + 左侧导航 + 右侧内容面板”，紧凑但清晰的中文信息层级。
- 禁止渐变、玻璃拟态、3D 装饰、Kubernetes/多节点元素和无关品牌标识。
- 页面数据均为交互与布局示例；运行时版本、模型名称、地址、指标数值不构成交付 BOM 或验收数据。

## 最终设计图

| 编号 | 模块 | 文件 | 页面重点 |
| --- | --- | --- | --- |
| 01 | 总览 | `01-dashboard-overview.png` | 四卡状态、任务队列、活动与快捷操作 |
| 02 | 模型资产 | `02-model-assets.png` | 多来源导入、能力标签、校验进度与模型详情 |
| 03 | 模型部署 | `03-model-deployments.png` | 生成/Embedding 分离部署、GPU 预留、端点与 API Key |
| 04 | 训练任务 | `04-training-jobs.png` | CPT/SFT、LoRA/QLoRA/Freeze、等待 GPU、训练曲线与 Checkpoint |
| 05 | 训练数据集 | `05-training-datasets.png` | JSONL 版本、校验结果、Token 分布与错误行 |
| 06 | 模型测评 | `06-model-evaluation.png` | C-Eval/CMMLU/领域集的训练前后百分比对比 |
| 07 | Playground | `07-playground.png` | `/v1/chat/completions` 流式对话与推理参数 |
| 08 | 资源监控 | `08-resource-monitoring.png` | 4×RTX 4090D、PCIe 无 NVLink、租约与系统指标 |
| 09 | 系统设置 | `09-system-settings.png` | HF/ModelScope/SFTP、受控目录、HTTPS/API Key 与安全策略 |

## 实现注意事项

- 设计图是布局、信息层级和交互方向参考，不应通过截图切图实现页面。
- GPU 必须以 UUID 管理；界面显示序号仅用于可读性。
- 训练、推理和测评共享整卡独占、非抢占式调度。
- 外部模型首版只接受 Safetensors，并关闭 `trust_remote_code`。
- 本版本接口范围为 `/v1/completions`、`/v1/chat/completions` 和 `/v1/embeddings`，不实现 `/v1/responses`。
- `drafts/` 仅保留生成过程中的候选图，不作为开发依据。
