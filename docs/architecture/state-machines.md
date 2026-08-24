# 任务与服务状态机

状态机用于约束所有异步操作。API 写入期望状态，Reconciler 主动查询 Worker/Agent 后更新实际状态；
首版没有 Agent 回调和追加式状态事件表。

## 模型导入

`pending → transferring → validating → ready`

- `pending/transferring/validating → cancelling → cancelled`
- 任意执行态可进入 `failed`，重试会创建新的尝试记录。

## 推理部署

部署的期望状态只有 `running` 与 `stopped`；实际状态为：

`created → queued → starting → running → stopping → stopped`

- `starting/running/stopping → failed`
- 编辑运行中部署：`running → stopping → stopped → starting → running`
- 删除：先把期望状态设为 `stopped`；确认对应 runtime 已不存在后，接口删除部署记录。

部署删除不是软删除，但 Node Agent 只有在签名请求中的 owner、资源 ID 与 generation 都精确匹配时才
清理终态容器；无法确认停止时保留部署记录并返回冲突。

## 训练任务

`created → queued → starting → running → succeeded`

- `created/queued/starting/running → canceling → canceled`
- `starting/running → failed`
- Agent 暂时失联时保留当前状态与 GPU 租约，待同 generation 的状态查询收敛；确认容器丢失后进入 `failed`，不得自动重跑。
- 首版没有 `interrupted` 或 `resume` 状态。成功任务可导出已移除 pickle 优化器状态的安全 checkpoint，但该产物不承诺恢复训练。

终止请求幂等：重复终止不会创建多个停止操作，也不会把已成功任务改为取消。

## 评测任务

`created → queued → starting → running → succeeded`

- 取消与失败规则和训练任务一致。
- 基线与候选必须使用同一数据集版本、同一评测模板和兼容的生成参数，才能生成“前后对比”。

## 并发与非法转换处理

控制面动作在数据库事务中锁定目标行，并使用 `state_version`/`runtime_generation` 区分配置修订和运行
实例。非法转换返回冲突且不修改当前状态。Node Agent 请求携带资源 ID、owner 与 generation；旧
generation 的检查结果不能驱动当前修订。HTTP 审计记录管理员操作，但首版详情页不承诺完整的状态
变化时间线。
