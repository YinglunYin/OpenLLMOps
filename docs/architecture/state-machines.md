# 任务与服务状态机

状态机用于约束所有异步操作。API 只提交命令，Worker/Agent 回报事件后才能进入相应完成状态。

## 模型导入

`pending → transferring → validating → ready`

- `pending/transferring/validating → cancelling → cancelled`
- 任意执行态可进入 `failed`，重试会创建新的尝试记录。

## 推理部署

部署的期望状态只有 `running` 与 `stopped`；实际状态为：

`draft → queued → starting → running → stopping → stopped`

- `starting/running/stopping → failed`
- 编辑运行中部署：`running → stopping → stopped → starting → running`
- 删除：先把期望状态设为 `stopped`，实际停止后进入 `deleting → deleted`。

删除是软删除。只要部署仍被评测或审计记录引用，就不能物理清除其元数据。

## 训练任务

`draft → queued → allocating → running → succeeded`

- `queued/allocating/running → cancelling → cancelled`
- `allocating/running → failed`
- 控制面或 Agent 异常且容器无法确认时进入 `interrupted`；只有管理员可从有效 checkpoint 创建恢复尝试。

终止请求幂等：重复终止不会创建多个停止操作，也不会把已成功任务改为取消。

## 评测任务

`draft → queued → allocating → running → aggregating → succeeded`

- 取消与失败规则和训练任务一致。
- 基线与候选必须使用同一数据集版本、同一评测模板和兼容的生成参数，才能生成“前后对比”。

## 非法转换处理

每次状态更新携带记录版本号，使用乐观锁防止重复回调覆盖新状态。非法转换返回冲突错误并写审计日志，但不得修改当前状态。Node Agent 回调必须包含操作 ID；过期操作的回调只记录，不驱动当前修订。
