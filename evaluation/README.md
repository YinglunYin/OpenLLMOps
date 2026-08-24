# OpenLLMOps Evaluation Worker

评测执行器支持三种 JSONL 任务：选择题、分类与短问答。它既能调用已有 OpenAI Compatible 部署，也能在一个独占 GPU 容器内顺序启动基线和候选 vLLM，实现相同数据、模板和生成参数下的前后比较。

```bash
openllmops-eval run \
  --dataset /workspace/dataset/eval.jsonl \
  --output /workspace/output/report.json \
  --base-url http://gateway \
  --model qwen-base \
  --template base

openllmops-eval run-pair \
  --dataset /workspace/dataset/eval.jsonl \
  --output-dir /workspace/output \
  --baseline-path /workspace/baseline \
  --baseline-template base \
  --candidate-path /workspace/candidate \
  --candidate-template instruct \
  --tensor-parallel-size 2
```

`run-pair` 先完整评测基线并终止其 vLLM 进程，确认释放显存后再启动候选模型。因此两者不会同时占用双倍 GPU。输出包括两份原始报告和一份比较报告，正确率使用百分比、差异使用百分点并同时给出相对变化。

C-Eval 与 CMMLU 数据不在镜像构建时隐式下载。管理员按其许可取得 CSV 后，用 `convert-benchmark` 转为带稳定样本 ID 的统一 JSONL，并把数据集版本与 SHA-256 登记到控制面。
