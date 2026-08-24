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

C-Eval 与 CMMLU 数据不会在镜像构建时隐式下载。下面的准备命令会生成带稳定样本 ID 的统一 JSONL，并同时生成可供控制面登记、审计的 manifest。

## 内置评测集的官方来源

本能力只使用以下项目的一手来源，不运行其仓库代码，也不提交官方大数据或答案：

| 数据集 | 官方资料与真实布局 | 数据许可 | 固定在线 revision | 固定制品 SHA-256 |
| --- | --- | --- | --- | --- |
| C-Eval | [官方仓库](https://github.com/hkust-nlp/ceval)，52 科；CSV 为 `dev/{subject}_dev.csv`、`val/{subject}_val.csv`、`test/{subject}_test.csv`，字段为 `id,question,A,B,C,D,answer[,explanation]` | [CC BY-NC-SA 4.0 数据许可](https://github.com/hkust-nlp/ceval/blob/cba65ae93bcf189149ced9f66ae0c958201faed9/LICENSE-DATA)；仓库代码的 MIT 许可不等于数据许可 | 官方 Hugging Face 数据仓库 `3923b519fd180e689d0961bf3a032ece929742f3` | `68786deeea68ff089c56563ee48fab8160da857b77b913437bb504d681fd8e20` |
| CMMLU | [官方仓库](https://github.com/haonan-li/CMMLU)，67 科；CSV 为 `data/dev/{subject}.csv`、`data/test/{subject}.csv`，字段为无名索引列及 `Question,A,B,C,D,Answer` | [官方 README 声明的 CC BY-NC-SA 4.0](https://github.com/haonan-li/CMMLU/blob/d6e7b716d8ac694f38969a6c0407437d1fded799/README.md#license) | 官方 GitHub commit `d6e7b716d8ac694f38969a6c0407437d1fded799` | `154593336d5074d793ed990222876b83490b0aed97638a62618d1fe2da7c2cac` |

上述许可均包含非商业与相同方式共享条件。使用者仍需自行确认具体使用方式符合许可；本文不是法律意见。

### 在线准备

在线模式只访问代码内固定的 HTTPS URL，并在转换前校验 revision 对应制品的 SHA-256。命令必须显式传入许可确认开关，否则不会发起网络请求。

```bash
openllmops-eval prepare-benchmark \
  --benchmark ceval \
  --online \
  --accept-non-commercial-license \
  --output-dir /workspace/builtin/ceval

openllmops-eval prepare-benchmark \
  --benchmark cmmlu \
  --online \
  --accept-non-commercial-license \
  --output-dir /workspace/builtin/cmmlu
```

C-Eval 固定的 CSV 历史快照中，`test` 没有标准答案，因此在线默认只生成 `dev` 与 `val`；这两部分可以计算准确率。CMMLU 默认生成有答案的 `dev` 与 `test`。若管理员取得了官方后续版本且该版本提供带答案的 C-Eval test，可通过下面的离线流程导入并记录其真实 revision，工具不会假定或补写任何答案。

### 离线准备

离线模式接受官方 CSV 根目录，或 ZIP、TAR、TAR.GZ、TGZ 归档。`--source-revision` 是管理员根据官方来源填写的 commit、tag 或发布版本；CLI 能验证布局与内容指纹，但无法在隔离环境中证明管理员提供文件的出处，因此 manifest 会将 `revision_verified` 记为 `false`。

```bash
openllmops-eval prepare-benchmark \
  --benchmark ceval \
  --source /srv/import/ceval-exam.zip \
  --source-revision 3923b519fd180e689d0961bf3a032ece929742f3 \
  --accept-non-commercial-license \
  --output-dir /workspace/builtin/ceval

openllmops-eval prepare-benchmark \
  --benchmark cmmlu \
  --source /srv/import/CMMLU \
  --source-revision d6e7b716d8ac694f38969a6c0407437d1fded799 \
  --accept-non-commercial-license \
  --output-dir /workspace/builtin/cmmlu
```

正式准备时，每个所选 split 必须分别包含 C-Eval 的 52 科或 CMMLU 的 67 科，并且不同 split 的科目集合相同。`--allow-partial` 仅供开发期合成小样本或明确的子集使用，使用情况与 `partial` 结果都会写入 manifest。输出已存在时默认失败，只有显式 `--overwrite` 才会替换。

### 输出与可复现性

命令输出两个文件：

- `<benchmark>.jsonl`：统一的 `openllmops-eval-jsonl-v1` 格式，字段包括 `id`、`task_type`、`category`、`question`、`choices`、`answer` 与 `metadata`。转换顺序和 JSON 序列化固定，同一来源会产生相同字节。
- `<benchmark>.manifest.json`：记录官方仓库、来源模式、revision、许可、来源 SHA-256、规范化 CSV 内容 SHA-256、输出 SHA-256、CSV 文件数、样本总数、split/科目计数和是否为部分数据。

目录来源的 `source.sha256` 是对“相对路径 + 原始 CSV 字节”按路径排序后的规范树指纹；归档来源的该字段是归档原始字节指纹。两种模式都会去除仓库包装目录、统一成 split/科目路径后额外生成 `content_sha256`，因此可比较目录与归档中的同一官方 CSV 内容。

### 输入安全边界

- 不导入、不执行上游仓库代码，只解析 UTF-8 CSV。
- 不接受 pickle；不会调用 ZIP/TAR 的 `extractall`，归档内容不会解压到文件系统。
- 拒绝绝对路径、`..`、反斜杠路径、加密 ZIP、符号/硬链接、设备文件与 FIFO。
- 限制下载大小、归档成员数、单个 CSV/CSV 总大小与 ZIP 压缩比，并校验字段、答案、官方科目名和 split/科目完整性。

旧的 `convert-benchmark` 命令仅为兼容已有脚本保留，不生成来源 manifest；新部署应统一使用 `prepare-benchmark`。

## 开发验证

测试只构造最小合成 CSV/归档，不下载或提交官方数据：

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```
