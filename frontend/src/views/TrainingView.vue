<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { CircleCheck, CloseBold, Clock, Download, Plus, VideoPlay, Warning } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'

import BaseChart from '@/components/BaseChart.vue'
import GpuCard from '@/components/GpuCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { formatBytes } from '@/api/adapters'
import { api, normalizeTraining } from '@/api/services'
import { useMocks } from '@/api/client'
import type { BackendTrainingArtifact } from '@/api/contracts'
import type { Dataset, GpuDevice, ModelAsset, StatusTone, TrainingJob } from '@/types/domain'

type TrainingArtifactRow = BackendTrainingArtifact & { jobId: string }

const router = useRouter()
const rows = ref<TrainingJob[]>([])
const selected = ref<TrainingJob | null>(null)
const artifactRows = ref<TrainingArtifactRow[]>([])
const artifactError = ref('')
const artifactsBusy = ref(false)
const publishBusy = ref(false)
const gpuOptions = ref<GpuDevice[]>([])
const modelOptions = ref<ModelAsset[]>([])
const datasetOptions = ref<Dataset[]>([])
const createVisible = ref(false)
const createBusy = ref(false)
let refreshTimer: number | undefined
let refreshRunning = false
let artifactRequestVersion = 0
const form = reactive({
  name: '', stage: 'SFT', algorithm: 'LoRA', modelAssetId: '', datasetId: '', gpuIds: [0] as number[],
  epochs: 3, learningRate: 0.0002, cutoffLen: 2048, batchSize: 2, gradientAccumulation: 8,
  loggingSteps: 10, saveSteps: 100, warmupRatio: 0.03, loraRank: 8, loraAlpha: 16,
  loraDropout: 0.05, freezeTrainableLayers: 2, maxSamples: null as number | null, seed: 42,
  template: 'qwen',
})

watch(() => form.stage, (stage) => {
  if (stage === 'CPT') form.algorithm = 'LoRA'
  form.datasetId = ''
})

const counts = computed(() => ({
  running: rows.value.filter((item) => item.status === 'running').length,
  queued: rows.value.filter((item) => item.status === 'queued').length,
  completed: rows.value.filter((item) => item.status === 'completed').length,
  failed: rows.value.filter((item) => item.status === 'failed').length,
}))
const reportedPaths = computed(() => selected.value ? [
  { label: '输出目录', path: selected.value.outputDir },
  { label: 'Checkpoint', path: selected.value.checkpointPath },
  { label: 'LoRA Adapter', path: selected.value.adapterPath },
  { label: '合并模型', path: selected.value.mergedModelPath },
].filter((item): item is { label: string; path: string } => Boolean(item.path)) : [])
const terminalStatuses: TrainingJob['status'][] = ['completed', 'failed', 'terminated']
const artifactLabels: Record<BackendTrainingArtifact['kind'], string> = {
  checkpoint: 'Checkpoint', adapter: 'Adapter', merged: '合并模型', full: '完整输出',
}
const artifactLabel = (kind: BackendTrainingArtifact['kind']) => artifactLabels[kind]
const gpuStateText = (gpu: GpuDevice) => ({ idle: '空闲', inference: '推理占用', training: '训练占用', reserved: '已预留', unmanaged: '未纳管占用', unknown: '状态未知' })[gpu.state]
const selectedGpus = computed(() => selected.value
  ? gpuOptions.value.filter((gpu) => selected.value?.gpuIds.includes(gpu.index))
  : [])
const trainingConfigPreview = computed(() => normalizeTraining({ ...form }).training_config)
const reportedMetricEntries = computed(() => {
  const metrics = selected.value?.metrics ?? {}
  const labels: Record<string, string> = {
    loss: 'Train Loss', eval_loss: 'Eval Loss', learning_rate: 'Learning Rate',
    epoch: 'Epoch', grad_norm: 'Grad Norm', tokens_per_second: 'Tokens/s',
    train_samples_per_second: 'Samples/s', eta: 'ETA',
  }
  return Object.entries(metrics)
    .filter(([, value]) => ['string', 'number', 'boolean'].includes(typeof value))
    .slice(0, 12)
    .map(([key, value]) => ({ key, label: labels[key] ?? key, value: String(value) }))
})

const statusMap: Record<TrainingJob['status'], { text: string; tone: StatusTone }> = {
  running: { text: '训练中', tone: 'success' }, queued: { text: '等待 GPU', tone: 'warning' }, completed: { text: '已完成', tone: 'primary' }, failed: { text: '失败', tone: 'danger' }, stopping: { text: '终止中', tone: 'info' }, terminated: { text: '已终止', tone: 'info' },
}
const statusMeta = (status: TrainingJob['status']) => statusMap[status]

const lossOption = (offset = 0) => ({
  grid: { top: 12, left: 32, right: 8, bottom: 22 }, xAxis: { type: 'category', data: ['0', '2.5k', '5k', '7.5k', '10k'], axisLabel: { fontSize: 10, color: '#788498' } }, yAxis: { type: 'value', axisLabel: { fontSize: 10, color: '#788498' }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } }, series: [{ type: 'line', data: [2.5 + offset, 1.7 + offset, 1.35 + offset, 1.08 + offset, .88 + offset], smooth: true, showSymbol: false, lineStyle: { color: '#1769f5', width: 2 } }],
})
const learningRateOption = { grid: { top: 12, left: 44, right: 8, bottom: 22 }, xAxis: { type: 'category', data: ['0', '2.5k', '5k', '7.5k', '10k'], axisLabel: { fontSize: 10, color: '#788498' } }, yAxis: { type: 'value', axisLabel: { fontSize: 9, color: '#788498', formatter: (value: number) => `${value}e-4` }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } }, series: [{ type: 'line', data: [0, 1.4, 1.65, .8, .18], smooth: true, showSymbol: false, lineStyle: { color: '#1769f5', width: 2 }, areaStyle: { color: '#1769f514' } }] }

async function stopJob() {
  if (!selected.value) return
  await ElMessageBox.confirm('终止会中断当前训练并释放 GPU；首版不支持从 checkpoint 恢复，未完成任务的原始状态文件不会开放下载。', `终止 ${selected.value.name}`, { type: 'warning' })
  try {
    if (useMocks) selected.value.status = 'stopping'
    await api.training.stop(selected.value.id)
    if (!useMocks) {
      rows.value = await api.training.list()
      selected.value = rows.value.find((item) => item.id === selected.value?.id) ?? null
    }
    ElMessage.success('终止请求已提交')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '终止请求失败')
  }
}

async function refreshJobs() {
  if (refreshRunning) return
  refreshRunning = true
  try {
    const selectedId = selected.value?.id
    ;[rows.value, gpuOptions.value] = await Promise.all([api.training.list(), api.resources.gpus()])
    selected.value = rows.value.find((item) => item.id === selectedId) ?? rows.value[0] ?? null
  } finally {
    refreshRunning = false
  }
}

async function loadArtifacts(job: TrainingJob | null) {
  const requestVersion = ++artifactRequestVersion
  artifactRows.value = []
  artifactError.value = ''
  if (!job || !terminalStatuses.includes(job.status)) return
  artifactsBusy.value = true
  try {
    const artifacts = await api.training.artifacts(job.id)
    if (requestVersion !== artifactRequestVersion || selected.value?.id !== job.id) return
    artifactRows.value = artifacts.map((artifact) => ({ ...artifact, jobId: job.id }))
  } catch (error) {
    if (requestVersion !== artifactRequestVersion || selected.value?.id !== job.id) return
    artifactError.value = error instanceof Error ? error.message : '训练产物清单加载失败'
  } finally {
    if (requestVersion === artifactRequestVersion) artifactsBusy.value = false
  }
}

function downloadArtifact(artifact: TrainingArtifactRow) {
  if (useMocks) {
    ElMessage.success(`演示下载：${artifact.archive_filename}`)
    return
  }
  // 让浏览器直接流式保存后端 FileResponse，避免把数 GB 模型压缩包读入页面内存。
  const link = document.createElement('a')
  link.href = api.training.artifactDownloadUrl(artifact.jobId, artifact.kind)
  link.download = artifact.archive_filename
  link.rel = 'noopener'
  document.body.append(link)
  link.click()
  link.remove()
}

async function publishModel() {
  if (!selected.value || selected.value.status !== 'completed') return
  if (selected.value.publishedModelAssetId) {
    await router.push({ name: 'deployments', query: { model: selected.value.publishedModelAssetId } })
    return
  }
  try {
    await ElMessageBox.confirm('系统将复核 Safetensors、配置和 tokenizer，并把完整模型原子发布到模型资产目录。', `发布 ${selected.value.name}`, { type: 'info' })
    publishBusy.value = true
    const asset = await api.training.publishModel(selected.value.id)
    selected.value.publishedModelAssetId = asset.id
    ElMessage.success('训练模型已发布，正在打开部署表单')
    await router.push({ name: 'deployments', query: { model: asset.id } })
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '训练模型发布失败')
  } finally {
    publishBusy.value = false
  }
}

async function removeJob(row: TrainingJob) {
  try {
    await ElMessageBox.confirm('只能删除已完成、失败或已终止的训练任务记录，checkpoint 文件不会在此操作中被递归清理。', `删除 ${row.name}`, { type: 'warning' })
    await api.training.remove(row.id)
    await refreshJobs()
    ElMessage.success('训练任务已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '删除训练任务失败')
  }
}

async function createJob() {
  if (!form.name || !form.modelAssetId || !form.datasetId || !form.gpuIds.length) { ElMessage.warning('请填写任务名称并选择模型、数据集及至少一张 GPU'); return }
  createBusy.value = true
  try {
    await api.training.create({ ...form })
    createVisible.value = false
    if (!useMocks) rows.value = await api.training.list()
    selected.value = rows.value[0] ?? null
    ElMessage.success('训练任务已进入非抢占式 GPU 队列')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '训练任务创建失败')
  } finally {
    createBusy.value = false
  }
}

onMounted(async () => {
  try {
    ;[rows.value, modelOptions.value, datasetOptions.value, gpuOptions.value] = await Promise.all([api.training.list(), api.models.list(), api.datasets.list(), api.resources.gpus()])
    selected.value = rows.value[0] ?? null
    refreshTimer = window.setInterval(() => { void refreshJobs().catch(() => undefined) }, 3_000)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '训练任务加载失败')
  }
})

onBeforeUnmount(() => {
  if (refreshTimer !== undefined) window.clearInterval(refreshTimer)
})

watch(
  () => `${selected.value?.id ?? ''}:${selected.value?.status ?? ''}`,
  () => { void loadArtifacts(selected.value) },
)
</script>

<template>
  <div>
    <PageHeader title="训练任务" subtitle="基于 LLaMA-Factory 执行继续预训练与 Freeze / LoRA / QLoRA 微调">
      <el-button type="primary" :icon="Plus" @click="createVisible=true">创建训练任务</el-button>
    </PageHeader>

    <div class="stats-grid">
      <StatCard label="运行中" :value="counts.running" :icon="VideoPlay" tone="green" />
      <StatCard label="等待 GPU" :value="counts.queued" :icon="Clock" tone="orange" />
      <StatCard label="已完成" :value="counts.completed" :icon="CircleCheck" tone="blue" />
      <StatCard label="失败" :value="counts.failed" :icon="CloseBold" tone="red" />
    </div>

    <PanelCard class="section-gap" flush>
      <el-table :data="rows" highlight-current-row row-key="id" @current-change="(row: TrainingJob) => selected=row">
        <el-table-column prop="name" label="任务名称" min-width="220"><template #default="{row}"><strong>{{ row.name }}</strong></template></el-table-column>
        <el-table-column prop="stage" label="训练阶段" width="105" />
        <el-table-column prop="algorithm" label="算法" width="112" />
        <el-table-column prop="baseModel" label="基础模型" min-width="170" />
        <el-table-column prop="gpuLabel" label="GPU 资源" width="125" />
        <el-table-column label="进度" min-width="210"><template #default="{row}"><div class="progress-cell"><span>{{ row.progress }}%</span><el-progress :percentage="row.progress" :show-text="false" :stroke-width="6" /></div></template></el-table-column>
        <el-table-column label="状态" width="110"><template #default="{row}"><StatusPill v-bind="statusMeta(row.status)" /></template></el-table-column>
        <el-table-column label="操作" width="120" align="center"><template #default="{ row }"><div class="table-actions"><span class="table-link" @click="selected=row">详情</span><span v-if="['completed','failed','terminated'].includes(row.status)" class="danger-link" @click="removeJob(row)">删除</span></div></template></el-table-column>
      </el-table>
      <div v-if="rows.some((item) => item.status === 'queued')" class="resource-warning"><el-icon><Warning /></el-icon><span>GPU 不足，等待中的训练不会抢占推理服务；请手动停止占用中的推理部署。</span><el-button link type="primary" @click="router.push({ name: 'deployments' })">前往部署管理</el-button></div>
    </PanelCard>

    <PanelCard v-if="selected" class="section-gap training-detail" flush>
      <div class="training-title"><strong>{{ selected.name }}</strong><div><el-button v-if="['running','queued'].includes(selected.status)" type="danger" plain @click="stopJob">终止任务</el-button></div></div>
      <div class="training-progress"><span>进度</span><el-progress :percentage="selected.progress" :stroke-width="7" /><span>{{ selected.progress }}%</span><span>步骤 <b>{{ selected.step.toLocaleString() }} / {{ selected.totalSteps.toLocaleString() }}</b></span><span>Epoch <b>{{ selected.epoch }}</b></span><span>预计剩余 <b>{{ selected.eta ?? '—' }}</b></span></div>
      <div v-if="useMocks" class="training-charts">
        <div class="chart-card"><h3>训练损失</h3><BaseChart :option="lossOption()" height="150px" /></div>
        <div class="chart-card"><h3>验证损失</h3><BaseChart :option="lossOption(.18)" height="150px" /></div>
        <div class="chart-card"><h3>学习率</h3><BaseChart :option="learningRateOption" height="150px" /></div>
        <div class="metric-blocks"><span><small>Train Loss</small><b>1.284</b></span><span><small>Eval Loss</small><b>1.391</b></span><span><small>Tokens/s</small><b>2,480</b></span><span><small>显存</small><b>21.6 / 24 GB</b></span></div>
      </div>
      <div v-else class="reported-metrics">
        <h3>任务上报指标</h3>
        <div v-if="reportedMetricEntries.length" class="reported-metric-grid"><span v-for="metric in reportedMetricEntries" :key="metric.key"><small>{{ metric.label }}</small><b>{{ metric.value }}</b></span></div>
        <pre v-if="selected.metrics && Object.keys(selected.metrics).length">{{ JSON.stringify(selected.metrics, null, 2) }}</pre>
        <el-empty v-else description="训练容器尚未上报指标" :image-size="54" />
      </div>
      <div v-if="selectedGpus.length" class="training-bottom">
        <div class="training-gpus"><GpuCard v-for="gpu in selectedGpus" :key="gpu.index" :gpu="gpu" compact /></div>
      </div>
      <div class="checkpoint-empty">
        <div class="artifact-heading"><h3>任务产物</h3><el-button v-if="selected.status === 'completed'" type="primary" size="small" :loading="publishBusy" @click="publishModel">{{ selected.publishedModelAssetId ? '部署已发布模型' : '发布并部署模型' }}</el-button></div>
        <el-table v-if="artifactRows.length" v-loading="artifactsBusy" :data="artifactRows" size="small">
          <el-table-column label="类型" width="105"><template #default="{row}"><el-tag effect="plain">{{ artifactLabel(row.kind) }}</el-tag></template></el-table-column>
          <el-table-column prop="path" label="受控路径" min-width="280"><template #default="{row}"><code>{{ row.path }}</code></template></el-table-column>
          <el-table-column label="文件" width="80"><template #default="{row}">{{ row.file_count }}</template></el-table-column>
          <el-table-column label="大小" width="100"><template #default="{row}">{{ formatBytes(row.size_bytes) }}</template></el-table-column>
          <el-table-column label="操作" width="95"><template #default="{row}"><el-button link type="primary" :icon="Download" @click="downloadArtifact(row)">导出</el-button></template></el-table-column>
        </el-table>
        <el-alert v-else-if="artifactError" :title="artifactError" type="warning" :closable="false" show-icon />
        <template v-else-if="!terminalStatuses.includes(selected.status) && reportedPaths.length"><div v-for="artifact in reportedPaths" :key="artifact.label" class="artifact-row"><span>{{ artifact.label }}</span><code>{{ artifact.path }}</code></div><p class="artifact-hint">只有成功结束并通过独立安全复核的产物才会开放压缩导出。</p></template>
        <el-empty v-else :description="terminalStatuses.includes(selected.status) ? '任务没有通过安全校验的可导出产物' : '任务尚未产生 checkpoint 或模型产物'" :image-size="54" />
      </div>
    </PanelCard>
    <PanelCard v-else class="section-gap"><el-empty description="暂无训练任务" /></PanelCard>

    <el-dialog v-model="createVisible" title="创建训练任务" width="min(760px, 95vw)">
      <el-form label-position="top">
        <div class="two-column-form"><el-form-item label="任务名称" required><el-input v-model="form.name" placeholder="例如 sft-qlora-finance" /></el-form-item><el-form-item label="训练阶段"><el-radio-group v-model="form.stage"><el-radio-button value="CPT">继续预训练 CPT</el-radio-button><el-radio-button value="SFT">指令微调 SFT</el-radio-button></el-radio-group></el-form-item></div>
        <div class="two-column-form"><el-form-item label="基础模型" required><el-select v-model="form.modelAssetId" style="width:100%" placeholder="选择已校验模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id"/></el-select></el-form-item><el-form-item label="数据集版本" required><el-select v-model="form.datasetId" style="width:100%" placeholder="选择匹配用途的数据集"><el-option v-for="dataset in datasetOptions.filter((item) => item.status === 'available' && item.purpose === form.stage)" :key="dataset.id" :label="`${dataset.name} ${dataset.version}`" :value="dataset.id"/></el-select></el-form-item></div>
        <el-form-item label="训练算法"><el-radio-group v-model="form.algorithm"><el-radio-button value="LoRA">LoRA</el-radio-button><el-radio-button v-if="form.stage==='SFT'" value="QLoRA">QLoRA</el-radio-button><el-radio-button v-if="form.stage==='SFT'" value="Freeze">Freeze</el-radio-button></el-radio-group><p v-if="form.stage==='CPT'" class="form-help">继续预训练在本版本固定使用 LoRA。</p></el-form-item>
        <div class="four-column-form"><el-form-item label="整卡选择" required><el-select v-model="form.gpuIds" multiple collapse-tags :max-collapse-tags="2" style="width:100%"><el-option v-for="gpu in gpuOptions" :key="gpu.index" :label="`GPU ${gpu.index} · ${gpuStateText(gpu)}`" :value="gpu.index" /></el-select></el-form-item><el-form-item label="Epoch"><el-input-number v-model="form.epochs" :min="0.1" :max="100" :step="0.5"/></el-form-item><el-form-item label="学习率"><el-input-number v-model="form.learningRate" :min="0.0000001" :max="1" :step="0.0001" :precision="7"/></el-form-item><el-form-item label="单卡批大小"><el-input-number v-model="form.batchSize" :min="1" :max="128"/></el-form-item></div>
        <div class="four-column-form"><el-form-item label="截断长度"><el-input-number v-model="form.cutoffLen" :min="128" :max="65536" :step="128"/></el-form-item><el-form-item label="梯度累积"><el-input-number v-model="form.gradientAccumulation" :min="1" :max="4096"/></el-form-item><el-form-item label="日志间隔"><el-input-number v-model="form.loggingSteps" :min="1" :max="100000"/></el-form-item><el-form-item label="保存间隔"><el-input-number v-model="form.saveSteps" :min="1" :max="1000000"/></el-form-item></div>
        <div class="four-column-form"><el-form-item label="Warmup 比例"><el-input-number v-model="form.warmupRatio" :min="0" :max="1" :step="0.01"/></el-form-item><el-form-item label="随机种子"><el-input-number v-model="form.seed" :min="0" :max="2147483647"/></el-form-item><el-form-item label="最大样本数"><el-input-number v-model="form.maxSamples" :min="1" :max="10000000" placeholder="全部"/></el-form-item><el-form-item v-if="form.algorithm==='Freeze'" label="可训练层数"><el-input-number v-model="form.freezeTrainableLayers" :min="1" :max="256"/></el-form-item></div>
        <div v-if="form.algorithm!=='Freeze'" class="three-column-form"><el-form-item label="LoRA Rank"><el-input-number v-model="form.loraRank" :min="1" :max="1024"/></el-form-item><el-form-item label="LoRA Alpha"><el-input-number v-model="form.loraAlpha" :min="1" :max="4096"/></el-form-item><el-form-item label="LoRA Dropout"><el-input-number v-model="form.loraDropout" :min="0" :max="0.99" :step="0.01"/></el-form-item></div>
        <el-form-item v-if="form.stage === 'SFT'" label="LLaMA-Factory 模板"><el-select v-model="form.template" style="width:100%"><el-option label="Qwen / Qwen2" value="qwen"/><el-option label="Llama 3" value="llama3"/><el-option label="ChatML" value="chatml"/><el-option label="Gemma" value="gemma"/></el-select><p class="form-help">模板必须与基础模型的对话格式一致，否则训练样本会被错误编码。</p></el-form-item>
        <el-collapse><el-collapse-item title="查看节点最终白名单参数" name="config"><pre class="config-preview">{{ JSON.stringify(trainingConfigPreview, null, 2) }}</pre></el-collapse-item></el-collapse>
        <el-alert title="训练产物" description="系统保存原始 checkpoint，并在 LoRA/QLoRA 任务完成后提供 Adapter；合并模型由受控产物流程生成后才能部署。" type="info" :closable="false" show-icon />
        <el-alert title="GPU 调度提示" description="训练任务非抢占式排队。GPU 不足时，需要管理员手动停止推理服务后才会自动启动。" type="warning" :closable="false" show-icon />
      </el-form>
      <template #footer><el-button @click="createVisible=false">取消</el-button><el-button type="primary" :loading="createBusy" @click="createJob">创建任务</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.progress-cell{display:grid;grid-template-columns:36px 1fr;align-items:center;gap:8px}.resource-warning{display:flex;align-items:center;gap:8px;padding:9px 18px;color:#d66e08;background:#fff8ed;font-size:12px}.resource-warning .el-button{margin-left:auto}.training-title{display:flex;align-items:center;justify-content:space-between;padding:13px 18px;border-bottom:1px solid #e2e8ef}.training-progress{display:grid;grid-template-columns:auto minmax(120px,260px) auto repeat(3,auto);align-items:center;gap:13px;padding:12px 18px;font-size:12px}.training-progress b{margin-left:5px;font-weight:550}.training-charts{display:grid;grid-template-columns:repeat(3,1fr) 1.3fr;gap:10px;padding:0 18px 12px}.chart-card,.metric-blocks,.checkpoint-card{border:1px solid #e1e7ef;border-radius:8px}.chart-card{padding:12px}.chart-card h3,.checkpoint-card h3{margin:0;font-size:12px}.metric-blocks{display:grid;grid-template-columns:1fr 1fr;gap:1px;overflow:hidden;background:#e1e7ef}.metric-blocks span{display:flex;justify-content:center;flex-direction:column;padding:12px;background:#fff}.metric-blocks small{color:#6f7b8e}.metric-blocks b{margin-top:6px;font-size:19px}.training-bottom{padding:0 18px 16px}.training-gpus{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.checkpoint-card{padding:12px}.two-column-form{display:grid;grid-template-columns:1fr 1fr;gap:14px}.four-column-form{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.form-help{margin:7px 0 0 12px;color:#798598;font-size:12px}
.reported-metrics,.checkpoint-empty{margin:0 18px 14px;padding:14px;border:1px solid #e1e7ef;border-radius:8px}.reported-metrics h3,.checkpoint-empty h3{margin:0;font-size:13px}.reported-metrics pre,.config-preview{margin:10px 0 0;padding:12px;border-radius:6px;color:#d9e7fb;background:#071b32;font-size:11px;overflow:auto}.reported-metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:10px}.reported-metric-grid span{display:flex;min-width:0;flex-direction:column;padding:9px;border-radius:6px;background:#f5f7fa}.reported-metric-grid small{overflow:hidden;color:#718095;text-overflow:ellipsis;white-space:nowrap}.reported-metric-grid b{margin-top:4px;overflow:hidden;text-overflow:ellipsis}.artifact-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}.artifact-row{display:grid;grid-template-columns:120px minmax(0,1fr);gap:10px;padding:9px 0;border-bottom:1px solid #edf1f5;font-size:12px}.artifact-row code,.checkpoint-empty :deep(.el-table code){overflow:hidden;color:#536176;text-overflow:ellipsis;white-space:nowrap}.artifact-hint{margin:10px 0 0;color:#798598;font-size:12px}.checkpoint-empty .el-alert{margin-top:12px}.three-column-form{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.config-preview{max-height:230px;margin:0}.el-collapse{margin-bottom:14px}
@media(max-width:1250px){.training-charts{grid-template-columns:repeat(2,1fr)}.training-bottom{grid-template-columns:1fr}.four-column-form{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.training-progress{grid-template-columns:auto 1fr auto}.training-progress>span:nth-last-child(-n+3){display:none}.training-charts,.two-column-form,.three-column-form,.four-column-form{grid-template-columns:1fr}.training-gpus{grid-template-columns:repeat(2,1fr)}}
</style>
