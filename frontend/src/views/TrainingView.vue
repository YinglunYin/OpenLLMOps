<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { CircleCheck, CloseBold, Clock, Download, Plus, VideoPlay, Warning } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import GpuCard from '@/components/GpuCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { Dataset, GpuDevice, ModelAsset, StatusTone, TrainingJob } from '@/types/domain'

const rows = ref<TrainingJob[]>([])
const selected = ref<TrainingJob | null>(null)
const mockGpus = ref<GpuDevice[]>([])
const modelOptions = ref<ModelAsset[]>([])
const datasetOptions = ref<Dataset[]>([])
const createVisible = ref(false)
const createBusy = ref(false)
const form = reactive({ name: '', stage: 'SFT', algorithm: 'LoRA', modelAssetId: '', datasetId: '', gpuCount: 1, epochs: 3, learningRate: 0.0002, batchSize: 2, gradientAccumulation: 8, loraRank: 8, template: 'qwen' })

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
const artifacts = computed(() => selected.value ? [
  { label: '输出目录', path: selected.value.outputDir },
  { label: 'Checkpoint', path: selected.value.checkpointPath },
  { label: 'LoRA Adapter', path: selected.value.adapterPath },
  { label: '合并模型', path: selected.value.mergedModelPath },
].filter((item): item is { label: string; path: string } => Boolean(item.path)) : [])

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
  await ElMessageBox.confirm('终止后将保存最近一个完整 checkpoint 并释放 GPU，当前 step 不可恢复。', `终止 ${selected.value.name}`, { type: 'warning' })
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

async function createJob() {
  if (!form.name || !form.modelAssetId || !form.datasetId) { ElMessage.warning('请填写任务名称并选择模型、数据集'); return }
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
    if (import.meta.env.VITE_USE_MOCKS === 'true') mockGpus.value = (await import('@/mock/data')).gpuDevices
    ;[rows.value, modelOptions.value, datasetOptions.value] = await Promise.all([api.training.list(), api.models.list(), api.datasets.list()])
    selected.value = rows.value[0] ?? null
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '训练任务加载失败')
  }
})
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
        <el-table-column label="操作" width="70" align="center"><template #default><el-dropdown><el-button link>•••</el-button><template #dropdown><el-dropdown-menu><el-dropdown-item>查看详情</el-dropdown-item><el-dropdown-item>查看日志</el-dropdown-item><el-dropdown-item>克隆任务</el-dropdown-item></el-dropdown-menu></template></el-dropdown></template></el-table-column>
      </el-table>
      <div v-if="rows.some((item) => item.status === 'queued')" class="resource-warning"><el-icon><Warning /></el-icon><span>GPU 不足，等待中的训练不会抢占推理服务；请手动停止占用中的推理部署。</span><el-button link type="primary">前往部署管理</el-button></div>
    </PanelCard>

    <PanelCard v-if="selected" class="section-gap training-detail" flush>
      <div class="training-title"><strong>{{ selected.name }}</strong><div><el-button v-if="selected.status === 'running'" type="danger" plain @click="stopJob">终止任务</el-button><el-button>查看日志</el-button></div></div>
      <div class="training-progress"><span>进度</span><el-progress :percentage="selected.progress" :stroke-width="7" /><span>{{ selected.progress }}%</span><span>步骤 <b>{{ selected.step.toLocaleString() }} / {{ selected.totalSteps.toLocaleString() }}</b></span><span>Epoch <b>{{ selected.epoch }}</b></span><span>预计剩余 <b>{{ selected.eta ?? '—' }}</b></span></div>
      <div v-if="useMocks" class="training-charts">
        <div class="chart-card"><h3>训练损失</h3><BaseChart :option="lossOption()" height="150px" /></div>
        <div class="chart-card"><h3>验证损失</h3><BaseChart :option="lossOption(.18)" height="150px" /></div>
        <div class="chart-card"><h3>学习率</h3><BaseChart :option="learningRateOption" height="150px" /></div>
        <div class="metric-blocks"><span><small>Train Loss</small><b>1.284</b></span><span><small>Eval Loss</small><b>1.391</b></span><span><small>Tokens/s</small><b>2,480</b></span><span><small>显存</small><b>21.6 / 24 GB</b></span></div>
      </div>
      <div v-else class="reported-metrics">
        <h3>任务上报指标</h3>
        <pre v-if="selected.metrics && Object.keys(selected.metrics).length">{{ JSON.stringify(selected.metrics, null, 2) }}</pre>
        <el-empty v-else description="训练容器尚未上报指标" :image-size="54" />
      </div>
      <div v-if="useMocks" class="training-bottom">
        <div class="training-gpus"><GpuCard v-for="gpu in mockGpus" :key="gpu.index" :gpu="{...gpu,state:'training'}" compact /></div>
        <div class="checkpoint-card"><h3>检查点</h3><el-table :data="[{name:'checkpoint-500',step:500,time:'2024-05-20 10:15'},{name:'checkpoint-1000',step:1000,time:'2024-05-20 10:28'}]" size="small"><el-table-column prop="name" label="检查点" min-width="130"/><el-table-column prop="step" label="步骤" width="75"/><el-table-column prop="time" label="保存时间" width="145"/><el-table-column label="操作" width="150"><template #default><div class="table-actions"><span class="table-link"><el-icon><Download/></el-icon> 导出</span><span class="table-link">从此恢复</span></div></template></el-table-column></el-table></div>
      </div>
      <div v-else class="checkpoint-empty"><template v-if="artifacts.length"><h3>任务产物</h3><div v-for="artifact in artifacts" :key="artifact.label" class="artifact-row"><span>{{ artifact.label }}</span><code>{{ artifact.path }}</code></div><el-alert title="产物导出端点尚未接入" description="当前仅展示控制面上报的受控路径。" type="info" :closable="false"/></template><el-empty v-else description="任务尚未产生 checkpoint 或模型产物" :image-size="54" /></div>
    </PanelCard>
    <PanelCard v-else class="section-gap"><el-empty description="暂无训练任务" /></PanelCard>

    <el-dialog v-model="createVisible" title="创建训练任务" width="min(760px, 95vw)">
      <el-form label-position="top">
        <div class="two-column-form"><el-form-item label="任务名称" required><el-input v-model="form.name" placeholder="例如 sft-qlora-finance" /></el-form-item><el-form-item label="训练阶段"><el-radio-group v-model="form.stage"><el-radio-button value="CPT">继续预训练 CPT</el-radio-button><el-radio-button value="SFT">指令微调 SFT</el-radio-button></el-radio-group></el-form-item></div>
        <div class="two-column-form"><el-form-item label="基础模型" required><el-select v-model="form.modelAssetId" style="width:100%" placeholder="选择已校验模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id"/></el-select></el-form-item><el-form-item label="数据集版本" required><el-select v-model="form.datasetId" style="width:100%" placeholder="选择匹配用途的数据集"><el-option v-for="dataset in datasetOptions.filter((item) => item.status === 'available' && item.purpose === form.stage)" :key="dataset.id" :label="`${dataset.name} ${dataset.version}`" :value="dataset.id"/></el-select></el-form-item></div>
        <el-form-item label="训练算法"><el-radio-group v-model="form.algorithm"><el-radio-button value="LoRA">LoRA</el-radio-button><el-radio-button v-if="form.stage==='SFT'" value="QLoRA">QLoRA</el-radio-button><el-radio-button v-if="form.stage==='SFT'" value="Freeze">Freeze</el-radio-button></el-radio-group><p v-if="form.stage==='CPT'" class="form-help">继续预训练在本版本固定使用 LoRA。</p></el-form-item>
        <div class="four-column-form"><el-form-item label="整卡数量"><el-input-number v-model="form.gpuCount" :min="1" :max="4"/></el-form-item><el-form-item label="Epoch"><el-input-number v-model="form.epochs" :min="1"/></el-form-item><el-form-item label="学习率"><el-input-number v-model="form.learningRate" :min="0.0000001" :max="1" :step="0.0001" :precision="7"/></el-form-item><el-form-item label="批大小"><el-input-number v-model="form.batchSize" :min="1"/></el-form-item></div>
        <div class="two-column-form"><el-form-item label="梯度累积"><el-input-number v-model="form.gradientAccumulation" :min="1"/></el-form-item><el-form-item v-if="form.algorithm!=='Freeze'" label="LoRA Rank"><el-input-number v-model="form.loraRank" :min="1"/></el-form-item></div>
        <el-form-item v-if="form.stage === 'SFT'" label="LLaMA-Factory 模板"><el-select v-model="form.template" style="width:100%"><el-option label="Qwen / Qwen2" value="qwen"/><el-option label="Llama 3" value="llama3"/><el-option label="ChatML" value="chatml"/><el-option label="Gemma" value="gemma"/></el-select><p class="form-help">模板必须与基础模型的对话格式一致，否则训练样本会被错误编码。</p></el-form-item>
        <el-alert title="训练产物" description="系统保存原始 checkpoint，并在 LoRA/QLoRA 任务完成后提供 Adapter；合并模型由受控产物流程生成后才能部署。" type="info" :closable="false" show-icon />
        <el-alert title="GPU 调度提示" description="训练任务非抢占式排队。GPU 不足时，需要管理员手动停止推理服务后才会自动启动。" type="warning" :closable="false" show-icon />
      </el-form>
      <template #footer><el-button @click="createVisible=false">取消</el-button><el-button type="primary" :loading="createBusy" @click="createJob">创建任务</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.progress-cell{display:grid;grid-template-columns:36px 1fr;align-items:center;gap:8px}.resource-warning{display:flex;align-items:center;gap:8px;padding:9px 18px;color:#d66e08;background:#fff8ed;font-size:12px}.resource-warning .el-button{margin-left:auto}.training-title{display:flex;align-items:center;justify-content:space-between;padding:13px 18px;border-bottom:1px solid #e2e8ef}.training-progress{display:grid;grid-template-columns:auto minmax(120px,260px) auto repeat(3,auto);align-items:center;gap:13px;padding:12px 18px;font-size:12px}.training-progress b{margin-left:5px;font-weight:550}.training-charts{display:grid;grid-template-columns:repeat(3,1fr) 1.3fr;gap:10px;padding:0 18px 12px}.chart-card,.metric-blocks,.checkpoint-card{border:1px solid #e1e7ef;border-radius:8px}.chart-card{padding:12px}.chart-card h3,.checkpoint-card h3{margin:0;font-size:12px}.metric-blocks{display:grid;grid-template-columns:1fr 1fr;gap:1px;overflow:hidden;background:#e1e7ef}.metric-blocks span{display:flex;justify-content:center;flex-direction:column;padding:12px;background:#fff}.metric-blocks small{color:#6f7b8e}.metric-blocks b{margin-top:6px;font-size:19px}.training-bottom{display:grid;grid-template-columns:1.5fr 1fr;gap:10px;padding:0 18px 16px}.training-gpus{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.checkpoint-card{padding:12px}.two-column-form{display:grid;grid-template-columns:1fr 1fr;gap:14px}.four-column-form{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.form-help{margin:7px 0 0 12px;color:#798598;font-size:12px}
.reported-metrics,.checkpoint-empty{margin:0 18px 14px;padding:14px;border:1px solid #e1e7ef;border-radius:8px}.reported-metrics h3,.checkpoint-empty h3{margin:0 0 10px;font-size:13px}.reported-metrics pre{margin:0;padding:12px;border-radius:6px;color:#d9e7fb;background:#071b32;font-size:11px;overflow:auto}.artifact-row{display:grid;grid-template-columns:120px minmax(0,1fr);gap:10px;padding:9px 0;border-bottom:1px solid #edf1f5;font-size:12px}.artifact-row code{overflow:hidden;color:#536176;text-overflow:ellipsis;white-space:nowrap}.checkpoint-empty .el-alert{margin-top:12px}
@media(max-width:1250px){.training-charts{grid-template-columns:repeat(2,1fr)}.training-bottom{grid-template-columns:1fr}.four-column-form{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.training-progress{grid-template-columns:auto 1fr auto}.training-progress>span:nth-last-child(-n+3){display:none}.training-charts,.two-column-form,.four-column-form{grid-template-columns:1fr}.training-gpus{grid-template-columns:repeat(2,1fr)}}
</style>
