<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { CircleCheck, Clock, CloseBold, CopyDocument, Plus, Promotion, VideoPause, VideoPlay } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRoute } from 'vue-router'

import BaseChart from '@/components/BaseChart.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { Deployment, ModelAsset, StatusTone } from '@/types/domain'

const rows = ref<Deployment[]>([])
const route = useRoute()
const selected = ref<Deployment | null>(null)
const trendTimes = ref<string[]>([])
const modelOptions = ref<ModelAsset[]>([])
const activeTab = ref('overview')
const createVisible = ref(false)
const configMode = ref<'simple' | 'advanced'>('simple')
const createBusy = ref(false)
const editingId = ref<string | null>(null)
const form = reactive({ name: '', servedModelName: '', modelAssetId: '', serviceType: 'generation', gpuCount: 1, maxModelLen: 32768, gpuMemoryUtilization: .9, dtype: 'auto', advancedArgs: '--enable-prefix-caching\n--max-num-seqs 64' })

watch(() => form.serviceType, (serviceType) => {
  const selectedModel = modelOptions.value.find((model) => model.id === form.modelAssetId)
  if (selectedModel && (serviceType === 'embedding') !== (selectedModel.type === 'embedding')) form.modelAssetId = ''
})

function resetForm(model?: ModelAsset) {
  Object.assign(form, {
    name: '', servedModelName: '', modelAssetId: model?.id ?? '',
    serviceType: model?.type ?? 'generation', gpuCount: 1, maxModelLen: 32768,
    gpuMemoryUtilization: .9, dtype: 'auto', advancedArgs: '--enable-prefix-caching\n--max-num-seqs 64',
  })
}

function openCreate(model?: ModelAsset) {
  editingId.value = null
  resetForm(model)
  createVisible.value = true
}

function serializeAdvancedArgs(args: Record<string, unknown>): string {
  return Object.entries(args).map(([key, value]) => {
    const flag = `--${key.replaceAll('_', '-')}`
    if (value === true) return flag
    if (typeof value === 'string' && !/\s/.test(value)) return `${flag} ${value}`
    return `${flag} ${JSON.stringify(value)}`
  }).join('\n')
}

function openEdit(row: Deployment) {
  if (!['stopped', 'error'].includes(row.status)) {
    ElMessage.warning('请先停止部署，再编辑配置')
    return
  }
  editingId.value = row.id
  Object.assign(form, {
    name: row.name,
    servedModelName: row.model,
    modelAssetId: row.modelAssetId,
    serviceType: row.serviceType,
    gpuCount: row.gpuIds.length || 1,
    maxModelLen: Number(row.simplifiedConfig?.max_model_len ?? 32768),
    gpuMemoryUtilization: Number(row.simplifiedConfig?.gpu_memory_utilization ?? .9),
    dtype: String(row.simplifiedConfig?.dtype ?? 'auto'),
    advancedArgs: serializeAdvancedArgs(row.vllmArgs ?? {}),
  })
  createVisible.value = true
}

const counts = computed(() => ({
  running: rows.value.filter((item) => item.status === 'running').length,
  stopped: rows.value.filter((item) => item.status === 'stopped').length,
  queued: rows.value.filter((item) => item.status === 'queued').length,
  error: rows.value.filter((item) => item.status === 'error').length,
}))

const statusMap: Record<Deployment['status'], { text: string; tone: StatusTone }> = {
  running: { text: '运行中', tone: 'success' }, stopped: { text: '已停止', tone: 'info' }, queued: { text: '等待 GPU', tone: 'warning' }, error: { text: '异常', tone: 'danger' }, starting: { text: '启动中', tone: 'primary' }, stopping: { text: '停止中', tone: 'info' },
}
const statusMeta = (status: Deployment['status']) => statusMap[status]

const lineOption = (values: number[], color = '#1769f5') => ({
  tooltip: { trigger: 'axis' }, grid: { top: 18, left: 34, right: 10, bottom: 24 },
  xAxis: { type: 'category', boundaryGap: false, data: trendTimes.value, axisLabel: { color: '#7b8798', fontSize: 10 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
  yAxis: { type: 'value', axisLabel: { color: '#7b8798', fontSize: 10 }, splitLine: { lineStyle: { color: '#eef2f6', type: 'dashed' } } },
  series: [{ type: 'line', data: values, showSymbol: false, smooth: true, lineStyle: { color, width: 2 }, areaStyle: { color: `${color}12` } }],
})

function selectRow(row: Deployment) { selected.value = row }

async function refreshRows(preferredId?: string) {
  rows.value = await api.deployments.list()
  selected.value = rows.value.find((item) => item.id === preferredId) ?? rows.value[0] ?? null
}

async function toggleDeployment(row: Deployment) {
  if (row.status === 'stopping') return
  const nextRunning = !['running', 'starting', 'queued'].includes(row.status)
  if (!nextRunning) await ElMessageBox.confirm('停止后将释放整卡资源，现有请求会被中断。', `停止 ${row.name}`, { type: 'warning' })
  try {
    if (useMocks) row.status = nextRunning ? 'starting' : 'stopped'
    if (nextRunning) await api.deployments.start(row.id)
    else await api.deployments.stop(row.id)
    if (useMocks) row.status = nextRunning ? 'running' : 'stopped'
    else await refreshRows(row.id)
    ElMessage.success(nextRunning ? '部署已进入启动队列' : '部署停止指令已提交')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '部署状态操作失败')
  }
}

async function submitDeployment() {
  if (!form.name || !form.modelAssetId) { ElMessage.warning('请填写部署名称并选择模型'); return }
  createBusy.value = true
  try {
    if (editingId.value) await api.deployments.update(editingId.value, { ...form, configMode: configMode.value })
    else await api.deployments.create({ ...form, configMode: configMode.value })
    createVisible.value = false
    if (!useMocks) await refreshRows()
    ElMessage.success(editingId.value ? '部署配置已更新，可重新启动' : '部署任务已创建，调度器将在有足够整卡资源时启动')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '部署保存失败')
  } finally {
    createBusy.value = false
  }
}

async function removeDeployment(row: Deployment) {
  try {
    await ElMessageBox.confirm('只能删除已停止或失败的部署；运行中的实例不会被隐式停止。', `删除 ${row.name}`, { type: 'warning' })
    await api.deployments.remove(row.id)
    await refreshRows()
    ElMessage.success('部署已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '删除部署失败')
  }
}

onMounted(async () => {
  try {
    if (import.meta.env.VITE_USE_MOCKS === 'true') trendTimes.value = (await import('@/mock/data')).trendTimes
    ;[rows.value, modelOptions.value] = await Promise.all([api.deployments.list(), api.models.list()])
    selected.value = rows.value[0] ?? null
    const requestedModelId = typeof route.query.model === 'string' ? route.query.model : null
    const requestedModel = modelOptions.value.find((model) => model.id === requestedModelId && model.status === 'available')
    if (requestedModel) openCreate(requestedModel)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '部署数据加载失败')
  }
})
</script>

<template>
  <div>
    <PageHeader title="模型部署" subtitle="基于 vLLM 管理生成与 Embedding 服务，统一提供 OpenAI Compatible 接口">
      <el-button type="primary" :icon="Plus" @click="openCreate()">创建部署</el-button>
    </PageHeader>

    <div class="stats-grid">
      <StatCard label="运行中" :value="counts.running" :icon="VideoPlay" tone="green" />
      <StatCard label="已停止" :value="counts.stopped" :icon="VideoPause" tone="slate" />
      <StatCard label="等待 GPU" :value="counts.queued" :icon="Clock" tone="orange" />
      <StatCard label="异常" :value="counts.error" :icon="CloseBold" tone="red" />
    </div>

    <PanelCard class="section-gap" flush>
      <el-table :data="rows" highlight-current-row row-key="id" @current-change="selectRow">
        <el-table-column width="50"><template #default="{ row }"><span :class="['row-radio', { active: selected?.id === row.id }]" /></template></el-table-column>
        <el-table-column prop="name" label="部署名称" min-width="220" />
        <el-table-column prop="model" label="模型" min-width="160" />
        <el-table-column label="服务类型" width="116"><template #default="{ row }"><StatusPill :text="row.serviceType === 'generation' ? '文本生成' : 'Embedding'" :tone="row.serviceType === 'generation' ? 'primary' : 'info'" /></template></el-table-column>
        <el-table-column prop="gpuLabel" label="GPU" width="105" />
        <el-table-column prop="parallelism" label="并行策略" width="110" />
        <el-table-column label="状态" width="110"><template #default="{ row }"><StatusPill v-bind="statusMeta(row.status)" dot /></template></el-table-column>
        <el-table-column label="访问地址" min-width="210"><template #default="{ row }"><a v-if="row.endpoint" class="table-link">{{ row.endpoint }}</a><span v-else class="empty-dash">—</span></template></el-table-column>
        <el-table-column label="操作" width="220" fixed="right">
          <template #default="{ row }"><div class="table-actions"><span class="table-link" @click="selected = row">详情</span><span class="table-link" @click="toggleDeployment(row)">{{ ['running','starting','queued'].includes(row.status) ? '停止' : row.status === 'stopping' ? '停止中' : '启动' }}</span><span class="table-link" @click="openEdit(row)">编辑</span><span class="danger-link" @click="removeDeployment(row)">删除</span></div></template>
        </el-table-column>
      </el-table>
    </PanelCard>

    <PanelCard v-if="selected" class="section-gap deployment-detail" flush>
      <div class="detail-title">
        <div><strong>部署详情：{{ selected.name }}</strong><StatusPill v-bind="statusMeta(selected.status)" /></div>
        <div v-if="useMocks"><span>创建时间：2024-05-20 10:35:21</span><el-divider direction="vertical" /><span>运行时长：2 天 3 小时 18 分钟</span></div>
      </div>
      <el-tabs v-model="activeTab" class="detail-tabs">
        <el-tab-pane label="概览" name="overview">
          <div class="deployment-metrics">
            <div class="health-card"><span>服务状态</span><div><el-icon color="#12a865" :size="29"><CircleCheck /></el-icon><strong>{{ statusMeta(selected.status).text }}</strong></div><small>{{ selected.status === 'running' ? '服务实例正在运行' : '以控制面实际状态为准' }}</small></div>
            <template v-if="useMocks">
              <div class="metric-chart"><span>请求速率（QPS）</span><strong>{{ selected.qps ?? 0 }}</strong><BaseChart :option="lineOption([18, 14, 20, 16, 19, 17, 21])" height="108px" /></div>
              <div class="metric-chart"><span>TTFT（p50）</span><strong>{{ selected.ttft ?? 0 }} <small>ms</small></strong><BaseChart :option="lineOption([202, 184, 196, 158, 175, 186, 181], '#7c3aed')" height="108px" /></div>
              <div class="metric-chart"><span>KV Cache 命中率</span><strong>{{ selected.kvHitRate ?? 0 }}%</strong><BaseChart :option="lineOption([78, 91, 87, 94, 88, 93, 92], '#12a865')" height="108px" /></div>
            </template>
            <el-empty v-else class="metrics-empty" description="时序指标端点尚未接入" :image-size="52" />
          </div>
          <div class="deployment-info-grid">
            <div class="info-card"><h3>服务端点</h3><p>/v1/chat/completions <el-tag size="small">POST</el-tag></p><p>/v1/completions <el-tag size="small">POST</el-tag></p><p v-if="selected.serviceType === 'embedding'">/v1/embeddings <el-tag size="small">POST</el-tag></p></div>
            <div class="info-card"><h3>GPU 资源分配（整卡）</h3><strong>{{ selected.gpuLabel }}</strong><p>{{ selected.parallelism }}</p></div>
            <div class="info-card"><h3>访问地址</h3><strong>{{ selected.endpoint ?? '等待调度' }}</strong><el-button v-if="selected.endpoint" size="small" :icon="CopyDocument">复制地址</el-button><p>内网 OpenAI Compatible 网关</p></div>
            <div class="info-card"><h3>API Key</h3><template v-if="useMocks"><el-input model-value="sk-••••••••••••••••••••" readonly show-password /><el-button type="primary" size="small" :icon="CopyDocument">复制</el-button></template><p v-else>使用系统设置中配置的 X-API-Key</p></div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="配置" name="config"><pre class="config-code">{{ JSON.stringify({ simplified_config: selected.simplifiedConfig ?? {}, vllm_args: selected.vllmArgs ?? {} }, null, 2) }}</pre></el-tab-pane>
        <el-tab-pane label="监控" name="monitor"><el-empty description="Prometheus 指标面板将在服务启动后加载" /></el-tab-pane>
        <el-tab-pane label="日志" name="logs"><pre v-if="useMocks" class="log-view">[INFO] vLLM OpenAI server started\n[INFO] model loaded on GPU 0\n[INFO] application startup complete</pre><el-empty v-else description="日志端点尚未接入" /></el-tab-pane>
      </el-tabs>
    </PanelCard>
    <PanelCard v-else class="section-gap"><el-empty description="暂无部署，请先创建部署" /></PanelCard>

    <el-dialog v-model="createVisible" :title="editingId ? '编辑模型部署' : '创建模型部署'" width="min(720px, 94vw)">
      <div class="mode-switch"><span>配置模式</span><el-segmented v-model="configMode" :options="[{ label: '简化配置', value: 'simple' }, { label: '详细参数', value: 'advanced' }]" /></div>
      <el-form label-position="top">
        <div class="two-column-form">
          <el-form-item label="部署名称" required><el-input v-model="form.name" placeholder="例如 qwen2-7b-chat" /></el-form-item>
          <el-form-item label="服务类型"><el-select v-model="form.serviceType" :disabled="Boolean(editingId)" style="width:100%"><el-option label="文本生成" value="generation" /><el-option label="Embedding" value="embedding" /></el-select></el-form-item>
        </div>
        <div class="two-column-form">
          <el-form-item label="模型资产" required><el-select v-model="form.modelAssetId" :disabled="Boolean(editingId)" filterable placeholder="选择已校验且类型匹配的模型" style="width:100%"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && (form.serviceType === 'embedding' ? item.type === 'embedding' : item.type === 'generation'))" :key="model.id" :label="model.name" :value="model.id" /></el-select></el-form-item>
          <el-form-item label="对外模型名"><el-input v-model="form.servedModelName" :disabled="Boolean(editingId)" placeholder="留空则使用部署名称" /></el-form-item>
        </div>
        <template v-if="configMode === 'simple'">
          <div class="three-column-form">
            <el-form-item label="整卡数量"><el-input-number v-model="form.gpuCount" :min="1" :max="4" /></el-form-item>
            <el-form-item label="最大上下文"><el-input-number v-model="form.maxModelLen" :min="1024" :step="1024" /></el-form-item>
            <el-form-item label="显存利用率"><el-input-number v-model="form.gpuMemoryUtilization" :min="0.1" :max="0.99" :step="0.05" /></el-form-item>
          </div>
          <el-form-item label="数据类型"><el-radio-group v-model="form.dtype"><el-radio-button value="auto">Auto</el-radio-button><el-radio-button value="float16">FP16</el-radio-button><el-radio-button value="bfloat16">BF16</el-radio-button></el-radio-group></el-form-item>
        </template>
        <el-form-item v-else label="vLLM 额外参数"><el-input v-model="form.advancedArgs" type="textarea" :rows="8" spellcheck="false" /><p class="form-help">每行一个参数。模型路径、服务端口与 GPU 可见性由系统托管，不能覆盖。</p></el-form-item>
        <el-alert title="资源策略" description="部署使用整卡独占。资源不足时进入非抢占队列，不会自动停止其他推理服务。" type="warning" :closable="false" show-icon />
      </el-form>
      <template #footer><el-button @click="createVisible = false">取消</el-button><el-button type="primary" :icon="Promotion" :loading="createBusy" @click="submitDeployment">{{ editingId ? '保存配置' : '创建部署' }}</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.row-radio { display: block; width: 17px; height: 17px; border: 1.5px solid #c8d2df; border-radius: 50%; }.row-radio.active { border: 5px solid #1769f5; }
.detail-title { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 13px 18px 0; }.detail-title > div { display: flex; align-items: center; gap: 12px; }.detail-title > div:last-child { color: #69768a; font-size: 12px; }
.detail-tabs { padding: 0 18px 16px; }.deployment-metrics { display: grid; grid-template-columns: .85fr repeat(3, 1fr); gap: 12px; }
.metrics-empty { grid-column: span 3; border: 1px dashed #dfe6ef; border-radius: 8px; }
.health-card, .metric-chart, .info-card { min-width: 0; padding: 14px; border: 1px solid #e1e7ef; border-radius: 8px; }.health-card > span, .metric-chart > span { display: block; margin-bottom: 8px; font-size: 13px; font-weight: 600; }.health-card > div { display: flex; align-items: center; gap: 8px; margin: 19px 0 7px; color: #12a865; }.health-card small { color: #728095; }.metric-chart > strong { font-size: 22px; }.metric-chart > strong small { font-size: 11px; font-weight: 500; }
.deployment-info-grid { display: grid; grid-template-columns: .8fr 1.25fr .95fr 1fr; gap: 12px; margin-top: 12px; }.info-card h3 { margin: 0 0 13px; font-size: 13px; }.info-card p { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 9px 0; color: #5f6c80; font-size: 12px; }.info-card > strong { display: block; overflow: hidden; margin-bottom: 11px; font-size: 13px; text-overflow: ellipsis; }.info-card .el-button { margin-top: 10px; }
.mini-gpus { display: grid; grid-template-columns: repeat(4,1fr); gap: 6px; }.mini-gpus span { padding: 8px 4px; text-align: center; border: 1px solid #e0e6ee; border-radius: 5px; font-size: 11px; }.mini-gpus span.allocated { border-color: #b9d8ff; background: #edf5ff; }.mini-gpus small { display:block; margin-top:4px; color:#7a8698; }
.config-code,.log-view { min-height:150px; margin:0; padding:15px; overflow:auto; border-radius:7px; color:#d9e7fb; background:#071b32; font-size:12px; line-height:1.7; }.mode-switch { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; padding:10px 12px; border-radius:7px; background:#f5f7fa; }.two-column-form { display:grid; grid-template-columns:1fr 1fr; gap:14px; }.three-column-form { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }.form-help { margin:6px 0 0; color:#7a8698; font-size:12px; }
@media (max-width:1200px){.deployment-metrics{grid-template-columns:repeat(2,1fr)}.deployment-info-grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:700px){.detail-title{align-items:flex-start;flex-direction:column}.detail-title>div:last-child{display:none}.deployment-metrics,.deployment-info-grid,.two-column-form,.three-column-form{grid-template-columns:1fr}}
</style>
