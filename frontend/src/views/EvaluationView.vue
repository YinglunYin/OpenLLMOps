<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { DataAnalysis, Download, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { Dataset, EvaluationComparisonScore, EvaluationRunDetail, EvaluationRunSummary, GpuDevice, ModelAsset, StatusTone } from '@/types/domain'

const activeTab = ref('comparison')
const taskRows = ref<EvaluationRunSummary[]>([])
const selectedTaskId = ref('')
const selectedDetail = ref<EvaluationRunDetail | null>(null)
const detailLoading = ref(false)
const modelOptions = ref<ModelAsset[]>([])
const datasetOptions = ref<Dataset[]>([])
const gpuOptions = ref<GpuDevice[]>([])
const createVisible = ref(false)
const createBusy = ref(false)
const categoryQuery = ref('')
const categoryPage = ref(1)
const categoryPageSize = 20
let refreshTimer: number | undefined
let refreshRunning = false
let detailRequestVersion = 0
const form = reactive({ name: '', baseModelAssetId: '', candidateModelAssetId: '', datasets: ['ceval', 'cmmlu'], customDatasetId: '', gpuIds: [0] as number[] })

const rows = computed(() => selectedDetail.value?.results ?? [])
const categoryRows = computed(() => selectedDetail.value?.categories ?? [])
const filteredCategoryRows = computed(() => {
  const query = categoryQuery.value.trim().toLocaleLowerCase()
  if (!query) return categoryRows.value
  return categoryRows.value.filter((item) => `${item.dataset} ${item.category}`.toLocaleLowerCase().includes(query))
})
const pagedCategoryRows = computed(() => filteredCategoryRows.value.slice((categoryPage.value - 1) * categoryPageSize, categoryPage.value * categoryPageSize))
const modelMetricRows = computed(() => {
  if (!selectedDetail.value) return []
  return [
    { role: '基线模型', model: selectedDetail.value.baseModel, metric: selectedDetail.value.baselineMetric },
    { role: '候选模型', model: selectedDetail.value.candidateModel, metric: selectedDetail.value.candidateMetric },
  ]
})

// 旧版 comparison 没有 overall 时仍可从同一批样本做加权汇总；严格报告直接使用后端值。
const overall = computed<EvaluationComparisonScore | undefined>(() => {
  if (selectedDetail.value?.overall) return selectedDetail.value.overall
  const total = rows.value.reduce((sum, item) => sum + item.samples, 0)
  if (!total) return undefined
  const before = rows.value.reduce((sum, item) => sum + item.before * item.samples, 0) / total
  const after = rows.value.reduce((sum, item) => sum + item.after * item.samples, 0) / total
  const pointChange = after - before
  return { before, after, pointChange, relativeChange: before === 0 ? null : pointChange * 100 / before }
})

const signed = (value: number) => `${value > 0 ? '+' : ''}${formatNumber(value)}`
const formatNumber = (value: number) => Number(value.toFixed(4)).toLocaleString()
const formatPercent = (value: number) => `${formatNumber(value)}%`
const formatRelative = (value: number | null) => value === null ? '—' : `${signed(value)}%`
const relativeTone = (value: number | null) => value === null ? '' : value >= 0 ? 'number-positive' : 'danger-link'
const formatCount = (correct: number | null, total: number | null) => correct === null || total === null ? '—' : `${correct.toLocaleString()} / ${total.toLocaleString()}`
const templateText = (value: 'base' | 'instruct') => value === 'base' ? 'Base' : 'Instruct'
const warningText = (value: string) => ({
  baseline_all_outputs_invalid: '基线模型全部输出均无法解析，基线得分不具备正常参考意义。',
  candidate_all_outputs_invalid: '候选模型全部输出均无法解析，候选得分不具备正常参考意义。',
})[value] ?? value

const taskStatusMap: Record<EvaluationRunSummary['status'], { text: string; tone: StatusTone }> = {
  queued: { text: '等待 GPU', tone: 'warning' },
  running: { text: '测评中', tone: 'primary' },
  completed: { text: '已完成', tone: 'success' },
  failed: { text: '失败', tone: 'danger' },
  stopping: { text: '取消中', tone: 'info' },
  terminated: { text: '已取消', tone: 'info' },
}
const taskStatusMeta = (status: EvaluationRunSummary['status']) => taskStatusMap[status]
const gpuStateText = (gpu: GpuDevice) => ({ idle: '空闲', inference: '推理占用', training: '训练占用', reserved: '已预留', unmanaged: '未纳管占用', unknown: '状态未知' })[gpu.state]

const comparisonOption = computed(() => ({
  color: ['#1769f5', '#12a865'], tooltip: { trigger: 'axis' }, legend: { top: 0, right: 10, itemWidth: 10, itemHeight: 10 }, grid: { top: 42, left: 42, right: 10, bottom: 50 },
  xAxis: { type: 'category', data: rows.value.map((item) => item.dataset), axisLabel: { color: '#69768a', interval: 0, rotate: rows.value.length > 3 ? 20 : 0 } }, yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%', color: '#69768a' }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } },
  series: [{ name: '基线模型', type: 'bar', data: rows.value.map((item) => item.before), barMaxWidth: 32 }, { name: '候选模型', type: 'bar', data: rows.value.map((item) => item.after), barMaxWidth: 32 }],
}))

const categoryOption = computed(() => {
  const chartRows = [...categoryRows.value].sort((left, right) => Math.abs(right.pointChange) - Math.abs(left.pointChange)).slice(0, 12).reverse()
  return {
    color: ['#12a865'], tooltip: { trigger: 'axis', formatter: '{b}<br/>百分点变化：{c}' }, grid: { top: 10, left: 130, right: 46, bottom: 25 },
    xAxis: { type: 'value', axisLabel: { formatter: '{value}', color: '#718095' }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } },
    yAxis: { type: 'category', data: chartRows.map((item) => `${item.dataset} / ${item.category}`), axisLabel: { color: '#566276', width: 112, overflow: 'truncate', fontSize: 11 } },
    series: [{ type: 'bar', data: chartRows.map((item) => ({ value: item.pointChange, itemStyle: { color: item.pointChange >= 0 ? '#12a865' : '#e94b5f' } })), barWidth: 12, label: { show: true, position: 'right', formatter: ({ value }: { value: number }) => signed(value), color: '#596579' } }],
  }
})

async function loadEvaluationDetail(id: string, showError = true) {
  const version = ++detailRequestVersion
  // 切换任务时先移除旧结果，避免选择器已指向新任务却仍展示上一任务的分数。
  if (selectedTaskId.value === id && selectedDetail.value?.id !== id) selectedDetail.value = null
  detailLoading.value = true
  try {
    const detail = await api.evaluations.get(id)
    if (version !== detailRequestVersion || selectedTaskId.value !== id) return
    selectedDetail.value = detail
  } catch (error) {
    if (version !== detailRequestVersion) return
    if (selectedTaskId.value === id) selectedDetail.value = null
    if (showError) ElMessage.error(error instanceof Error ? error.message : '测评详情加载失败')
  } finally {
    if (version === detailRequestVersion) detailLoading.value = false
  }
}

async function selectEvaluation(id: string, switchTab = false) {
  selectedTaskId.value = id
  categoryPage.value = 1
  if (switchTab) activeTab.value = 'comparison'
  await loadEvaluationDetail(id)
}

async function refreshEvaluations(preferredId?: string) {
  if (refreshRunning) return
  refreshRunning = true
  try {
    const tasks = await api.evaluations.list()
    taskRows.value = tasks
    const requestedId = preferredId ?? selectedTaskId.value
    const target = tasks.find((item) => item.id === requestedId) ?? tasks.find((item) => item.hasResult) ?? tasks[0]
    if (!target) {
      selectedTaskId.value = ''
      selectedDetail.value = null
      return
    }
    selectedTaskId.value = target.id
    await loadEvaluationDetail(target.id, false)
  } finally {
    refreshRunning = false
  }
}

async function createEvaluation() {
  if (!form.name || !form.baseModelAssetId || !form.candidateModelAssetId || !form.datasets.length || !form.gpuIds.length) { ElMessage.warning('请填写任务名称并选择基线、候选模型、数据集及 GPU'); return }
  if (form.datasets.includes('custom') && !form.customDatasetId) { ElMessage.warning('请选择自定义领域评测集'); return }
  createBusy.value = true
  try {
    const created = await api.evaluations.create({ ...form })
    createVisible.value = false
    await refreshEvaluations(created.id)
    activeTab.value = 'tasks'
    gpuOptions.value = await api.resources.gpus()
    ElMessage.success('测评任务已创建并进入 GPU 队列')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '测评任务创建失败')
  } finally {
    createBusy.value = false
  }
}

async function cancelEvaluation(row: EvaluationRunSummary) {
  try {
    await ElMessageBox.confirm('取消会终止当前顺序评测并释放整卡，已生成的中间结果不计入最终对比。', `取消 ${row.name}`, { type: 'warning' })
    await api.evaluations.cancel(row.id)
    await refreshEvaluations(row.id)
    ElMessage.success('取消指令已记录')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '取消测评失败')
  }
}

async function removeEvaluation(row: EvaluationRunSummary) {
  try {
    await ElMessageBox.confirm('该操作只删除已结束的任务记录。', `删除 ${row.name}`, { type: 'warning' })
    await api.evaluations.remove(row.id)
    if (selectedTaskId.value === row.id) selectedTaskId.value = ''
    await refreshEvaluations()
    ElMessage.success('测评任务已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '删除测评任务失败')
  }
}

function exportReport() {
  if (!selectedDetail.value?.hasResult) return
  const blob = new Blob([JSON.stringify({ exportedAt: new Date().toISOString(), evaluation: selectedDetail.value }, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `openllmops-evaluation-${selectedDetail.value.name}-${new Date().toISOString().slice(0, 10)}.json`
  link.click()
  URL.revokeObjectURL(url)
}

watch(categoryQuery, () => { categoryPage.value = 1 })

onMounted(async () => {
  try {
    const [tasks, models, datasets, gpus] = await Promise.all([api.evaluations.list(), api.models.list(), api.datasets.list(), api.resources.gpus()])
    taskRows.value = tasks
    modelOptions.value = models
    datasetOptions.value = datasets
    gpuOptions.value = gpus
    const initial = tasks.find((item) => item.hasResult) ?? tasks[0]
    if (initial) {
      selectedTaskId.value = initial.id
      await loadEvaluationDetail(initial.id)
    }
    if (!useMocks) refreshTimer = window.setInterval(() => { void refreshEvaluations().catch(() => undefined) }, 3_000)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '测评数据加载失败')
  }
})

onBeforeUnmount(() => {
  detailRequestVersion += 1
  if (refreshTimer !== undefined) window.clearInterval(refreshTimer)
})
</script>

<template>
  <div>
    <PageHeader title="模型测评" subtitle="量化比较训练前后通用能力保持与目标领域提升">
      <el-button :icon="Download" :disabled="!selectedDetail?.hasResult" @click="exportReport">导出当前报告</el-button>
      <el-button type="primary" :icon="Plus" @click="createVisible=true">创建测评</el-button>
    </PageHeader>

    <el-tabs v-model="activeTab" class="evaluation-tabs">
      <el-tab-pane label="测评任务" name="tasks">
        <PanelCard flush>
          <el-table
            :data="taskRows"
            row-key="id"
            highlight-current-row
            :current-row-key="selectedTaskId"
            empty-text="暂无测评任务"
            @row-click="(row: EvaluationRunSummary) => selectEvaluation(row.id)"
          >
            <el-table-column type="expand">
              <template #default="{ row }">
                <div class="task-detail">
                  <span><b>模型：</b>{{ row.baseModel }} → {{ row.candidateModel }}</span>
                  <span><b>GPU：</b>{{ row.gpuIds.map((id: number) => `GPU ${id}`).join('、') }}</span>
                  <span><b>开始：</b>{{ row.startedAt ?? '—' }}</span>
                  <span><b>结束：</b>{{ row.finishedAt ?? '—' }}</span>
                  <span v-if="row.errorMessage" class="task-error"><b>错误：</b>{{ row.errorMessage }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="name" label="任务名称" min-width="180" />
            <el-table-column prop="baseModel" label="基线模型" min-width="170" show-overflow-tooltip />
            <el-table-column prop="candidateModel" label="候选模型" min-width="170" show-overflow-tooltip />
            <el-table-column prop="datasets" label="测评数据集" min-width="240" show-overflow-tooltip />
            <el-table-column label="进度" width="150"><template #default="{ row }"><el-progress :percentage="row.progress" :stroke-width="6" /></template></el-table-column>
            <el-table-column label="状态" width="110"><template #default="{ row }"><StatusPill v-bind="taskStatusMeta(row.status)" /></template></el-table-column>
            <el-table-column prop="updatedAt" label="更新时间" width="165" />
            <el-table-column label="操作" width="155" fixed="right">
              <template #default="{ row }">
                <div class="table-actions">
                  <span class="action-link" @click.stop="selectEvaluation(row.id, true)">查看</span>
                  <span v-if="['queued','running','stopping'].includes(row.status)" class="danger-link" @click.stop="cancelEvaluation(row)">取消</span>
                  <span v-else class="danger-link" @click.stop="removeEvaluation(row)">删除</span>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </PanelCard>
      </el-tab-pane>

      <el-tab-pane label="训练前后对比" name="comparison">
        <PanelCard title="选择测评任务" class="evaluation-selector">
          <template #actions><StatusPill v-if="selectedDetail" v-bind="taskStatusMeta(selectedDetail.status)" /></template>
          <el-select v-model="selectedTaskId" filterable placeholder="选择要查看的测评任务" style="width:min(520px, 100%)" @change="(id: string) => selectEvaluation(id)">
            <el-option v-for="task in taskRows" :key="task.id" :value="task.id" :label="`${task.name} · ${taskStatusMeta(task.status).text}`" />
          </el-select>
          <span v-if="selectedDetail" class="selector-meta">更新于 {{ selectedDetail.updatedAt }}</span>
        </PanelCard>

        <div v-loading="detailLoading" class="evaluation-detail">
          <template v-if="selectedDetail">
            <el-alert v-if="selectedDetail.errorMessage" class="section-gap" type="error" :title="selectedDetail.errorMessage" :closable="false" show-icon />
            <el-alert v-for="warning in selectedDetail.warnings" :key="warning" class="section-gap" type="warning" :title="warningText(warning)" :closable="false" show-icon />

            <template v-if="selectedDetail.hasResult && overall">
              <PanelCard class="model-comparison section-gap">
                <div class="model-side"><div class="model-icon blue"><el-icon><DataAnalysis /></el-icon></div><span>基线模型 · {{ templateText(selectedDetail.baseTemplate) }}<strong>{{ selectedDetail.baseModel }}</strong></span></div>
                <span class="versus">VS</span>
                <div class="model-side"><div class="model-icon green"><el-icon><DataAnalysis /></el-icon></div><span>候选模型 · {{ templateText(selectedDetail.candidateTemplate) }}<strong>{{ selectedDetail.candidateModel }}</strong></span></div>
                <el-divider direction="vertical" />
                <div class="dataset-tags"><span>对比数据集</span><div><el-tag v-for="dataset in selectedDetail.datasetNames" :key="dataset" effect="plain">{{ dataset }}</el-tag></div></div>
              </PanelCard>

              <div class="score-grid section-gap">
                <div class="score-card"><span>基线模型得分</span><strong><i>{{ formatPercent(overall.before) }}</i></strong></div>
                <div class="score-card"><span>候选模型得分</span><strong><em>{{ formatPercent(overall.after) }}</em></strong></div>
                <div class="score-card"><span>绝对变化</span><strong :class="overall.pointChange >= 0 ? 'number-positive' : 'danger-link'">{{ signed(overall.pointChange) }} <small>个百分点</small></strong></div>
                <div class="score-card"><span>相对变化</span><strong :class="relativeTone(overall.relativeChange)">{{ formatRelative(overall.relativeChange) }} <small v-if="overall.relativeChange === null">基线为 0，未定义</small></strong></div>
              </div>

              <div class="evaluation-charts section-gap">
                <PanelCard title="各数据集得分对比"><BaseChart :option="comparisonOption" height="280px" /></PanelCard>
                <PanelCard title="科目百分点变化（绝对值前 12）"><BaseChart v-if="categoryRows.length" :option="categoryOption" height="280px" /><el-empty v-else description="当前报告没有科目明细" :image-size="54" /></PanelCard>
              </div>

              <PanelCard title="模型运行统计" class="section-gap" flush>
                <el-table :data="modelMetricRows">
                  <el-table-column prop="role" label="角色" width="110" />
                  <el-table-column prop="model" label="模型" min-width="210" show-overflow-tooltip />
                  <el-table-column label="模板" width="100"><template #default="{ row }">{{ row.metric ? templateText(row.metric.template) : '—' }}</template></el-table-column>
                  <el-table-column label="得分" width="120"><template #default="{ row }"><b v-if="row.metric" class="number-primary">{{ formatPercent(row.metric.score) }}</b><span v-else>—</span></template></el-table-column>
                  <el-table-column label="正确 / 总数" width="150"><template #default="{ row }">{{ row.metric ? `${row.metric.correct.toLocaleString()} / ${row.metric.total.toLocaleString()}` : '—' }}</template></el-table-column>
                  <el-table-column label="无效输出" width="120"><template #default="{ row }">{{ row.metric?.invalid.toLocaleString() ?? '—' }}</template></el-table-column>
                  <el-table-column label="平均延迟" width="140"><template #default="{ row }">{{ row.metric ? `${formatNumber(row.metric.averageLatencyMs)} ms` : '—' }}</template></el-table-column>
                </el-table>
              </PanelCard>

              <PanelCard title="数据集汇总" class="section-gap" flush>
                <el-table :data="rows">
                  <el-table-column prop="dataset" label="数据集" min-width="190" />
                  <el-table-column prop="samples" label="样本数" width="110"><template #default="{ row }">{{ row.samples.toLocaleString() }}</template></el-table-column>
                  <el-table-column label="基线得分" width="125"><template #default="{ row }"><b class="number-primary">{{ formatPercent(row.before) }}</b></template></el-table-column>
                  <el-table-column label="基线正确 / 总数" width="155"><template #default="{ row }">{{ formatCount(row.beforeCorrect, row.beforeTotal) }}</template></el-table-column>
                  <el-table-column label="基线无效" width="105"><template #default="{ row }">{{ row.beforeInvalid?.toLocaleString() ?? '—' }}</template></el-table-column>
                  <el-table-column label="候选得分" width="125"><template #default="{ row }"><b class="number-positive">{{ formatPercent(row.after) }}</b></template></el-table-column>
                  <el-table-column label="候选正确 / 总数" width="155"><template #default="{ row }">{{ formatCount(row.afterCorrect, row.afterTotal) }}</template></el-table-column>
                  <el-table-column label="候选无效" width="105"><template #default="{ row }">{{ row.afterInvalid?.toLocaleString() ?? '—' }}</template></el-table-column>
                  <el-table-column label="绝对变化" width="125"><template #default="{ row }"><b :class="row.pointChange >= 0 ? 'number-positive' : 'danger-link'">{{ signed(row.pointChange) }}</b></template></el-table-column>
                  <el-table-column label="相对变化" width="135"><template #default="{ row }"><b :class="relativeTone(row.relativeChange)">{{ formatRelative(row.relativeChange) }}</b><el-tooltip v-if="row.relativeChange === null" content="基线得分为 0，相对变化没有数学定义"><span class="undefined-hint">?</span></el-tooltip></template></el-table-column>
                </el-table>
                <div class="method-note">ⓘ 所有对比使用相同数据集、提示模板与推理参数；绝对变化单位为百分点，不设置通过/失败阈值。</div>
              </PanelCard>

              <PanelCard v-if="categoryRows.length" title="数据集 / 科目明细" class="section-gap" flush>
                <div class="category-toolbar"><el-input v-model="categoryQuery" clearable placeholder="筛选数据集或科目" style="width:280px" /><span>共 {{ filteredCategoryRows.length.toLocaleString() }} 个科目</span></div>
                <el-table :data="pagedCategoryRows">
                  <el-table-column prop="dataset" label="数据集" min-width="170" show-overflow-tooltip />
                  <el-table-column prop="category" label="科目 / 分类" min-width="180" show-overflow-tooltip />
                  <el-table-column prop="samples" label="样本数" width="100" />
                  <el-table-column label="基线（正确 / 总数）" width="185"><template #default="{ row }"><b class="number-primary">{{ formatPercent(row.before) }}</b> · {{ formatCount(row.beforeCorrect, row.beforeTotal) }}</template></el-table-column>
                  <el-table-column label="候选（正确 / 总数）" width="185"><template #default="{ row }"><b class="number-positive">{{ formatPercent(row.after) }}</b> · {{ formatCount(row.afterCorrect, row.afterTotal) }}</template></el-table-column>
                  <el-table-column label="无效输出（前 / 后）" width="160"><template #default="{ row }">{{ row.beforeInvalid ?? '—' }} / {{ row.afterInvalid ?? '—' }}</template></el-table-column>
                  <el-table-column label="绝对变化" width="120"><template #default="{ row }"><b :class="row.pointChange >= 0 ? 'number-positive' : 'danger-link'">{{ signed(row.pointChange) }}</b></template></el-table-column>
                  <el-table-column label="相对变化" width="130"><template #default="{ row }"><b :class="relativeTone(row.relativeChange)">{{ formatRelative(row.relativeChange) }}</b></template></el-table-column>
                </el-table>
                <el-pagination v-if="filteredCategoryRows.length > categoryPageSize" v-model:current-page="categoryPage" class="category-pagination" background layout="prev, pager, next" :page-size="categoryPageSize" :total="filteredCategoryRows.length" />
              </PanelCard>

              <PanelCard title="执行配置" class="section-gap">
                <div class="execution-config">
                  <span><b>GPU：</b>{{ selectedDetail.gpuIds.map((id) => `GPU ${id}`).join('、') }}</span>
                  <span><b>Tensor Parallel：</b>{{ selectedDetail.tensorParallelSize }}</span>
                  <span><b>并发：</b>{{ selectedDetail.concurrency }}</span>
                  <span><b>最大生成 Token：</b>{{ selectedDetail.maxTokens }}</span>
                  <span><b>显存利用率：</b>{{ formatPercent(selectedDetail.gpuMemoryUtilization * 100) }}</span>
                  <span><b>开始：</b>{{ selectedDetail.startedAt ?? '—' }}</span>
                  <span><b>结束：</b>{{ selectedDetail.finishedAt ?? '—' }}</span>
                </div>
              </PanelCard>
            </template>

            <PanelCard v-else class="section-gap">
              <el-empty :description="selectedDetail.status === 'failed' ? '测评失败，尚无可比较结果' : selectedDetail.status === 'terminated' ? '测评已取消，尚无可比较结果' : '任务尚未完成，结果生成后会自动刷新'" />
            </PanelCard>
          </template>
          <PanelCard v-else class="section-gap"><el-empty description="暂无测评任务，请先创建测评" /></PanelCard>
        </div>
      </el-tab-pane>

      <el-tab-pane label="测评数据集" name="datasets">
        <PanelCard><el-empty description="内置 C-Eval、CMMLU；自定义 JSONL 评测集请在训练数据集模块上传和预览" /></PanelCard>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="createVisible" title="创建模型测评" width="min(680px, 94vw)">
      <el-form label-position="top">
        <div class="two-column-form"><el-form-item label="任务名称" required><el-input v-model="form.name" placeholder="例如 domain-regression-v1" /></el-form-item><el-form-item label="模板策略"><el-input model-value="按每个模型资产自动选择 Base / Instruct" disabled /></el-form-item></div>
        <div class="two-column-form"><el-form-item label="基线模型" required><el-select v-model="form.baseModelAssetId" filterable style="width:100%" placeholder="选择训练前模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id" /></el-select></el-form-item><el-form-item label="候选模型" required><el-select v-model="form.candidateModelAssetId" filterable style="width:100%" placeholder="选择训练后模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id" /></el-select></el-form-item></div>
        <el-form-item label="测评数据集" required><el-checkbox-group v-model="form.datasets"><el-checkbox value="ceval">C-Eval（中文通用）</el-checkbox><el-checkbox value="cmmlu">CMMLU（中文通用）</el-checkbox><el-checkbox value="custom">自定义领域集</el-checkbox></el-checkbox-group></el-form-item>
        <el-form-item v-if="form.datasets.includes('custom')" label="自定义领域集"><el-select v-model="form.customDatasetId" style="width:100%"><el-option v-for="dataset in datasetOptions.filter((item) => item.status === 'available' && item.purpose === 'Evaluation')" :key="dataset.id" :label="`${dataset.name} ${dataset.version}`" :value="dataset.id" /></el-select></el-form-item>
        <el-form-item label="整卡选择" required><el-select v-model="form.gpuIds" multiple collapse-tags :max-collapse-tags="3" style="width:100%"><el-option v-for="gpu in gpuOptions" :key="gpu.index" :label="`GPU ${gpu.index} · ${gpuStateText(gpu)}`" :value="gpu.index" /></el-select><p class="form-help">基线与候选模型会在同一组 GPU 上顺序加载，避免同时占用双份显存。</p></el-form-item>
        <div class="two-column-form"><el-form-item label="生成参数"><el-input model-value="由控制面固定，基线与候选完全一致" disabled /></el-form-item><el-form-item label="计分方式"><el-input model-value="准确率（百分比）" disabled /></el-form-item></div>
        <el-alert title="Base 模型适配" description="系统根据模型类型应用对应模板；Base 模型使用续写/选项概率，Instruct 模型使用对话指令模板。" type="info" :closable="false" show-icon />
      </el-form>
      <template #footer><el-button @click="createVisible=false">取消</el-button><el-button type="primary" :loading="createBusy" @click="createEvaluation">创建测评</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.evaluation-tabs { margin-top: -14px; }
.evaluation-selector :deep(.panel-body) { display: flex; align-items: center; gap: 14px; }
.selector-meta { color: #788598; font-size: 12px; }
.evaluation-detail { min-height: 180px; }
.model-comparison :deep(.panel-body) { display: flex; align-items: center; gap: 28px; padding: 16px 20px; }
.model-side { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
.model-side > span { display: flex; flex-direction: column; min-width: 0; color: #6c788b; font-size: 12px; }
.model-side strong { margin-top: 5px; overflow: hidden; color: #121c2c; font-size: 18px; text-overflow: ellipsis; white-space: nowrap; }
.model-icon { width: 48px; height: 48px; display: grid; flex: 0 0 auto; place-items: center; border-radius: 9px; font-size: 23px; }
.model-icon.blue { color: #1769f5; background: #eaf2ff; }
.model-icon.green { color: #12a865; background: #e7f7ef; }
.versus { display: grid; width: 40px; height: 40px; flex: 0 0 auto; place-items: center; border-radius: 50%; color: #6f7b8f; background: #f1f4f8; font-weight: 700; }
.dataset-tags { display: flex; min-width: 220px; flex-direction: column; gap: 8px; }
.dataset-tags > span { font-size: 12px; }
.dataset-tags > div { display: flex; flex-wrap: wrap; gap: 8px; }
.score-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; }
.score-card { padding: 18px; border: 1px solid #dfe6ef; border-radius: 8px; background: #fff; }
.score-card > span { font-size: 13px; }
.score-card > strong { display: flex; min-height: 34px; align-items: baseline; gap: 8px; margin-top: 12px; font-size: 25px; }
.score-card i { color: #1769f5; font-style: normal; }
.score-card em { color: #12a865; font-style: normal; }
.score-card small { color: #7a8799; font-size: 12px; font-weight: 500; }
.evaluation-charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 13px; }
.task-detail { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px 22px; padding: 4px 52px; color: #667386; font-size: 13px; }
.task-error { grid-column: 1 / -1; color: #d94a5b; }
.category-toolbar { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; color: #778397; font-size: 12px; }
.category-pagination { justify-content: flex-end; padding: 15px 18px; }
.execution-config { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px 22px; color: #667386; font-size: 13px; }
.method-note { padding: 12px 18px; color: #738095; font-size: 12px; }
.undefined-hint { display: inline-grid; width: 16px; height: 16px; margin-left: 5px; place-items: center; border-radius: 50%; color: #748196; background: #eef2f6; font-size: 11px; cursor: help; }
.two-column-form { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.form-help { margin: 6px 0 0; color: #798598; font-size: 12px; }

@media (max-width: 1200px) {
  .score-grid { grid-template-columns: repeat(2, 1fr); }
  .task-detail, .execution-config { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 760px) {
  .evaluation-selector :deep(.panel-body), .model-comparison :deep(.panel-body) { align-items: flex-start; flex-direction: column; gap: 14px; }
  .versus, .model-comparison .el-divider { display: none; }
  .score-grid, .evaluation-charts, .two-column-form, .task-detail, .execution-config { grid-template-columns: 1fr; }
  .model-side strong { font-size: 16px; }
  .task-detail { padding: 4px 18px; }
  .category-toolbar { align-items: flex-start; flex-direction: column; gap: 10px; }
}
</style>
