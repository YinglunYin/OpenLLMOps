<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { DataAnalysis, Download, Plus } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { Dataset, EvaluationRunSummary, EvaluationSummary, ModelAsset, StatusTone } from '@/types/domain'

const activeTab = ref('comparison')
const rows = ref<EvaluationSummary[]>([])
const taskRows = ref<EvaluationRunSummary[]>([])
const modelOptions = ref<ModelAsset[]>([])
const datasetOptions = ref<Dataset[]>([])
const createVisible = ref(false)
const createBusy = ref(false)
const form = reactive({ name: '', modelKind: 'instruct', baseModelAssetId: '', candidateModelAssetId: '', datasets: ['ceval', 'cmmlu'], customDatasetId: '', temperature: 0, maxTokens: 512 })

const generalRows = computed(() => rows.value.slice(0, 2))
const generalBefore = computed(() => generalRows.value.length ? generalRows.value.reduce((sum, item) => sum + item.before, 0) / generalRows.value.length : 0)
const generalAfter = computed(() => generalRows.value.length ? generalRows.value.reduce((sum, item) => sum + item.after, 0) / generalRows.value.length : 0)
const domain = computed<EvaluationSummary>(() => rows.value.at(-1) ?? { dataset: '领域测试集', samples: 0, before: 0, after: 0, pointChange: 0, relativeChange: 0 })
const taskStatusMap: Record<EvaluationRunSummary['status'], { text: string; tone: StatusTone }> = {
  queued: { text: '等待 GPU', tone: 'warning' }, running: { text: '测评中', tone: 'primary' }, completed: { text: '已完成', tone: 'success' }, failed: { text: '失败', tone: 'danger' }, stopping: { text: '取消中', tone: 'info' }, terminated: { text: '已取消', tone: 'info' },
}
const taskStatusMeta = (status: EvaluationRunSummary['status']) => taskStatusMap[status]

const comparisonOption = computed(() => ({
  color: ['#1769f5', '#12a865'], tooltip: { trigger: 'axis' }, legend: { top: 0, right: 10, itemWidth: 10, itemHeight: 10 }, grid: { top: 42, left: 42, right: 10, bottom: 32 },
  xAxis: { type: 'category', data: rows.value.map((item) => item.dataset), axisLabel: { color: '#69768a' } }, yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%', color: '#69768a' }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } },
  series: [{ name: '训练前', type: 'bar', data: rows.value.map((item) => item.before), barMaxWidth: 32 }, { name: '训练后', type: 'bar', data: rows.value.map((item) => item.after), barMaxWidth: 32 }],
}))

const domainOption = { color: ['#12a865'], grid: { top: 12, left: 72, right: 36, bottom: 20 }, xAxis: { type: 'value', min: -20, max: 40, axisLabel: { color: '#718095' }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } }, yAxis: { type: 'category', data: ['知识问答','信息抽取','阅读理解','文本生成','数学推理','代码理解','对话能力'], axisLabel: { color: '#566276', fontSize: 11 } }, series: [{ type: 'bar', data: [14.8,18.9,16.3,12.6,8.7,11.5,15.2], barWidth: 11, label: { show: true, position: 'right', formatter: '+{c}', color: '#05965b' }, itemStyle: { borderRadius: [0,3,3,0] } }] }

async function createEvaluation() {
  if (!form.name || !form.baseModelAssetId || !form.candidateModelAssetId || !form.datasets.length) { ElMessage.warning('请填写任务名称并选择基线、候选模型和数据集'); return }
  createBusy.value = true
  try {
    await api.evaluations.create({ ...form })
    createVisible.value = false
    ;[rows.value, taskRows.value] = await Promise.all([api.evaluations.comparison(), api.evaluations.list()])
    ElMessage.success('测评任务已创建并进入 GPU 队列')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '测评任务创建失败')
  } finally {
    createBusy.value = false
  }
}

onMounted(async () => {
  try {
    ;[rows.value, taskRows.value, modelOptions.value, datasetOptions.value] = await Promise.all([api.evaluations.comparison(), api.evaluations.list(), api.models.list(), api.datasets.list()])
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '测评数据加载失败')
  }
})
</script>

<template>
  <div>
    <PageHeader title="模型测评" subtitle="量化比较训练前后通用能力保持与目标领域提升">
      <el-button :icon="Download">导出报告</el-button>
      <el-button type="primary" :icon="Plus" @click="createVisible=true">创建测评</el-button>
    </PageHeader>

    <el-tabs v-model="activeTab" class="evaluation-tabs">
      <el-tab-pane label="测评任务" name="tasks">
        <PanelCard flush><el-table :data="taskRows" empty-text="暂无测评任务"><el-table-column prop="name" label="任务名称" min-width="180"/><el-table-column prop="model" label="候选模型" min-width="180"/><el-table-column prop="datasets" label="测评数据集" min-width="250"/><el-table-column label="进度" min-width="180"><template #default="{row}"><el-progress :percentage="row.progress" :stroke-width="6"/></template></el-table-column><el-table-column label="状态" width="110"><template #default="{row}"><StatusPill v-bind="taskStatusMeta(row.status)"/></template></el-table-column><el-table-column prop="updatedAt" label="更新时间" width="165"/><el-table-column label="操作" width="100"><template #default><span class="table-link">查看报告</span></template></el-table-column></el-table></PanelCard>
      </el-tab-pane>
      <el-tab-pane label="训练前后对比" name="comparison">
        <template v-if="rows.length">
        <PanelCard class="model-comparison">
          <div class="model-side"><div class="model-icon blue"><el-icon><DataAnalysis/></el-icon></div><span>基线模型<strong>{{ useMocks ? 'ChineseLM-8B-Base' : '基线模型' }}</strong></span></div><span class="versus">VS</span><div class="model-side"><div class="model-icon green"><el-icon><DataAnalysis/></el-icon></div><span>训练后模型<strong>{{ useMocks ? 'ChineseLM-8B-Domain' : '候选模型' }}</strong></span></div><el-divider direction="vertical"/><div class="dataset-tags"><span>对比数据集</span><div><el-tag v-for="row in rows" :key="row.dataset" effect="plain">{{ row.dataset }}</el-tag></div></div>
        </PanelCard>

        <div class="score-grid section-gap">
          <div class="score-card"><span>通用能力</span><strong><i>{{ generalBefore.toFixed(1) }}%</i><b>→</b><em>{{ generalAfter.toFixed(1) }}%</em></strong></div>
          <div class="score-card"><span>领域能力</span><strong><i>{{ domain.before }}%</i><b>→</b><em>{{ domain.after }}%</em></strong></div>
          <div class="score-card"><span>通用变化</span><strong class="change">+{{ (generalAfter-generalBefore).toFixed(1) }} <small>个百分点</small></strong></div>
          <div class="score-card"><span>领域提升</span><strong class="change">+{{ domain.pointChange }} <small>个百分点</small></strong></div>
        </div>

        <div class="evaluation-charts section-gap">
          <PanelCard title="训练前后得分对比"><BaseChart :option="comparisonOption" height="230px" /></PanelCard>
          <PanelCard title="分领域变化"><BaseChart v-if="useMocks" :option="domainOption" height="230px" /><el-empty v-else description="当前 comparison 未包含分领域明细" :image-size="54" /></PanelCard>
          <PanelCard title="样本级正确性变化（领域测试集）">
            <template v-if="useMocks">
            <table class="matrix"><thead><tr><th /><th>训练后正确</th><th>训练后错误</th><th>合计</th></tr></thead><tbody><tr><th>原来正确</th><td class="good">5,382</td><td>1,218</td><td>6,600</td></tr><tr><th>原来错误</th><td>2,842</td><td class="bad">2,358</td><td>5,200</td></tr><tr><th>合计</th><td>8,224</td><td>3,576</td><td>11,800</td></tr></tbody></table>
            </template><el-empty v-else description="当前 comparison 未包含混淆矩阵" :image-size="54" />
          </PanelCard>
        </div>

        <PanelCard title="测评结果汇总" class="section-gap" flush>
          <el-table :data="rows"><el-table-column prop="dataset" label="数据集" min-width="180"/><el-table-column prop="samples" label="样本数" width="130"><template #default="{row}">{{ row.samples.toLocaleString() }}</template></el-table-column><el-table-column prop="before" label="训练前" width="140"><template #default="{row}"><b class="number-primary">{{ row.before }}%</b></template></el-table-column><el-table-column prop="after" label="训练后" width="140"><template #default="{row}"><b class="number-positive">{{ row.after }}%</b></template></el-table-column><el-table-column prop="pointChange" label="百分点变化" width="150"><template #default="{row}"><b class="number-positive">+{{ row.pointChange }}</b></template></el-table-column><el-table-column prop="relativeChange" label="相对变化" width="150"><template #default="{row}"><b class="number-positive">+{{ row.relativeChange }}%</b></template></el-table-column></el-table>
          <div class="method-note">ⓘ 所有对比使用相同数据集、提示模板与推理参数；不设置通过/失败阈值。</div>
        </PanelCard>
        </template>
        <PanelCard v-else><el-empty description="暂无已完成且包含 comparison 的测评结果" /></PanelCard>
      </el-tab-pane>
      <el-tab-pane label="测评数据集" name="datasets"><el-empty description="内置 C-Eval、CMMLU；自定义数据集请在训练数据集模块上传" /></el-tab-pane>
    </el-tabs>

    <el-dialog v-model="createVisible" title="创建模型测评" width="min(680px, 94vw)">
      <el-form label-position="top">
        <div class="two-column-form"><el-form-item label="任务名称" required><el-input v-model="form.name" placeholder="例如 domain-regression-v1"/></el-form-item><el-form-item label="模型类型"><el-radio-group v-model="form.modelKind"><el-radio-button value="base">Base</el-radio-button><el-radio-button value="instruct">Instruct</el-radio-button></el-radio-group></el-form-item></div>
        <div class="two-column-form"><el-form-item label="基线模型" required><el-select v-model="form.baseModelAssetId" filterable style="width:100%" placeholder="选择训练前模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id"/></el-select></el-form-item><el-form-item label="候选模型" required><el-select v-model="form.candidateModelAssetId" filterable style="width:100%" placeholder="选择训练后模型"><el-option v-for="model in modelOptions.filter((item) => item.status === 'available' && item.type === 'generation')" :key="model.id" :label="model.name" :value="model.id"/></el-select></el-form-item></div>
        <el-form-item label="测评数据集" required><el-checkbox-group v-model="form.datasets"><el-checkbox value="ceval">C-Eval（中文通用）</el-checkbox><el-checkbox value="cmmlu">CMMLU（中文通用）</el-checkbox><el-checkbox value="custom">自定义领域集</el-checkbox></el-checkbox-group></el-form-item>
        <el-form-item v-if="form.datasets.includes('custom')" label="自定义领域集"><el-select v-model="form.customDatasetId" style="width:100%"><el-option v-for="dataset in datasetOptions.filter((item) => item.status === 'available' && item.purpose === 'Evaluation')" :key="dataset.id" :label="`${dataset.name} ${dataset.version}`" :value="dataset.id"/></el-select></el-form-item>
        <div class="three-column-form"><el-form-item label="Temperature"><el-input-number v-model="form.temperature" :min="0" :max="2" :step="0.1"/></el-form-item><el-form-item label="最大 Token"><el-input-number v-model="form.maxTokens" :min="32" :step="32"/></el-form-item><el-form-item label="计分方式"><el-input model-value="准确率（百分比）" disabled/></el-form-item></div>
        <el-alert title="Base 模型适配" description="系统根据模型类型应用对应模板；Base 模型使用续写/选项概率，Instruct 模型使用对话指令模板。" type="info" :closable="false" show-icon/>
      </el-form>
      <template #footer><el-button @click="createVisible=false">取消</el-button><el-button type="primary" :loading="createBusy" @click="createEvaluation">创建测评</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.evaluation-tabs{margin-top:-14px}.model-comparison :deep(.panel-body){display:flex;align-items:center;gap:28px;padding:16px 20px}.model-side{display:flex;align-items:center;gap:12px;flex:1}.model-side>span{display:flex;flex-direction:column;color:#6c788b;font-size:12px}.model-side strong{margin-top:5px;color:#121c2c;font-size:20px}.model-icon{width:48px;height:48px;display:grid;place-items:center;border-radius:9px;font-size:23px}.model-icon.blue{color:#1769f5;background:#eaf2ff}.model-icon.green{color:#12a865;background:#e7f7ef}.versus{display:grid;width:40px;height:40px;place-items:center;border-radius:50%;color:#6f7b8f;background:#f1f4f8;font-weight:700}.dataset-tags{display:flex;flex-direction:column;gap:8px}.dataset-tags>span{font-size:12px}.dataset-tags>div{display:flex;gap:8px}.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.score-card{padding:18px;border:1px solid #dfe6ef;border-radius:8px;background:#fff}.score-card>span{font-size:13px}.score-card>strong{display:flex;align-items:center;gap:11px;margin-top:12px;font-size:25px}.score-card i{color:#1769f5;font-style:normal}.score-card b{color:#8590a0}.score-card em,.score-card .change{color:#12a865;font-style:normal}.score-card small{font-size:12px;font-weight:500}.evaluation-charts{display:grid;grid-template-columns:1fr 1.1fr 1fr;gap:13px}.matrix{width:100%;border-collapse:separate;border-spacing:0;margin-top:18px;font-size:12px}.matrix th,.matrix td{padding:14px 8px;text-align:center;border-right:1px solid #e0e6ee;border-bottom:1px solid #e0e6ee}.matrix tr:first-child th{border-top:1px solid #e0e6ee}.matrix th:first-child{border-left:1px solid #e0e6ee}.matrix .good{background:#eaf8f1}.matrix .bad{background:#fff0f1}.method-note{padding:12px 18px;color:#738095;font-size:12px}.two-column-form{display:grid;grid-template-columns:1fr 1fr;gap:14px}.three-column-form{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:1200px){.evaluation-charts{grid-template-columns:1fr 1fr}.evaluation-charts>*:last-child{grid-column:1/-1}.score-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:720px){.model-comparison :deep(.panel-body){align-items:flex-start;flex-direction:column;gap:14px}.versus,.model-comparison .el-divider{display:none}.score-grid,.evaluation-charts,.two-column-form,.three-column-form{grid-template-columns:1fr}.evaluation-charts>*:last-child{grid-column:auto}.model-side strong{font-size:16px}}
</style>
