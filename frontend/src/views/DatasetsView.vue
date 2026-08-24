<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, shallowRef } from 'vue'
import { Box, DataAnalysis, Files, Refresh, Search, Upload, Warning } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { UploadFile } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { Dataset, StatusTone } from '@/types/domain'

const rows = ref<Dataset[]>([])
const selected = ref<Dataset | null>(null)
const uploadFile = shallowRef<File>()
const previewRows = ref<Array<Record<string, unknown>>>(useMocks ? [{ instruction: '概括以下金融文本', input: '……', output: '……' }] : [])
const detailTab = ref('overview')
const uploadVisible = ref(false)
const uploadBusy = ref(false)
const uploadProgress = ref(0)
let uploadController: AbortController | undefined
const filters = reactive({ keyword: '', purpose: '', status: '' })
const uploadForm = reactive({ name: '', purpose: 'SFT', version: 'v1.0.0' })

const filtered = computed(() => rows.value.filter((item) => (!filters.keyword || item.name.includes(filters.keyword)) && (!filters.purpose || item.purpose === filters.purpose) && (!filters.status || item.status === filters.status)))
const versionRows = computed(() => selected.value
  ? rows.value.filter((item) => item.name === selected.value?.name).sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
  : [])
const counts = computed(() => ({
  cpt: rows.value.filter((item) => item.purpose === 'CPT').length,
  sft: rows.value.filter((item) => item.purpose === 'SFT').length,
  evaluation: rows.value.filter((item) => item.purpose === 'Evaluation').length,
  failed: rows.value.filter((item) => item.status === 'failed').length,
}))
const statusMap: Record<Dataset['status'], { text: string; tone: StatusTone }> = { available: { text: '可用', tone: 'success' }, validating: { text: '校验中', tone: 'warning' }, failed: { text: '校验失败', tone: 'danger' } }
const statusMeta = (status: Dataset['status']) => statusMap[status]

const histogramOption = {
  color: ['#1769f5'], tooltip: { trigger: 'axis' }, grid: { top: 15, left: 38, right: 10, bottom: 34 },
  xAxis: { type: 'category', data: ['0', '128', '256', '384', '512', '768', '1k', '2k', '4k', '8k+'], axisLabel: { color: '#758195', fontSize: 10 } },
  yAxis: { type: 'value', axisLabel: { color: '#758195', fontSize: 10 }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } },
  series: [{ type: 'bar', data: [420, 980, 1770, 2360, 2840, 2010, 1570, 880, 320, 73], barMaxWidth: 28, itemStyle: { borderRadius: [3, 3, 0, 0] } }],
}

function reset() { Object.assign(filters, { keyword: '', purpose: '', status: '' }) }

async function refreshRows() {
  rows.value = await api.datasets.list()
  selected.value = selected.value ? rows.value.find((row) => row.id === selected.value?.id) ?? rows.value[0] ?? null : rows.value[0] ?? null
}

function handleFileChange(file: UploadFile) {
  uploadFile.value = file.raw
}

async function selectDataset(row: Dataset) {
  selected.value = row
  if (!useMocks) {
    try { previewRows.value = await api.datasets.preview(row.id) }
    catch { previewRows.value = [] }
  }
}

async function showPreview(row: Dataset) {
  await selectDataset(row)
  detailTab.value = 'preview'
}

async function removeDataset(row: Dataset) {
  try {
    await ElMessageBox.confirm('存在训练或测评任务引用时，控制面会拒绝删除。', `删除 ${row.name}`, { type: 'warning' })
    await api.datasets.remove(row.id)
    await refreshRows()
    previewRows.value = []
    ElMessage.success('数据集已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '删除数据集失败')
  }
}

async function submitUpload() {
  if (!uploadForm.name || !uploadFile.value) { ElMessage.warning('请填写数据集名称并选择 JSONL 文件'); return }
  uploadBusy.value = true
  uploadProgress.value = 0
  uploadController = new AbortController()
  try {
    await api.datasets.upload(
      { name: uploadForm.name, purpose: uploadForm.purpose as Dataset['purpose'], version: uploadForm.version },
      uploadFile.value,
      { signal: uploadController.signal, onProgress: (percent) => { uploadProgress.value = percent } },
    )
    uploadVisible.value = false
    uploadFile.value = undefined
    await refreshRows()
    if (selected.value) await selectDataset(selected.value)
    ElMessage.success('上传与逐行校验已完成')
  } catch (error) {
    if (uploadController.signal.aborted) ElMessage.info('已中断浏览器传输或等待；若服务端已开始校验，结果仍可能稍后出现在列表中')
    else ElMessage.error(error instanceof Error ? error.message : '数据集上传失败')
  } finally {
    uploadBusy.value = false
    uploadController = undefined
  }
}

function cancelUpload() {
  if (uploadBusy.value) uploadController?.abort()
  uploadVisible.value = false
}

function beforeCloseUpload(done: () => void) {
  uploadController?.abort()
  done()
}

onBeforeUnmount(() => uploadController?.abort())

onMounted(async () => {
  try {
    rows.value = await api.datasets.list()
    selected.value = rows.value[0] ?? null
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '数据集加载失败')
  }
})
</script>

<template>
  <div>
    <PageHeader title="训练数据集" subtitle="管理 CPT、SFT 与自定义测评 JSONL 数据及不可变版本">
      <el-button :icon="Upload" @click="uploadVisible = true">上传数据集</el-button>
      <el-button type="primary" :icon="Refresh" @click="refreshRows">刷新列表</el-button>
    </PageHeader>

    <div class="stats-grid">
      <StatCard label="CPT" :value="counts.cpt" :icon="Box" tone="blue" />
      <StatCard label="SFT" :value="counts.sft" :icon="Files" tone="green" />
      <StatCard label="测评集" :value="counts.evaluation" :icon="DataAnalysis" tone="orange" />
      <StatCard label="校验失败" :value="counts.failed" :icon="Warning" tone="purple" />
    </div>

    <PanelCard class="section-gap" flush>
      <div class="filters">
        <el-input v-model="filters.keyword" placeholder="搜索数据集名称" :prefix-icon="Search" clearable />
        <el-select v-model="filters.purpose" placeholder="用途：全部" clearable><el-option label="CPT" value="CPT" /><el-option label="SFT" value="SFT" /><el-option label="测评" value="Evaluation" /></el-select>
        <el-select model-value="JSONL" disabled placeholder="格式" />
        <el-select v-model="filters.status" placeholder="状态：全部" clearable><el-option label="可用" value="available" /><el-option label="校验中" value="validating" /><el-option label="校验失败" value="failed" /></el-select>
        <el-button :icon="Refresh" @click="reset">重置</el-button>
      </div>
      <el-table :data="filtered" highlight-current-row row-key="id" @current-change="selectDataset">
        <el-table-column prop="name" label="数据集名称" min-width="250"><template #default="{ row }"><span class="table-link">{{ row.name }}</span></template></el-table-column>
        <el-table-column prop="version" label="版本" width="95" />
        <el-table-column prop="purpose" label="用途" width="105" />
        <el-table-column prop="format" label="格式" width="92" />
        <el-table-column prop="samples" label="样本数" width="120"><template #default="{ row }">{{ row.samples.toLocaleString() }}</template></el-table-column>
        <el-table-column prop="tokens" label="Token 数" width="135"><template #default="{ row }">{{ row.tokens.toLocaleString() }}</template></el-table-column>
        <el-table-column label="状态" width="105"><template #default="{ row }"><StatusPill v-bind="statusMeta(row.status)" /></template></el-table-column>
        <el-table-column prop="updatedAt" label="更新时间" width="165" />
        <el-table-column label="操作" width="150" fixed="right"><template #default="{ row }"><div class="table-actions"><span class="table-link" @click.stop="showPreview(row)">预览</span><span class="danger-link" @click.stop="removeDataset(row)">删除</span></div></template></el-table-column>
      </el-table>
    </PanelCard>

    <PanelCard v-if="selected" class="section-gap dataset-detail" flush>
      <div class="dataset-title"><div><strong>{{ selected.name }}</strong><span>{{ selected.version }}</span><StatusPill v-bind="statusMeta(selected.status)" /></div><span>更新时间：{{ selected.updatedAt }}</span></div>
      <el-tabs v-model="detailTab" class="dataset-tabs">
        <el-tab-pane label="概览" name="overview">
          <div class="dataset-overview">
            <div class="overview-card"><h3>基本信息</h3><dl><dt>SHA-256</dt><dd>{{ selected.sha256 ?? (useMocks ? '8f2e7c3b9a1d4f6e8b7c2a9d3f1e5b6c' : '—') }}</dd><dt>文件大小</dt><dd>{{ selected.size ?? (useMocks ? '245.62 MB' : '—') }}</dd><dt>文件名</dt><dd>{{ selected.fileName ?? '—' }}</dd><dt>样本数</dt><dd>{{ selected.samples.toLocaleString() }}</dd></dl></div>
            <div class="overview-card validation-card"><h3>校验概览</h3><div class="validation-metrics"><span><b class="number-positive">{{ selected.samples.toLocaleString() }}</b><small>记录</small></span><span><b class="danger-link">{{ selected.validationErrors?.length ?? (useMocks ? 3 : 0) }}</b><small>错误</small></span></div><template v-if="useMocks"><h3>Token 长度分布</h3><BaseChart :option="histogramOption" height="145px" /></template><el-empty v-else description="Token 分布统计尚未生成" :image-size="48" /></div>
            <div class="overview-card"><h3>错误详情（按行）</h3><el-table :data="selected.validationErrors ?? (useMocks ? [{line:12845,type:'格式错误',reason:'instruction 字段缺失'},{line:14672,type:'超长',reason:'输出长度超过上限（8192）'},{line:17123,type:'JSON 解析错误',reason:'无法解析 JSON 对象'}] : [])" size="small"><el-table-column prop="line" label="行号" width="74"/><el-table-column prop="type" label="错误类型" width="105"/><el-table-column prop="reason" label="错误原因" min-width="155"/></el-table></div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="数据预览" name="preview"><pre v-if="previewRows.length" class="json-preview">{{ previewRows.map((row) => JSON.stringify(row)).join('\n') }}</pre><el-empty v-else description="暂无可预览记录" /></el-tab-pane>
        <el-tab-pane label="校验报告" name="report"><el-result :icon="selected.status === 'failed' ? 'error' : 'success'" :title="statusMeta(selected.status).text" :sub-title="`${selected.samples.toLocaleString()} 条记录，${selected.validationErrors?.length ?? 0} 条校验错误`" /></el-tab-pane>
        <el-tab-pane label="版本记录" name="versions"><el-timeline v-if="versionRows.length"><el-timeline-item v-for="version in versionRows" :key="version.id" :timestamp="version.updatedAt">{{ version.version }} · {{ version.samples.toLocaleString() }} 条 · {{ statusMeta(version.status).text }}</el-timeline-item></el-timeline><el-empty v-else description="暂无版本记录" /><p class="version-hint">数据集内容不可原地覆盖；上传同名的新记录即可形成新版本。</p></el-tab-pane>
      </el-tabs>
    </PanelCard>
    <PanelCard v-else class="section-gap"><el-empty description="暂无数据集，请上传 JSONL 文件" /></PanelCard>

    <el-dialog v-model="uploadVisible" title="上传 JSONL 数据集" width="min(620px, 94vw)" :before-close="beforeCloseUpload" :close-on-click-modal="!uploadBusy" :close-on-press-escape="!uploadBusy" :show-close="!uploadBusy">
      <el-form label-position="top">
        <div class="two-column-form"><el-form-item label="数据集名称" required><el-input v-model="uploadForm.name" placeholder="请输入业务含义明确的名称" /></el-form-item><el-form-item label="用途"><el-select v-model="uploadForm.purpose" style="width:100%"><el-option label="继续预训练（CPT）" value="CPT"/><el-option label="指令微调（SFT）" value="SFT"/><el-option label="模型测评" value="Evaluation"/></el-select></el-form-item></div>
        <el-form-item label="版本号"><el-input v-model="uploadForm.version" placeholder="例如 v1.0.0" /></el-form-item>
        <el-form-item label="JSONL 文件" required><el-upload drag action="#" :auto-upload="false" accept=".jsonl" :limit="1" :on-change="handleFileChange"><el-icon class="el-icon--upload"><Upload /></el-icon><div class="el-upload__text">将文件拖到此处，或<em>点击选择</em></div><template #tip><div class="el-upload__tip">每行必须是独立 JSON 对象；上传后自动校验字段、长度与重复样本。</div></template></el-upload></el-form-item>
        <el-progress v-if="uploadBusy" :percentage="uploadProgress" :indeterminate="uploadProgress === 0" :duration="2" />
      </el-form>
      <template #footer><el-button @click="cancelUpload">{{ uploadBusy ? '终止上传' : '取消' }}</el-button><el-button type="primary" :loading="uploadBusy" :disabled="uploadBusy" @click="submitUpload">上传并校验</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.dataset-title { display:flex; align-items:center; justify-content:space-between; padding:15px 18px 0; }.dataset-title>div{display:flex;align-items:center;gap:10px}.dataset-title>span,.dataset-title>div>span{color:#748094;font-size:12px}.dataset-tabs{padding:0 18px 16px}.dataset-overview{display:grid;grid-template-columns:.8fr 1.2fr 1.1fr;gap:12px}.overview-card{min-width:0;padding:14px;border:1px solid #e1e7ef;border-radius:8px}.overview-card h3{margin:0 0 12px;font-size:13px}.overview-card dl{margin:0}.overview-card dt{margin-top:10px;color:#667388;font-size:11px}.overview-card dd{overflow:hidden;margin:3px 0 0;font-size:12px;text-overflow:ellipsis}.validation-metrics{display:grid;grid-template-columns:repeat(4,1fr);margin-bottom:14px}.validation-metrics span{display:flex;align-items:center;flex-direction:column;border-right:1px solid #e5eaf0}.validation-metrics span:last-child{border:0}.validation-metrics b{font-size:18px}.validation-metrics small{order:-1;color:#6d798d;font-size:10px}.warning-number{color:#ed8a16}.json-preview{margin:0;padding:16px;border-radius:7px;color:#d9e7fb;background:#071b32;line-height:1.8}.version-hint{margin:8px 0;color:#748094;font-size:12px}.two-column-form{display:grid;grid-template-columns:1fr 1fr;gap:14px}.el-upload{width:100%}:deep(.el-upload-dragger){width:100%}
@media(max-width:1200px){.dataset-overview{grid-template-columns:1fr 1fr}.dataset-overview>div:last-child{grid-column:1/-1}}@media(max-width:700px){.dataset-overview,.two-column-form{grid-template-columns:1fr}.dataset-overview>div:last-child{grid-column:auto}.dataset-title>span{display:none}}
</style>
