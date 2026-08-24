<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { Box, Connection, Cpu, DocumentChecked, FolderOpened, Refresh, Search, Upload } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRouter } from 'vue-router'

import PageHeader from '@/components/PageHeader.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { formatBytes, formatBytesPerSecond, formatDateTime } from '@/api/adapters'
import type { BackendInboxCandidate } from '@/api/contracts'
import type { ModelAsset, ModelImportStatus, ModelImportTask, StatusTone } from '@/types/domain'

const models = ref<ModelAsset[]>([])
const router = useRouter()
const selected = ref<ModelAsset | null>(null)
const inboxCandidates = ref<BackendInboxCandidate[]>([])
const importJobs = ref<ModelImportTask[]>([])
const drawerVisible = ref(false)
const importVisible = ref(false)
const importBusy = ref(false)
const scanBusy = ref(false)
let importPollTimer: number | undefined
let importRefreshRunning = false
const importProgressSamples = new Map<string, { bytes: number; sampledAt: number }>()
const importInstantRates = ref<Record<string, number>>({})

const filters = reactive({ keyword: '', source: '', type: '', status: '' })
const importForm = reactive({ source: 'huggingface', repository: '', revision: 'main', sourceDirectory: '', name: '', modelKind: 'instruct' })

const activeImport = computed(() => importJobs.value.find((job) => ['pending', 'transferring', 'validating', 'canceling'].includes(job.status)))
const activeImportProgress = computed(() => activeImport.value?.progressPercent ?? (activeImport.value?.status === 'pending' ? 0 : undefined))

const importStatusText: Record<ModelImportStatus, string> = {
  pending: '等待导入',
  transferring: '正在获取模型文件',
  validating: '正在执行安全校验',
  ready: '导入完成',
  failed: '导入失败',
  canceling: '正在取消',
  canceled: '已取消',
}

const importStageText: Record<ModelImportStatus, string> = {
  pending: '非抢占队列',
  transferring: '文件传输',
  validating: 'Safetensors 安全校验',
  ready: '发布完成',
  failed: '任务失败',
  canceling: '取消收敛',
  canceled: '已取消',
}

const importStatusMap: Record<ModelImportStatus, { text: string; tone: StatusTone }> = {
  pending: { text: '排队中', tone: 'warning' },
  transferring: { text: '传输中', tone: 'primary' },
  validating: { text: '校验中', tone: 'primary' },
  ready: { text: '已完成', tone: 'success' },
  failed: { text: '失败', tone: 'danger' },
  canceling: { text: '取消中', tone: 'warning' },
  canceled: { text: '已取消', tone: 'info' },
}

const filteredModels = computed(() => models.value.filter((model) => {
  const keywordMatch = !filters.keyword || model.name.toLowerCase().includes(filters.keyword.toLowerCase())
  return keywordMatch
    && (!filters.source || model.source === filters.source)
    && (!filters.type || model.type === filters.type)
    && (!filters.status || model.status === filters.status)
}))

const counts = computed(() => ({
  total: models.value.length,
  generation: models.value.filter((item) => item.type === 'generation').length,
  embedding: models.value.filter((item) => item.type === 'embedding').length,
  importing: importJobs.value.filter((item) => ['pending', 'transferring', 'validating', 'canceling'].includes(item.status)).length,
}))

const statusMap: Record<ModelAsset['status'], { text: string; tone: StatusTone }> = {
  available: { text: '可用', tone: 'success' },
  validating: { text: '校验中', tone: 'primary' },
  importing: { text: '导入中', tone: 'warning' },
  failed: { text: '导入失败', tone: 'danger' },
}
const statusMeta = (status: ModelAsset['status']) => statusMap[status]
const importStatusMeta = (status: ModelImportStatus) => importStatusMap[status]
const canCancelImport = (status: ModelImportStatus) => ['pending', 'transferring', 'validating'].includes(status)
const importStage = (status: ModelImportStatus) => importStageText[status]

function shortRevision(value?: string) {
  if (!value) return '—'
  return value.length > 16 ? `${value.slice(0, 12)}…` : value
}

function sourceReference(job: ModelImportTask) {
  return job.repository ?? (job.sourceDirectory ? `inbox://${job.sourceDirectory}` : '—')
}

function progressBytes(job: ModelImportTask) {
  return `${formatBytes(job.progressCompleted)} / ${formatBytes(job.progressTotal)}`
}

function transferRate(job: ModelImportTask) {
  if (job.status !== 'transferring') return job.status === 'validating' || job.status === 'ready' ? '传输已完成' : '—'
  const instant = importInstantRates.value[job.id]
  if (instant !== undefined) return `实时 ${formatBytesPerSecond(instant)}`
  const startedAt = job.startedAt ? Date.parse(job.startedAt) : Number.NaN
  const elapsedSeconds = (Date.now() - startedAt) / 1_000
  const average = Number.isFinite(elapsedSeconds) && elapsedSeconds > 0 && job.progressCompleted > 0
    ? job.progressCompleted / elapsedSeconds
    : undefined
  return average === undefined ? '等待进度样本' : `平均 ${formatBytesPerSecond(average)}`
}

function recordImportRates(jobs: ModelImportTask[]) {
  const sampledAt = Date.now()
  const nextRates: Record<string, number> = {}
  const currentIds = new Set(jobs.map((job) => job.id))
  for (const jobId of importProgressSamples.keys()) {
    if (!currentIds.has(jobId)) importProgressSamples.delete(jobId)
  }
  for (const job of jobs) {
    const previous = importProgressSamples.get(job.id)
    if (job.status === 'transferring' && previous && job.progressCompleted > previous.bytes && sampledAt > previous.sampledAt) {
      nextRates[job.id] = (job.progressCompleted - previous.bytes) * 1_000 / (sampledAt - previous.sampledAt)
    }
    importProgressSamples.set(job.id, { bytes: job.progressCompleted, sampledAt })
  }
  // 每轮没有新增字节就回退到任务平均速率，避免把上一次瞬时速度误当成当前速度。
  importInstantRates.value = nextRates
}

async function manualRefreshImports() {
  try {
    await refreshImports()
    ElMessage.success('导入任务已刷新')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '刷新导入任务失败')
  }
}

function checksumSummary(model: ModelAsset) {
  const manifest = model.manifest
  if (!manifest) return '未提供导入校验清单'
  const total = manifest.fileCount ?? manifest.files.length
  return `已记录 ${manifest.files.length} / ${total} 个文件的 SHA-256`
}

function showDetails(row: ModelAsset) {
  selected.value = row
  drawerVisible.value = true
}

function resetFilters() {
  Object.assign(filters, { keyword: '', source: '', type: '', status: '' })
}

async function scanInbox() {
  scanBusy.value = true
  try {
    // 扫描只建立候选清单，模型仍需管理员在导入对话框确认后才进入资产库。
    inboxCandidates.value = await api.models.scanInbox()
    const readyCount = inboxCandidates.value.filter((item) => item.ready_for_import).length
    ElMessage.success(`扫描完成：发现 ${readyCount} 个可导入目录`)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '扫描失败')
  } finally {
    scanBusy.value = false
  }
}

async function submitImport() {
  if (importForm.source !== 'controlled_directory' && !importForm.repository) {
    ElMessage.warning('请填写 namespace/model 格式的模型仓库标识')
    return
  }
  if (importForm.source === 'controlled_directory' && !importForm.sourceDirectory) {
    ElMessage.warning('请先扫描并选择一个可导入目录')
    return
  }
  importBusy.value = true
  try {
    await api.models.import({ ...importForm })
    importVisible.value = false
    await refreshImports()
    ElMessage.success('导入任务已创建；校验通过后才会发布为可用资产')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '创建导入任务失败')
  } finally {
    importBusy.value = false
  }
}

async function refreshImports() {
  if (importRefreshRunning) return
  importRefreshRunning = true
  try {
    const previousReadyIds = new Set(importJobs.value.filter((job) => job.status === 'ready').map((job) => job.id))
    const nextJobs = await api.models.imports()
    recordImportRates(nextJobs)
    importJobs.value = nextJobs
    const hasNewReadyJob = nextJobs.some((job) => job.status === 'ready' && !previousReadyIds.has(job.id))
    if (hasNewReadyJob) models.value = await api.models.list()
  } finally {
    importRefreshRunning = false
  }
}

async function cancelImport(job: ModelImportTask) {
  try {
    await api.models.cancelImport(job.id)
    await refreshImports()
    ElMessage.success('取消指令已记录')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '取消导入失败')
  }
}

function deployModel(model: ModelAsset) {
  void router.push({ path: '/deployments', query: { model: model.id } })
}

async function removeModel(model: ModelAsset) {
  try {
    await ElMessageBox.confirm('删除资产记录不会自动停止或改写引用它的任务；存在引用时控制面会拒绝。', `删除 ${model.name}`, { type: 'warning' })
    await api.models.remove(model.id)
    models.value = await api.models.list()
    if (selected.value?.id === model.id) selected.value = models.value[0] ?? null
    ElMessage.success('模型资产已删除')
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '删除模型资产失败')
  }
}

onMounted(async () => {
  try {
    const [nextModels, nextJobs] = await Promise.all([api.models.list(), api.models.imports()])
    models.value = nextModels
    recordImportRates(nextJobs)
    importJobs.value = nextJobs
    selected.value = models.value[0] ?? null
    // 只轮询轻量任务接口；页面卸载时清理定时器，避免后台继续请求。
    importPollTimer = window.setInterval(() => { void refreshImports().catch(() => undefined) }, 2_000)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '模型资产加载失败')
  }
})

onBeforeUnmount(() => {
  if (importPollTimer !== undefined) window.clearInterval(importPollTimer)
})
</script>

<template>
  <div>
    <PageHeader title="模型资产" subtitle="统一纳管在线仓库与管理员人工复制到受控目录中的模型">
      <el-button :icon="FolderOpened" :loading="scanBusy" @click="scanInbox">扫描受控目录</el-button>
      <el-button type="primary" :icon="Upload" @click="importVisible = true">导入模型</el-button>
    </PageHeader>

    <div v-if="activeImport" class="import-banner">
      <el-icon color="#1769f5"><Connection /></el-icon>
      <span>{{ activeImport.name }}：{{ importStatusText[activeImport.status] }}</span>
      <el-progress v-if="activeImportProgress !== undefined" :percentage="activeImportProgress" :stroke-width="7" />
      <el-progress v-else :percentage="100" :indeterminate="true" :duration="2" :stroke-width="7" />
      <span class="import-banner-rate">{{ progressBytes(activeImport) }} · {{ transferRate(activeImport) }}</span>
      <el-button link :disabled="activeImport.status === 'canceling'" @click="cancelImport(activeImport)">取消</el-button>
    </div>

    <div class="stats-grid asset-stats">
      <StatCard label="全部" :value="counts.total" :icon="Box" tone="blue" />
      <StatCard label="生成模型" :value="counts.generation" :icon="Connection" tone="green" />
      <StatCard label="Embedding" :value="counts.embedding" :icon="Cpu" tone="purple" />
      <StatCard label="导入进行中" :value="counts.importing" :icon="Refresh" tone="orange" />
    </div>

    <section class="panel-card section-gap flush">
      <div class="filters">
        <el-input v-model="filters.keyword" placeholder="搜索模型名称" clearable :prefix-icon="Search" />
        <el-select v-model="filters.source" placeholder="来源：全部" clearable>
          <el-option v-for="source in ['Hugging Face', 'ModelScope', '受控目录', '训练产物']" :key="source" :value="source" />
        </el-select>
        <el-select v-model="filters.type" placeholder="模型类型：全部" clearable>
          <el-option label="生成模型" value="generation" /><el-option label="Embedding" value="embedding" />
        </el-select>
        <el-select v-model="filters.status" placeholder="状态：全部" clearable>
          <el-option label="可用" value="available" /><el-option label="校验中" value="validating" /><el-option label="导入失败" value="failed" />
        </el-select>
        <el-button type="primary" :icon="Search">查询</el-button>
        <el-button :icon="Refresh" @click="resetFilters">重置</el-button>
      </div>
      <el-table :data="filteredModels" row-key="id" @row-click="showDetails">
        <el-table-column type="selection" width="46" />
        <el-table-column prop="name" label="模型名称" min-width="210"><template #default="{ row }"><span class="table-link">{{ row.name }}</span></template></el-table-column>
        <el-table-column prop="version" label="版本" width="92" />
        <el-table-column label="类型" width="100"><template #default="{ row }">{{ row.type === 'generation' ? '生成模型' : 'Embedding' }}</template></el-table-column>
        <el-table-column prop="source" label="来源" width="122" />
        <el-table-column prop="format" label="格式" width="112" />
        <el-table-column prop="size" label="大小" width="92" />
        <el-table-column label="状态" width="105"><template #default="{ row }"><StatusPill v-bind="statusMeta(row.status)" dot /></template></el-table-column>
        <el-table-column prop="updatedAt" label="更新时间" width="152" />
        <el-table-column label="操作" width="176" fixed="right">
          <template #default="{ row }"><div class="table-actions"><span class="table-link" @click.stop="showDetails(row)">详情</span><span class="table-link" @click.stop="deployModel(row)">部署</span><span class="danger-link" @click.stop="removeModel(row)">删除</span></div></template>
        </el-table-column>
      </el-table>
      <div class="table-footer"><span>共 {{ filteredModels.length }} 条</span><el-pagination background layout="prev, pager, next" :total="filteredModels.length" :page-size="10" /></div>
    </section>

    <section class="panel-card section-gap flush import-history">
      <div class="import-history-heading">
        <div>
          <h3><el-icon><DocumentChecked /></el-icon> 模型导入任务</h3>
          <p>字节进度来自导入 worker；实时速率由相邻两次进度样本计算，无样本时显示任务平均值。</p>
        </div>
        <el-button :icon="Refresh" @click="manualRefreshImports">刷新</el-button>
      </div>
      <el-table :data="importJobs.slice(0, 10)" row-key="id" empty-text="暂无导入任务">
        <el-table-column type="expand" width="44">
          <template #default="{ row }">
            <div class="import-expand">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="来源位置"><code>{{ sourceReference(row) }}</code></el-descriptions-item>
                <el-descriptions-item label="模型类型">{{ row.modelKind }}</el-descriptions-item>
                <el-descriptions-item label="Requested revision"><code>{{ row.requestedRevision ?? '—' }}</code></el-descriptions-item>
                <el-descriptions-item label="Resolved revision"><code>{{ row.resolvedRevision ?? '—' }}</code></el-descriptions-item>
                <el-descriptions-item label="已处理 / 总字节">{{ progressBytes(row) }}</el-descriptions-item>
                <el-descriptions-item label="速率">{{ transferRate(row) }}</el-descriptions-item>
                <el-descriptions-item label="开始时间">{{ row.startedAt ? formatDateTime(row.startedAt) : '—' }}</el-descriptions-item>
                <el-descriptions-item label="完成时间">{{ row.finishedAt ? formatDateTime(row.finishedAt) : '—' }}</el-descriptions-item>
                <el-descriptions-item label="结果资产 ID" :span="2"><code>{{ row.resultAssetId ?? '—' }}</code></el-descriptions-item>
              </el-descriptions>
              <el-alert v-if="row.errorMessage" class="import-error" title="导入失败原因" :description="row.errorMessage" type="error" :closable="false" show-icon />
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="任务" min-width="160" />
        <el-table-column prop="source" label="来源" width="112" />
        <el-table-column label="阶段" min-width="165"><template #default="{ row }">{{ importStage(row.status) }}</template></el-table-column>
        <el-table-column label="进度" min-width="145">
          <template #default="{ row }">
            <el-progress v-if="row.progressPercent !== undefined" :percentage="row.progressPercent" :stroke-width="6" />
            <span v-else class="muted-cell">{{ formatBytes(row.progressCompleted) }} / 未知</span>
          </template>
        </el-table-column>
        <el-table-column label="字节" min-width="170"><template #default="{ row }">{{ progressBytes(row) }}</template></el-table-column>
        <el-table-column label="传输速率" min-width="140"><template #default="{ row }">{{ transferRate(row) }}</template></el-table-column>
        <el-table-column label="Revision" width="130"><template #default="{ row }"><el-tooltip :content="row.resolvedRevision ?? row.requestedRevision ?? '未提供'" placement="top"><code>{{ shortRevision(row.resolvedRevision ?? row.requestedRevision) }}</code></el-tooltip></template></el-table-column>
        <el-table-column label="状态" width="100"><template #default="{ row }"><StatusPill v-bind="importStatusMeta(row.status)" dot /></template></el-table-column>
        <el-table-column label="操作" width="82" fixed="right"><template #default="{ row }"><el-button v-if="canCancelImport(row.status)" link type="danger" @click="cancelImport(row)">取消</el-button><span v-else class="muted-cell">—</span></template></el-table-column>
      </el-table>
      <div class="table-footer"><span>最近 {{ Math.min(importJobs.length, 10) }} 条</span><span>展开行可查看来源、完整 revision 与错误原因</span></div>
    </section>

    <el-drawer v-model="drawerVisible" title="模型详情" size="min(560px, 92vw)">
      <template v-if="selected">
        <div class="detail-heading">
          <div class="detail-icon"><el-icon><Box /></el-icon></div>
          <div><strong>{{ selected.name }}</strong><span>版本：{{ selected.version }}</span></div>
          <StatusPill v-bind="statusMeta(selected.status)" />
        </div>
        <el-divider content-position="left">能力</el-divider>
        <div class="capabilities"><template v-if="selected.type === 'generation'"><el-tag>Chat</el-tag><el-tag>Completion</el-tag></template><el-tag v-else>Embedding</el-tag></div>
        <el-divider content-position="left">基本信息</el-divider>
        <el-descriptions :column="1" label-width="100px">
          <el-descriptions-item label="模型类型">{{ selected.type === 'generation' ? '生成模型' : 'Embedding' }}</el-descriptions-item>
          <el-descriptions-item label="来源">{{ selected.source }}</el-descriptions-item>
          <el-descriptions-item label="来源定位"><code>{{ selected.sourceUri ?? '—' }}</code></el-descriptions-item>
          <el-descriptions-item label="Requested revision"><code>{{ selected.requestedRevision ?? '—' }}</code></el-descriptions-item>
          <el-descriptions-item label="Resolved revision"><code>{{ selected.resolvedRevision ?? selected.revision ?? '—' }}</code></el-descriptions-item>
          <el-descriptions-item label="模型族">{{ selected.family ?? '未提供' }}</el-descriptions-item>
          <el-descriptions-item label="架构">{{ selected.architecture ?? '未提供' }}</el-descriptions-item>
          <el-descriptions-item label="参数量">{{ selected.parameterCount === undefined ? '未提供（不推断）' : selected.parameterCount.toLocaleString() }}</el-descriptions-item>
          <el-descriptions-item label="权重精度">{{ selected.weightDtypes?.join(' / ') ?? '未提供' }}</el-descriptions-item>
          <el-descriptions-item label="格式">{{ selected.format }}</el-descriptions-item>
          <el-descriptions-item label="大小">{{ selected.size }}</el-descriptions-item>
          <el-descriptions-item label="上下文长度">{{ selected.contextLength?.toLocaleString() ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="受控路径"><code>{{ selected.path ?? '—' }}</code></el-descriptions-item>
          <el-descriptions-item label="更新时间">{{ selected.updatedAt }}</el-descriptions-item>
        </el-descriptions>
        <el-divider content-position="left">文件校验摘要</el-divider>
        <el-descriptions :column="1" label-width="100px">
          <el-descriptions-item label="Manifest">{{ selected.manifest ? '已记录' : '未提供' }}</el-descriptions-item>
          <el-descriptions-item label="文件统计">{{ selected.manifest ? `${selected.manifest.fileCount ?? selected.manifest.files.length} 个 / ${formatBytes(selected.manifest.totalSizeBytes)}` : '—' }}</el-descriptions-item>
          <el-descriptions-item label="SHA-256">{{ checksumSummary(selected) }}</el-descriptions-item>
          <el-descriptions-item label="资产级校验值"><code>{{ selected.checksum ?? '未提供' }}</code></el-descriptions-item>
        </el-descriptions>
        <el-table v-if="selected.manifest?.files.length" :data="selected.manifest.files.slice(0, 8)" size="small" class="manifest-files">
          <el-table-column prop="path" label="文件" min-width="145" show-overflow-tooltip />
          <el-table-column label="大小" width="78"><template #default="{ row }">{{ formatBytes(row.sizeBytes) }}</template></el-table-column>
          <el-table-column label="SHA-256" width="98"><template #default="{ row }"><el-tooltip :content="row.sha256" placement="top"><code>{{ shortRevision(row.sha256) }}</code></el-tooltip></template></el-table-column>
        </el-table>
        <p v-if="selected.manifest && selected.manifest.files.length > 8" class="manifest-more">仅预览前 8 个文件，完整校验值仍保存在资产 manifest 中。</p>
        <el-alert class="asset-security" title="安全策略" :description="selected.manifest ? '导入器已校验 Safetensors 并生成文件级 SHA-256 清单；trust_remote_code 全局关闭。' : '该资产没有可展示的导入 manifest；系统仍限定 Safetensors 格式且关闭 trust_remote_code。'" :type="selected.manifest ? 'success' : 'warning'" :closable="false" show-icon />
      </template>
    </el-drawer>

    <el-dialog v-model="importVisible" title="导入模型" width="min(620px, 92vw)">
      <el-form label-position="top">
        <el-form-item label="导入方式">
          <el-segmented v-model="importForm.source" :options="[{ label: 'Hugging Face', value: 'huggingface' }, { label: 'ModelScope', value: 'modelscope' }, { label: '受控目录', value: 'controlled_directory' }]" />
        </el-form-item>
        <el-form-item v-if="importForm.source !== 'controlled_directory'" label="仓库标识">
          <el-input v-model="importForm.repository" placeholder="Qwen/Qwen2-7B-Instruct" />
        </el-form-item>
        <el-form-item v-else label="受控目录候选">
          <el-select v-model="importForm.sourceDirectory" placeholder="请先扫描受控目录" style="width: 100%">
            <el-option v-for="candidate in inboxCandidates" :key="candidate.name" :label="candidate.ready_for_import ? candidate.path : `${candidate.path}（${candidate.reason ?? '不可导入'}）`" :value="candidate.name" :disabled="!candidate.ready_for_import" />
          </el-select>
        </el-form-item>
        <div class="two-column-form">
          <el-form-item label="资产名称"><el-input v-model="importForm.name" placeholder="留空则使用仓库或目录名称" /></el-form-item>
          <el-form-item v-if="importForm.source !== 'controlled_directory'" label="固定 Revision"><el-input v-model="importForm.revision" placeholder="建议填写 commit SHA" /></el-form-item>
          <el-form-item label="模型类型"><el-select v-model="importForm.modelKind" style="width:100%"><el-option label="Base" value="base"/><el-option label="Instruct" value="instruct"/><el-option label="Embedding" value="embedding"/></el-select></el-form-item>
        </div>
        <el-alert title="SFTP 仅用于人工传输" description="管理员先登录服务器，把完整模型目录复制到受控 inbox，再回到这里扫描并导入；平台不保存或使用 SFTP 凭证。" type="info" :closable="false" show-icon />
        <el-alert class="import-security" title="导入安全限制" description="仅接收 Safetensors；建议固定 commit revision；不执行模型仓库中的远程代码。" type="success" :closable="false" show-icon />
      </el-form>
      <template #footer><el-button @click="importVisible = false">取消</el-button><el-button type="primary" :loading="importBusy" @click="submitImport">创建导入任务</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.import-banner { display: grid; grid-template-columns: auto auto minmax(180px, 1fr) auto auto; align-items: center; gap: 12px; margin-bottom: 14px; padding: 9px 14px; border: 1px solid #cfe0ff; border-radius: 8px; background: #f4f8ff; font-size: 13px; }
.import-banner-rate { color: #526179; font-variant-numeric: tabular-nums; white-space: nowrap; }
.asset-stats { margin-top: 0; }
.table-footer { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; color: #68758a; font-size: 13px; }
.import-history-heading { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 17px 18px 13px; }
.import-history-heading h3 { display: flex; align-items: center; gap: 8px; margin: 0; color: #1e2b40; font-size: 15px; }
.import-history-heading h3 .el-icon { color: #1769f5; }
.import-history-heading p { margin: 6px 0 0; color: #748096; font-size: 12px; }
.import-expand { padding: 12px 22px 18px 54px; }
.import-error { margin-top: 12px; }
.muted-cell { color: #8a95a6; font-size: 12px; }
.detail-heading { display: grid; grid-template-columns: 48px 1fr auto; align-items: center; gap: 12px; }
.detail-icon { width: 48px; height: 48px; display: grid; place-items: center; border-radius: 9px; color: #1769f5; background: #eaf2ff; font-size: 22px; }
.detail-heading > div:nth-child(2) { display: flex; flex-direction: column; gap: 5px; }.detail-heading span { color: #7a8597; font-size: 12px; }
.capabilities { display: flex; gap: 8px; } code { font-size: 11px; word-break: break-all; }.two-column-form { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
.manifest-files { margin-top: 10px; }
.manifest-more { margin: 8px 2px 0; color: #748096; font-size: 12px; line-height: 1.6; }
.asset-security, .import-security { margin-top: 12px; }
@media (max-width: 760px) { .import-banner { grid-template-columns: auto 1fr; }.import-banner .el-progress { grid-column: 1 / -1; }.import-banner-rate { grid-column: 1 / -1; }.two-column-form { grid-template-columns: 1fr; }.import-history-heading { align-items: flex-start; }.import-history-heading p { display: none; }.import-expand { padding-left: 12px; } }
</style>
