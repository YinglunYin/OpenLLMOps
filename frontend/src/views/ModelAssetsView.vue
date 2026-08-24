<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { Box, Connection, Cpu, FolderOpened, Refresh, Search, Upload } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import PageHeader from '@/components/PageHeader.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import type { BackendInboxCandidate, BackendModelImport } from '@/api/contracts'
import type { ModelAsset, StatusTone } from '@/types/domain'

const models = ref<ModelAsset[]>([])
const selected = ref<ModelAsset | null>(null)
const inboxCandidates = ref<BackendInboxCandidate[]>([])
const importJobs = ref<BackendModelImport[]>([])
const drawerVisible = ref(false)
const importVisible = ref(false)
const importBusy = ref(false)
const scanBusy = ref(false)
let importPollTimer: number | undefined

const filters = reactive({ keyword: '', source: '', type: '', status: '' })
const importForm = reactive({ source: 'huggingface', repository: '', revision: 'main', sourceDirectory: '', name: '', modelKind: 'instruct' })

const activeImport = computed(() => importJobs.value.find((job) => ['pending', 'transferring', 'validating', 'canceling'].includes(job.status)))
const activeImportProgress = computed(() => activeImport.value?.progress_percent ?? (activeImport.value?.status === 'pending' ? 0 : undefined))

const importStatusText: Record<BackendModelImport['status'], string> = {
  pending: '等待导入',
  transferring: '正在获取模型文件',
  validating: '正在执行安全校验',
  ready: '导入完成',
  failed: '导入失败',
  canceling: '正在取消',
  canceled: '已取消',
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
  importing: models.value.filter((item) => item.status === 'importing' || item.status === 'validating').length,
}))

const statusMap: Record<ModelAsset['status'], { text: string; tone: StatusTone }> = {
  available: { text: '可用', tone: 'success' },
  validating: { text: '校验中', tone: 'primary' },
  importing: { text: '导入中', tone: 'warning' },
  failed: { text: '导入失败', tone: 'danger' },
}
const statusMeta = (status: ModelAsset['status']) => statusMap[status]

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
  const previousReadyIds = new Set(importJobs.value.filter((job) => job.status === 'ready').map((job) => job.id))
  importJobs.value = await api.models.imports()
  const hasNewReadyJob = importJobs.value.some((job) => job.status === 'ready' && !previousReadyIds.has(job.id))
  if (hasNewReadyJob) models.value = await api.models.list()
}

async function cancelImport(job: BackendModelImport) {
  try {
    await api.models.cancelImport(job.id)
    await refreshImports()
    ElMessage.success('取消指令已记录')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '取消导入失败')
  }
}

onMounted(async () => {
  try {
    ;[models.value, importJobs.value] = await Promise.all([api.models.list(), api.models.imports()])
    selected.value = models.value[0] ?? null
    // 只轮询轻量任务接口；页面卸载时清理定时器，避免后台继续请求。
    importPollTimer = window.setInterval(() => { void refreshImports() }, 2_000)
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
      <el-button link type="primary" @click="importVisible = true">查看任务</el-button>
      <el-button link :disabled="activeImport.status === 'canceling'" @click="cancelImport(activeImport)">取消</el-button>
    </div>

    <div class="stats-grid asset-stats">
      <StatCard label="全部" :value="counts.total" :icon="Box" tone="blue" />
      <StatCard label="生成模型" :value="counts.generation" :icon="Connection" tone="green" />
      <StatCard label="Embedding" :value="counts.embedding" :icon="Cpu" tone="purple" />
      <StatCard label="校验中" :value="counts.importing" :icon="Refresh" tone="orange" />
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
          <template #default="{ row }"><div class="table-actions"><span class="table-link" @click.stop="showDetails(row)">详情</span><span class="table-link">部署</span><span class="danger-link">删除</span></div></template>
        </el-table-column>
      </el-table>
      <div class="table-footer"><span>共 {{ filteredModels.length }} 条</span><el-pagination background layout="prev, pager, next" :total="filteredModels.length" :page-size="10" /></div>
    </section>

    <el-drawer v-model="drawerVisible" title="模型详情" size="410px">
      <template v-if="selected">
        <div class="detail-heading">
          <div class="detail-icon"><el-icon><Box /></el-icon></div>
          <div><strong>{{ selected.name }}</strong><span>版本：{{ selected.version }}</span></div>
          <StatusPill v-bind="statusMeta(selected.status)" />
        </div>
        <el-divider content-position="left">能力</el-divider>
        <div class="capabilities"><el-tag>Chat</el-tag><el-tag>Completion</el-tag><el-tag v-if="selected.type === 'embedding'">Embedding</el-tag></div>
        <el-divider content-position="left">基本信息</el-divider>
        <el-descriptions :column="1" label-width="100px">
          <el-descriptions-item label="模型类型">{{ selected.type === 'generation' ? '生成模型' : 'Embedding' }}</el-descriptions-item>
          <el-descriptions-item label="来源">{{ selected.source }}</el-descriptions-item>
          <el-descriptions-item label="格式">{{ selected.format }}</el-descriptions-item>
          <el-descriptions-item label="大小">{{ selected.size }}</el-descriptions-item>
          <el-descriptions-item label="上下文长度">{{ selected.contextLength?.toLocaleString() ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="受控路径"><code>{{ selected.path ?? '—' }}</code></el-descriptions-item>
          <el-descriptions-item label="更新时间">{{ selected.updatedAt }}</el-descriptions-item>
        </el-descriptions>
        <el-alert title="安全策略" description="已验证 Safetensors 格式，trust_remote_code 全局关闭。" type="success" :closable="false" show-icon />
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
        <el-table v-if="importJobs.length" :data="importJobs.slice(0, 5)" size="small" class="import-jobs">
          <el-table-column prop="name" label="最近任务" min-width="145" />
          <el-table-column label="状态" width="105"><template #default="{ row }">{{ importStatusText[row.status as BackendModelImport['status']] }}</template></el-table-column>
          <el-table-column label="进度" width="85"><template #default="{ row }">{{ row.progress_percent == null ? '—' : `${row.progress_percent}%` }}</template></el-table-column>
        </el-table>
      </el-form>
      <template #footer><el-button @click="importVisible = false">取消</el-button><el-button type="primary" :loading="importBusy" @click="submitImport">创建导入任务</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.import-banner { display: grid; grid-template-columns: auto auto minmax(180px, 1fr) auto auto; align-items: center; gap: 12px; margin-bottom: 14px; padding: 9px 14px; border: 1px solid #cfe0ff; border-radius: 8px; background: #f4f8ff; font-size: 13px; }
.asset-stats { margin-top: 0; }
.table-footer { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; color: #68758a; font-size: 13px; }
.detail-heading { display: grid; grid-template-columns: 48px 1fr auto; align-items: center; gap: 12px; }
.detail-icon { width: 48px; height: 48px; display: grid; place-items: center; border-radius: 9px; color: #1769f5; background: #eaf2ff; font-size: 22px; }
.detail-heading > div:nth-child(2) { display: flex; flex-direction: column; gap: 5px; }.detail-heading span { color: #7a8597; font-size: 12px; }
.capabilities { display: flex; gap: 8px; } code { font-size: 11px; word-break: break-all; }.two-column-form { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
.import-security { margin-top: 10px; }.import-jobs { margin-top: 14px; }
@media (max-width: 620px) { .import-banner { grid-template-columns: auto 1fr; }.import-banner .el-progress { grid-column: 1 / -1; }.two-column-form { grid-template-columns: 1fr; } }
</style>
