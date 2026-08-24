<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { Box, Connection, Cpu, FolderOpened, Refresh, Search, Upload } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import PageHeader from '@/components/PageHeader.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { ModelAsset, StatusTone } from '@/types/domain'

const models = ref<ModelAsset[]>([])
const selected = ref<ModelAsset | null>(null)
const inboxCandidates = ref<string[]>(useMocks ? ['/inbox/Qwen2-7B-Instruct', '/inbox/BGE-M3'] : [])
const drawerVisible = ref(false)
const importVisible = ref(false)
const importBusy = ref(false)
const scanBusy = ref(false)
const importProgress = ref(useMocks ? 72 : 100)

const filters = reactive({ keyword: '', source: '', type: '', status: '' })
const importForm = reactive({ source: 'huggingface', repository: '', revision: 'main', localPath: '', name: '', modelKind: 'instruct' })

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
    const result = await api.models.scanInbox()
    inboxCandidates.value = result.data
    if (!result.supported) ElMessage.warning(result.reason)
    else ElMessage.success(`扫描完成：发现 ${result.data.length} 个可导入目录`)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '扫描失败')
  } finally {
    scanBusy.value = false
  }
}

async function submitImport() {
  if (importForm.source !== 'inbox' && !importForm.repository) {
    ElMessage.warning('请填写模型仓库标识或 SFTP 路径')
    return
  }
  importBusy.value = true
  try {
    await api.models.import({ ...importForm })
    importVisible.value = false
    if (useMocks) importProgress.value = 8
    else models.value = await api.models.list()
    ElMessage.success(useMocks ? '导入任务已创建，将在后台校验 Safetensors 文件' : '模型资产记录已创建')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '创建导入任务失败')
  } finally {
    importBusy.value = false
  }
}

onMounted(async () => {
  try {
    models.value = await api.models.list()
    selected.value = models.value[0] ?? null
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '模型资产加载失败')
  }
})
</script>

<template>
  <div>
    <PageHeader title="模型资产" subtitle="统一纳管在线仓库、SFTP 与人工受控目录中的模型">
      <el-button :icon="FolderOpened" :loading="scanBusy" @click="scanInbox">扫描受控目录</el-button>
      <el-button type="primary" :icon="Upload" @click="importVisible = true">导入模型</el-button>
    </PageHeader>

    <div v-if="importProgress < 100" class="import-banner">
      <el-icon color="#1769f5"><Connection /></el-icon>
      <span>正在导入模型 Qwen2-7B-Instruct…</span>
      <el-progress :percentage="importProgress" :stroke-width="7" />
      <el-button link type="primary">查看详情</el-button>
      <el-button link @click="importProgress = 100">关闭</el-button>
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
          <el-option v-for="source in ['Hugging Face', 'ModelScope', 'SFTP', '受控目录', '训练产物']" :key="source" :value="source" />
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
          <el-segmented v-model="importForm.source" :options="[{ label: 'Hugging Face', value: 'huggingface' }, { label: 'ModelScope', value: 'modelscope' }, { label: 'SFTP', value: 'sftp' }, { label: '受控目录', value: 'inbox' }]" />
        </el-form-item>
        <el-form-item v-if="importForm.source !== 'inbox'" :label="importForm.source === 'sftp' ? '远程模型路径' : '仓库标识'">
          <el-input v-model="importForm.repository" :placeholder="importForm.source === 'sftp' ? '/data/models/Qwen2-7B-Instruct' : 'Qwen/Qwen2-7B-Instruct'" />
        </el-form-item>
        <el-form-item v-else label="受控目录候选">
          <el-select v-model="importForm.localPath" placeholder="请先扫描受控目录" style="width: 100%"><el-option v-for="candidate in inboxCandidates" :key="candidate" :label="candidate" :value="candidate" /></el-select>
        </el-form-item>
        <div class="two-column-form">
          <el-form-item label="资产名称"><el-input v-model="importForm.name" placeholder="留空则读取模型配置" /></el-form-item>
          <el-form-item v-if="importForm.source !== 'inbox'" label="固定 Revision"><el-input v-model="importForm.revision" placeholder="commit SHA" /></el-form-item>
          <el-form-item label="模型类型"><el-select v-model="importForm.modelKind" style="width:100%"><el-option label="Base" value="base"/><el-option label="Instruct" value="instruct"/><el-option label="Embedding" value="embedding"/></el-select></el-form-item>
        </div>
        <el-alert title="导入安全限制" description="仅接收 Safetensors；固定 revision 后下载；不执行模型仓库中的远程代码。" type="info" :closable="false" show-icon />
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
@media (max-width: 620px) { .import-banner { grid-template-columns: auto 1fr; }.import-banner .el-progress { grid-column: 1 / -1; }.two-column-form { grid-template-columns: 1fr; } }
</style>
