<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { CopyDocument, Key, Lock, Plus, User } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { useMocks } from '@/api/client'
import { api } from '@/api/services'
import type { ApiKeySummary, CreatedApiKey } from '@/types/domain'

const activeTab = ref('sources')
const showHfToken = ref(false)
const showMsToken = ref(false)
const keyDialog = ref(false)
const newKeyName = ref('')
const keyBusy = ref(false)
const createdKey = ref<CreatedApiKey | null>(null)
const sourceForm = reactive({ hfToken: useMocks ? 'hf_••••••••••••••••••••' : '', msToken: useMocks ? 'ms_••••••••••••••••••••' : '', sftpHost: useMocks ? '10.0.0.22' : '', sftpPort: 22, sftpUser: useMocks ? 'llmops' : '', sftpSecret: useMocks ? '••••••••••••••••' : '', sftpPath: useMocks ? '/data/models' : '', inbox: useMocks ? '/data/openllmops/models/inbox' : '', safetensorsOnly: true, trustRemoteCode: false, pinRevision: true })
const baseForm = reactive({ systemName: useMocks ? 'OpenLLMOps' : '', adminName: useMocks ? '管理员' : '', timezone: useMocks ? 'Asia/Shanghai' : '', language: useMocks ? '简体中文' : '' })
const storageForm = reactive({ modelRoot: useMocks ? '/data/openllmops/models' : '', datasetRoot: useMocks ? '/data/openllmops/datasets' : '', checkpointRoot: useMocks ? '/data/openllmops/checkpoints' : '', minFreeSpace: useMocks ? 200 : 50 })

const maskedHf = computed(() => showHfToken.value ? 'hf_demo_token_not_a_real_secret' : sourceForm.hfToken)
const maskedMs = computed(() => showMsToken.value ? 'ms_demo_token_not_a_real_secret' : sourceForm.msToken)
const apiKeyRows = ref<ApiKeySummary[]>([])
const runtimeRows = useMocks ? [{name:'控制面 API',version:'0.1.0',status:'运行中'},{name:'节点代理',version:'0.1.0',status:'运行中'},{name:'vLLM',version:'0.10.1',status:'可用'},{name:'LLaMA-Factory',version:'0.9.4',status:'可用'},{name:'PostgreSQL',version:'16',status:'运行中'},{name:'Prometheus',version:'3.x',status:'运行中'}] : []
const activeApiKeyCount = computed(() => apiKeyRows.value.filter((item) => item.active).length)
function unsupported() { ElMessage.warning('控制面尚未提供该设置的读取或保存端点') }
function save(message = '设置已保存') { if (!useMocks) { unsupported(); return }; ElMessage.success(message) }
function testConnection(name: string) { if (!useMocks) { unsupported(); return }; ElMessage.success(`${name} 连接测试通过`) }
function openKeyDialog() { newKeyName.value = ''; createdKey.value = null; keyDialog.value = true }
async function createKey() {
  if (!newKeyName.value) { ElMessage.warning('请填写 Key 名称'); return }
  keyBusy.value = true
  try {
    createdKey.value = await api.apiKeys.create(newKeyName.value)
    apiKeyRows.value.unshift(createdKey.value)
    ElMessage.success('API Key 已创建，请立即复制保存')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'API Key 创建失败')
  } finally {
    keyBusy.value = false
  }
}
async function revokeKey(row: ApiKeySummary) {
  if (!row.active) return
  try {
    const revoked = await api.apiKeys.revoke(row.id)
    Object.assign(row, revoked)
    ElMessage.success('API Key 已撤销')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'API Key 撤销失败')
  }
}
async function copyCreatedKey() {
  if (!createdKey.value) return
  try {
    await navigator.clipboard.writeText(createdKey.value.key)
    ElMessage.success('密钥已复制')
  } catch {
    ElMessage.warning('浏览器未允许剪贴板访问，请手动复制')
  }
}
function useCreatedKeyInPlayground() {
  if (!createdKey.value) return
  sessionStorage.setItem('openllmops_api_key', createdKey.value.key)
  ElMessage.success('已在当前浏览器会话中用于 Playground')
}

onMounted(async () => {
  try { apiKeyRows.value = await api.apiKeys.list() }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : 'API Key 列表加载失败') }
})
</script>

<template>
  <div>
    <PageHeader title="系统设置" subtitle="配置模型来源、受控存储、安全凭证与运行时版本" />
    <el-alert v-if="!useMocks" class="settings-api-alert" title="部分设置端点尚未接入" description="API Key 已连接真实控制面；模型来源、存储、HTTPS 与运行时版本只呈现可配置字段和明确空态。" type="warning" :closable="false" show-icon />
    <el-tabs v-model="activeTab" class="settings-tabs">
      <el-tab-pane label="基础设置" name="base">
        <PanelCard title="控制台信息"><el-form label-width="130px" class="settings-form"><el-form-item label="系统名称"><el-input v-model="baseForm.systemName"/></el-form-item><el-form-item label="管理员名称"><el-input v-model="baseForm.adminName"/></el-form-item><el-form-item label="时区"><el-select v-model="baseForm.timezone"><el-option label="Asia/Shanghai" value="Asia/Shanghai"/></el-select></el-form-item><el-form-item label="界面语言"><el-select v-model="baseForm.language"><el-option label="简体中文" value="简体中文"/></el-select></el-form-item><el-form-item><el-button type="primary" @click="save()">保存设置</el-button></el-form-item></el-form></PanelCard>
      </el-tab-pane>
      <el-tab-pane label="存储与目录" name="storage">
        <PanelCard title="受控存储路径"><el-form label-width="150px" class="settings-form wide"><el-form-item label="模型资产目录"><el-input v-model="storageForm.modelRoot"/></el-form-item><el-form-item label="数据集目录"><el-input v-model="storageForm.datasetRoot"/></el-form-item><el-form-item label="Checkpoint 目录"><el-input v-model="storageForm.checkpointRoot"/></el-form-item><el-form-item label="最低可用磁盘"><el-input-number v-model="storageForm.minFreeSpace" :min="50"/><span class="unit">GB</span></el-form-item><el-form-item><el-button type="primary" @click="save('存储设置已保存')">保存设置</el-button></el-form-item></el-form></PanelCard>
      </el-tab-pane>
      <el-tab-pane label="模型来源" name="sources">
        <div class="settings-layout">
          <div class="settings-main">
            <PanelCard title="在线模型源">
              <div class="source-table"><div class="source-row header"><span>名称</span><span>Token</span><span>状态</span><span>操作</span></div><div class="source-row"><strong>Hugging Face</strong><el-input :model-value="maskedHf" readonly><template #suffix><button v-if="useMocks" class="text-button" @click="showHfToken=!showHfToken">{{ showHfToken?'隐藏':'显示' }}</button></template></el-input><StatusPill :text="useMocks ? '已配置' : '未读取'" :tone="useMocks ? 'success' : 'info'"/><div><el-button @click="testConnection('Hugging Face')">测试连接</el-button><el-button type="primary" plain @click="unsupported">编辑</el-button></div></div><div class="source-row"><strong>ModelScope</strong><el-input :model-value="maskedMs" readonly><template #suffix><button v-if="useMocks" class="text-button" @click="showMsToken=!showMsToken">{{ showMsToken?'隐藏':'显示' }}</button></template></el-input><StatusPill :text="useMocks ? '已配置' : '未读取'" :tone="useMocks ? 'success' : 'info'"/><div><el-button @click="testConnection('ModelScope')">测试连接</el-button><el-button type="primary" plain @click="unsupported">编辑</el-button></div></div></div>
            </PanelCard>
            <PanelCard title="SFTP 模型源" class="section-gap">
              <el-form label-width="95px" class="sftp-form"><div class="three-column-form"><el-form-item label="主机"><el-input v-model="sourceForm.sftpHost"/></el-form-item><el-form-item label="端口"><el-input-number v-model="sourceForm.sftpPort" :min="1" :max="65535" controls-position="right"/></el-form-item><el-form-item label="用户名"><el-input v-model="sourceForm.sftpUser"/></el-form-item></div><div class="two-column-form"><el-form-item label="密码 / 密钥"><el-input v-model="sourceForm.sftpSecret" type="password" show-password/></el-form-item><el-form-item label="远程目录"><el-input v-model="sourceForm.sftpPath"/></el-form-item></div><div class="form-actions"><StatusPill :text="useMocks ? '连接正常' : '未读取'" :tone="useMocks ? 'success' : 'info'"/><el-button @click="testConnection('SFTP')">测试连接</el-button><el-button type="primary" @click="save('SFTP 配置已保存')">保存</el-button></div></el-form>
            </PanelCard>
            <PanelCard title="人工导入目录" class="section-gap"><div class="inbox-row"><span>受控目录（只读）</span><el-input v-model="sourceForm.inbox" readonly/><StatusPill :text="useMocks ? '目录可用' : '未读取'" :tone="useMocks ? 'success' : 'info'"/></div><el-alert title="操作说明" description="管理员将模型复制到受控目录后，在“模型资产”页面扫描并人工确认导入。" type="info" :closable="false" show-icon/></PanelCard>
            <PanelCard title="模型拉取与安全策略" class="section-gap"><div class="policy-grid"><label><span>仅接受 Safetensors<small>拒绝 Pickle 等可执行序列化格式</small></span><el-switch v-model="sourceForm.safetensorsOnly"/></label><label><span>允许 trust_remote_code<small>全局安全锁，本版本保持关闭</small></span><el-switch v-model="sourceForm.trustRemoteCode" disabled/></label><label><span>固定 revision / SHA-256<small>保证导入结果可复现</small></span><el-switch v-model="sourceForm.pinRevision"/></label></div></PanelCard>
          </div>
          <aside class="settings-aside">
            <PanelCard title="系统安全"><div class="security-list"><div><el-icon><Lock/></el-icon><span>HTTPS</span><StatusPill :text="useMocks ? '已启用' : '未读取'" :tone="useMocks ? 'success' : 'info'"/></div><div><el-icon><Key/></el-icon><span>有效 API Key</span><b>{{ activeApiKeyCount }}</b></div><div><el-icon><User/></el-icon><span>单管理员模式</span><StatusPill :text="useMocks ? '已启用' : '会话认证'" :tone="useMocks ? 'success' : 'primary'"/></div></div></PanelCard>
            <PanelCard title="运行时物料" class="section-gap"><div v-if="useMocks" class="runtime-list"><div><strong>vLLM 镜像</strong><span>vllm/vllm-openai:0.10.1</span></div><div><strong>LLaMA-Factory 镜像</strong><span>hiyouga/llamafactory:0.9.4</span></div><div><strong>NVIDIA Driver</strong><span>稳定版（由宿主机提供）</span></div><div><strong>基础镜像 Digest</strong><span>sha256:9b8d3a6c7f4e…</span></div></div><el-empty v-else description="运行时版本端点尚未接入" :image-size="54"/></PanelCard>
          </aside>
        </div>
      </el-tab-pane>
      <el-tab-pane label="HTTPS 与 API Key" name="security">
        <div class="settings-layout security-page">
          <PanelCard title="HTTPS 证书">
            <template v-if="useMocks"><el-descriptions :column="1" border><el-descriptions-item label="状态"><StatusPill text="已启用" tone="success"/></el-descriptions-item><el-descriptions-item label="证书主题">CN=openllmops.internal</el-descriptions-item><el-descriptions-item label="到期时间">2027-08-24</el-descriptions-item><el-descriptions-item label="私钥存储">Docker Secret（只读挂载）</el-descriptions-item></el-descriptions><el-button class="section-gap">更换证书</el-button></template>
            <el-empty v-else description="HTTPS 配置端点尚未接入" :image-size="54"/>
          </PanelCard>
          <PanelCard title="API Key"><template #actions><el-button type="primary" :icon="Plus" @click="openKeyDialog">创建 Key</el-button></template><el-table :data="apiKeyRows" empty-text="暂无 API Key"><el-table-column prop="name" label="名称" min-width="140"/><el-table-column prop="prefix" label="前缀" width="125"/><el-table-column prop="createdAt" label="创建时间" min-width="160"/><el-table-column prop="lastUsedAt" label="最后使用" min-width="160"/><el-table-column label="状态" width="85"><template #default="{row}"><StatusPill :text="row.active ? '有效' : '已撤销'" :tone="row.active ? 'success' : 'info'"/></template></el-table-column><el-table-column label="操作" width="70"><template #default="{row}"><button class="danger-link plain-action" :disabled="!row.active" @click="revokeKey(row)">撤销</button></template></el-table-column></el-table></PanelCard>
        </div>
      </el-tab-pane>
      <el-tab-pane label="运行时版本" name="runtime"><PanelCard title="组件版本"><el-table :data="runtimeRows" empty-text="运行时版本端点尚未接入"><el-table-column prop="name" label="组件"/><el-table-column prop="version" label="版本"/><el-table-column label="状态"><template #default="{row}"><StatusPill :text="row.status" tone="success"/></template></el-table-column></el-table></PanelCard></el-tab-pane>
    </el-tabs>

    <el-dialog v-model="keyDialog" title="创建 API Key" width="480px" :close-on-click-modal="!createdKey">
      <template v-if="createdKey"><el-alert title="密钥只显示这一次" description="控制面仅保存不可逆摘要，请在关闭前复制并安全保存。" type="warning" :closable="false" show-icon/><el-input class="created-key" :model-value="createdKey.key" readonly><template #append><el-button :icon="CopyDocument" @click="copyCreatedKey">复制</el-button></template></el-input></template>
      <el-form v-else label-position="top"><el-form-item label="Key 名称"><el-input v-model="newKeyName" placeholder="标识调用方，例如 evaluation-runner"/></el-form-item><el-alert title="密钥只显示一次" description="创建后请立即复制并安全保存，系统仅存储不可逆哈希。" type="warning" :closable="false" show-icon/></el-form>
      <template #footer><template v-if="createdKey"><el-button @click="useCreatedKeyInPlayground">用于当前 Playground</el-button><el-button type="primary" @click="keyDialog=false">我已保存，关闭</el-button></template><template v-else><el-button @click="keyDialog=false">取消</el-button><el-button type="primary" :loading="keyBusy" @click="createKey">创建</el-button></template></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.settings-api-alert{margin-bottom:14px}.settings-tabs{margin-top:-12px}.settings-layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px}.source-table{display:grid}.source-row{display:grid;grid-template-columns:160px minmax(260px,1fr) 110px 210px;align-items:center;gap:14px;padding:10px 12px;border:1px solid #e0e6ee;border-top:0}.source-row:first-child{border-top:1px solid #e0e6ee;border-radius:6px 6px 0 0}.source-row:last-child{border-radius:0 0 6px 6px}.source-row.header{padding-top:8px;padding-bottom:8px;color:#59667a;background:#f7f9fc;font-size:12px;font-weight:600}.source-row strong{font-size:13px}.text-button{border:0;color:#1769f5;background:none;cursor:pointer;font-size:11px}.three-column-form{display:grid;grid-template-columns:1fr .8fr 1fr;gap:14px}.two-column-form{display:grid;grid-template-columns:1fr 1fr;gap:14px}.form-actions{display:flex;align-items:center;justify-content:flex-end;gap:10px}.form-actions .status-pill{margin-right:auto}.inbox-row{display:grid;grid-template-columns:150px 1fr auto;align-items:center;gap:13px;margin-bottom:13px;font-size:13px}.policy-grid{display:grid;grid-template-columns:repeat(3,1fr)}.policy-grid label{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:4px 20px;border-right:1px solid #e1e7ef}.policy-grid label:first-child{padding-left:0}.policy-grid label:last-child{padding-right:0;border:0}.policy-grid label>span{display:flex;flex-direction:column;gap:5px;font-size:12px}.policy-grid small{color:#7b8799;font-size:10px}.security-list{display:grid;gap:8px}.security-list>div{display:grid;grid-template-columns:25px 1fr auto;align-items:center;gap:8px;padding:13px;border:1px solid #e1e7ef;border-radius:6px;font-size:13px}.security-list .el-icon{color:#12a865;font-size:18px}.runtime-list>div{display:flex;flex-direction:column;gap:5px;padding:13px;border-bottom:1px solid #e1e7ef}.runtime-list>div:last-child{border:0}.runtime-list strong{font-size:12px}.runtime-list span{color:#667388;font-size:11px;word-break:break-all}.settings-form{max-width:620px}.settings-form.wide{max-width:900px}.unit{margin-left:8px;color:#798598}.security-page{grid-template-columns:.75fr 1.25fr}.created-key{margin-top:16px}.plain-action{padding:0;border:0;background:none;cursor:pointer}.plain-action:disabled{opacity:.45;cursor:not-allowed}
@media(max-width:1100px){.settings-layout{grid-template-columns:1fr}.settings-aside{display:grid;grid-template-columns:1fr 1fr;gap:14px}.settings-aside .section-gap{margin-top:0}.policy-grid{grid-template-columns:1fr}.policy-grid label{padding:12px 0;border-right:0;border-bottom:1px solid #e1e7ef}}@media(max-width:760px){.source-row{grid-template-columns:1fr}.source-row.header{display:none}.source-row>div:last-child{display:flex}.three-column-form,.two-column-form,.settings-aside{grid-template-columns:1fr}.inbox-row{grid-template-columns:1fr}.security-page{grid-template-columns:1fr}}
</style>
