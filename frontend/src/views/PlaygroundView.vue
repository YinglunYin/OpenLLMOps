<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ChatDotRound, CopyDocument, Delete, Operation, Plus, Promotion, Setting, User } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api, streamChat } from '@/api/services'
import { useMocks } from '@/api/client'
import type { ChatMessage, PlaygroundMetrics, PlaygroundParams } from '@/types/domain'

const conversations = useMocks ? [
  { title: '如何优化大模型推理性能？', time: '刚刚' }, { title: '介绍一下向量数据库', time: '昨天 15:42' }, { title: '推荐一些机器学习书籍', time: '昨天 10:18' }, { title: '解释什么是注意力机制', time: '05-18 21:33' }, { title: 'Python 中的装饰器是什么', time: '05-18 16:07' },
] : []
const messages = ref<ChatMessage[]>([
  { id: 'sys-1', role: 'system', content: '你是一个专业、严谨且乐于助人的 AI 助手。' },
  ...(useMocks ? [
  { id: 'user-1', role: 'user', content: '如何优化大模型推理性能？', createdAt: '11:01:12' },
  { id: 'assistant-1', role: 'assistant', content: '优化大模型推理性能可以从模型、算法、系统和工程实践等多个层面入手：\n\n1. 模型层面：使用低精度量化、结构化剪枝、知识蒸馏与合并优化。\n2. 算法层面：利用 KV Cache、连续批处理和高效注意力实现。\n3. 系统层面：合理配置张量并行、显存利用率和请求并发度。', createdAt: '11:01:18' },
  ] satisfies ChatMessage[] : []),
])
const modelOptions = ref<string[]>(useMocks ? ['ChineseLM-8B-Instruct', 'Qwen2-7B-Instruct'] : [])
const params = reactive<PlaygroundParams>({ model: useMocks ? 'ChineseLM-8B-Instruct' : '', temperature: .7, topP: .9, maxTokens: 2048, repetitionPenalty: 1, stream: true })
const input = ref('')
const generating = ref(false)
const metrics = ref<PlaygroundMetrics | null>(useMocks ? {
  totalDurationMs: 1640, ttftMs: 186, inputTokens: 128, outputTokens: 356, outputTokensPerSecond: 42.8,
} : null)
const chatScroll = ref<HTMLElement>()
let controller: AbortController | undefined

async function scrollToBottom() {
  await nextTick()
  chatScroll.value?.scrollTo({ top: chatScroll.value.scrollHeight, behavior: 'smooth' })
}

async function sendMessage() {
  const content = input.value.trim()
  if (!content || generating.value) return
  if (!params.model) { ElMessage.warning('当前没有运行中的生成模型'); return }
  messages.value.push({ id: crypto.randomUUID(), role: 'user', content, createdAt: new Date().toLocaleTimeString('zh-CN', { hour12: false }) })
  input.value = ''
  const assistant: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: '', createdAt: '' }
  messages.value.push(assistant)
  generating.value = true
  metrics.value = null
  const requestController = new AbortController()
  controller = requestController
  await scrollToBottom()
  try {
    metrics.value = await streamChat({ messages: messages.value.slice(0, -1), params, signal: requestController.signal, onToken: (token) => { assistant.content += token; void scrollToBottom() } })
    assistant.createdAt = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  } catch (error) {
    if (!(error instanceof DOMException && error.name === 'AbortError')) {
      assistant.content ||= '请求失败，请检查部署状态与 API Key。'
      ElMessage.error(error instanceof Error ? error.message : '生成请求失败')
    }
  } finally {
    if (assistant.content && !assistant.createdAt) assistant.createdAt = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    if (controller === requestController) {
      generating.value = false
      controller = undefined
    }
  }
}

function stopGeneration() { controller?.abort() }
function clearChat() { stopGeneration(); messages.value = messages.value.filter((item) => item.role === 'system'); metrics.value = null }
function newChat() { clearChat(); input.value = '' }

async function copyMessage(content: string) {
  try {
    await navigator.clipboard.writeText(content)
    ElMessage.success('助手回复已复制')
  } catch {
    ElMessage.error('复制失败，请检查浏览器剪贴板权限')
  }
}

function formatMetric(value: number | null | undefined, fractionDigits = 0): string {
  return value === null || value === undefined ? '—' : value.toFixed(fractionDigits)
}

onBeforeUnmount(stopGeneration)
onMounted(async () => {
  if (useMocks) return
  try {
    const deployments = await api.deployments.list()
    modelOptions.value = deployments.filter((item) => item.status === 'running' && item.serviceType === 'generation').map((item) => item.model)
    params.model = modelOptions.value[0] ?? ''
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '运行模型加载失败')
  }
})
</script>

<template>
  <div class="playground-page">
    <PageHeader title="Playground" subtitle="直接验证已部署生成模型的 OpenAI Compatible 流式与非流式能力" />

    <section class="playground-shell">
      <div class="playground-toolbar">
        <el-select v-model="params.model" class="model-select" placeholder="暂无运行中的生成模型"><el-option v-for="model in modelOptions" :key="model" :label="`${model} · 运行中`" :value="model"/></el-select>
        <StatusPill :text="modelOptions.length ? '运行中' : '无可用模型'" :tone="modelOptions.length ? 'success' : 'warning'" dot />
        <el-divider direction="vertical" />
        <el-tag effect="plain">Chat</el-tag>
        <code>/v1/chat/completions</code>
        <el-button class="clear-button" :icon="Delete" @click="clearChat">清空对话</el-button>
      </div>

      <div class="playground-body">
        <aside class="conversation-list">
          <el-button type="primary" :icon="Plus" @click="newChat">新建对话</el-button>
          <button v-for="(conversation,index) in conversations" :key="conversation.title" :class="['conversation-item',{active:index===0}]">
            <el-icon><ChatDotRound/></el-icon><span><strong>{{ conversation.title }}</strong><small>{{ conversation.time }}</small></span>
          </button>
          <span v-if="!conversations.length" class="conversation-empty">会话仅保存在当前页面，不读取演示历史</span>
        </aside>

        <div class="chat-area">
          <div ref="chatScroll" class="message-list">
            <article v-for="message in messages" :key="message.id" :class="['message',`message-${message.role}`]">
              <div class="message-icon"><el-icon><Setting v-if="message.role==='system'"/><User v-else-if="message.role==='user'"/><Operation v-else/></el-icon></div>
              <div class="message-content"><header><strong>{{ message.role==='system'?'系统提示':message.role==='user'?'用户':'助手' }}</strong><span class="message-actions"><time>{{ message.createdAt }}</time><el-button v-if="message.role==='assistant' && message.content" text circle size="small" :icon="CopyDocument" aria-label="复制助手回复" @click="copyMessage(message.content)" /></span></header><div>{{ message.content }}<i v-if="generating && message === messages.at(-1)" class="typing-caret" /></div></div>
            </article>
          </div>
          <div class="composer">
            <el-input v-model="input" type="textarea" :autosize="{minRows:2,maxRows:5}" resize="none" placeholder="输入消息，按 Enter 发送；Shift + Enter 换行" @keydown.enter.exact.prevent="sendMessage" />
            <el-button v-if="generating" :icon="Operation" @click="stopGeneration">停止生成</el-button>
            <el-button type="primary" :icon="Promotion" :disabled="!input.trim() || generating || !params.model" aria-label="发送消息" @click="sendMessage" />
          </div>
        </div>

        <aside class="parameter-panel">
          <h2>推理参数</h2>
          <label>系统提示词<el-input v-model="messages[0]!.content" type="textarea" :rows="3" /></label>
          <label><span>Temperature <small>ⓘ</small></span><div class="slider-row"><el-slider v-model="params.temperature" :min="0" :max="2" :step=".1"/><el-input-number v-model="params.temperature" :min="0" :max="2" :step=".1" :controls="false"/></div></label>
          <label><span>Top P</span><div class="slider-row"><el-slider v-model="params.topP" :min="0" :max="1" :step=".05"/><el-input-number v-model="params.topP" :min="0" :max="1" :step=".05" :controls="false"/></div></label>
          <label>最大 Token<el-input-number v-model="params.maxTokens" :min="1" :max="32768" :step="128" :controls="false" /></label>
          <label>重复惩罚<el-input-number v-model="params.repetitionPenalty" :min=".1" :max="2" :step=".1" :controls="false" /></label>
          <label>Seed<el-input-number v-model="params.seed" placeholder="可选，留空随机" :controls="false" /></label>
          <div class="switch-row"><span>流式输出</span><el-switch v-model="params.stream" /></div>
          <div class="switch-row"><span>保存会话</span><el-switch :model-value="false" disabled /></div>
        </aside>
      </div>
    </section>

    <div class="playground-metrics">
      <div><span>总耗时</span><strong>{{ formatMetric(metrics?.totalDurationMs) }} <small v-if="metrics">ms</small></strong></div><div><span>TTFT</span><strong class="number-positive">{{ formatMetric(metrics?.ttftMs) }} <small v-if="metrics?.ttftMs != null">ms</small></strong></div><div><span>生成速度</span><strong class="number-primary">{{ formatMetric(metrics?.outputTokensPerSecond, 1) }} <small v-if="metrics?.outputTokensPerSecond != null">tok/s</small></strong></div><div><span>输入</span><strong class="purple-number">{{ formatMetric(metrics?.inputTokens) }} <small v-if="metrics?.inputTokens != null">tokens</small></strong></div><div><span>输出</span><strong class="orange-number">{{ formatMetric(metrics?.outputTokens) }} <small v-if="metrics?.outputTokens != null">tokens</small></strong></div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.playground-page{height:calc(100vh - 96px);min-height:720px;display:flex;flex-direction:column}.playground-page .page-header{flex:0 0 auto}.playground-shell{flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden;border:1px solid #dfe6ef;border-radius:9px;background:#fff}.playground-toolbar{height:61px;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid #e0e6ee}.model-select{width:290px}.playground-toolbar code{padding:7px 10px;border:1px solid #e0e6ee;border-radius:5px;color:#46546b;background:#fafbfd;font-size:12px}.clear-button{margin-left:auto}.playground-body{flex:1;min-height:0;display:grid;grid-template-columns:255px minmax(350px,1fr) 265px}.conversation-list{display:flex;flex-direction:column;gap:4px;padding:15px 10px;border-right:1px solid #e0e6ee}.conversation-list>.el-button{margin-bottom:7px}.conversation-item{display:flex;align-items:flex-start;gap:9px;padding:10px;border:0;border-radius:6px;text-align:left;background:transparent;cursor:pointer}.conversation-item:hover,.conversation-item.active{color:#1769f5;background:#edf5ff}.conversation-item>span{min-width:0;display:flex;flex-direction:column;gap:5px}.conversation-item strong{overflow:hidden;font-size:12px;white-space:nowrap;text-overflow:ellipsis}.conversation-item small{color:#8792a3;font-size:10px}.chat-area{min-width:0;display:flex;flex-direction:column;background:#f8fafc}.message-list{flex:1;min-height:0;overflow-y:auto;padding:16px}.message{display:grid;grid-template-columns:30px 1fr;gap:10px;margin-bottom:12px;padding:14px;border:1px solid #e0e6ee;border-radius:8px;background:#fff}.message-system{background:#f6f8fb}.message-icon{width:28px;height:28px;display:grid;place-items:center;border-radius:6px;color:#1769f5;background:#eaf2ff}.message-assistant .message-icon{color:#0da261;background:#e8f8ef}.message-content header{display:flex;justify-content:space-between;margin-bottom:7px}.message-content header strong{font-size:13px}.message-assistant header strong{color:#0b9e5f}.message-content time{color:#8792a3;font-size:10px}.message-content>div{color:#344055;font-size:13px;line-height:1.75;white-space:pre-wrap}.typing-caret{display:inline-block;width:2px;height:15px;margin-left:2px;vertical-align:middle;background:#1769f5;animation:blink .8s infinite}.composer{display:flex;align-items:flex-end;gap:8px;margin:0 16px 15px;padding:8px;border:1px solid #dce3ec;border-radius:8px;background:#fff}.composer:focus-within{border-color:#8ab8ff;box-shadow:0 0 0 2px #1769f512}.composer :deep(.el-textarea__inner){box-shadow:none!important}.parameter-panel{overflow-y:auto;padding:17px 16px;border-left:1px solid #e0e6ee}.parameter-panel h2{margin:0 0 20px;font-size:17px}.parameter-panel label{display:flex;flex-direction:column;gap:8px;margin-bottom:17px;color:#3f4b5e;font-size:12px}.parameter-panel label>span{display:flex;justify-content:space-between}.slider-row{display:grid;grid-template-columns:1fr 64px;align-items:center;gap:10px}.slider-row .el-input-number{width:64px}.parameter-panel label>.el-input-number{width:100%}.switch-row{display:flex;justify-content:space-between;margin:14px 0;font-size:12px}.playground-metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:12px}.playground-metrics>div{display:flex;flex-direction:column;gap:8px;padding:13px 16px;border:1px solid #dfe6ef;border-radius:8px;background:#fff}.playground-metrics span{font-size:11px}.playground-metrics strong{font-size:20px}.playground-metrics small{font-size:10px;font-weight:500}.purple-number{color:#7c3aed}.orange-number{color:#ed8a16}@keyframes blink{50%{opacity:0}}
.conversation-empty{padding:14px 8px;color:#8792a3;font-size:11px;line-height:1.6;text-align:center}
.message-actions{display:flex;align-items:center;gap:4px}
@media(max-width:1100px){.playground-body{grid-template-columns:200px 1fr}.parameter-panel{display:none}}@media(max-width:720px){.playground-page{height:auto;min-height:0}.playground-shell{height:calc(100vh - 200px);min-height:560px}.playground-body{grid-template-columns:1fr}.conversation-list{display:none}.playground-toolbar code,.playground-toolbar .el-divider,.playground-toolbar>.status-pill{display:none}.model-select{width:min(260px,65vw)}.playground-metrics{grid-template-columns:repeat(2,1fr)}}
</style>
