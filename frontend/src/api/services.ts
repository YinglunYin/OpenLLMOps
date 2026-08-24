import { formatDateTime, toApiKey, toDataset, toDeployment, toEvaluationRunSummary, toEvaluationSummaries, toGpuDevice, toModelAsset, toTrainingJob } from './adapters'
import { http, setCsrfToken, useMocks } from './client'
import type {
  BackendAdminIdentity,
  BackendApiKey,
  BackendCreatedApiKey,
  BackendDashboardSummary,
  BackendDataset,
  BackendDeployment,
  BackendEvaluationRun,
  BackendGpuHistory,
  BackendGpuHistoryMetric,
  BackendGpuStatus,
  BackendInboxCandidate,
  BackendModelAsset,
  BackendModelImport,
  BackendTrainingArtifact,
  BackendTrainingArtifactKind,
  BackendTrainingArtifactManifest,
  BackendTrainingJob,
} from './contracts'
import { createOpenAIStreamParser } from './sse'
import type {
  ChatMessage,
  AdminIdentity,
  ApiKeySummary,
  CreatedApiKey,
  DashboardActivity,
  DashboardSummary,
  Dataset,
  Deployment,
  EvaluationRunSummary,
  EvaluationSummary,
  GpuDevice,
  ModelAsset,
  PlaygroundParams,
  TrainingJob,
} from '@/types/domain'

type MockData = typeof import('@/mock/data')
let mockDataPromise: Promise<MockData> | undefined
const mockDataLoader = import.meta.env.VITE_USE_MOCKS === 'true' ? () => import('@/mock/data') : undefined

function loadMockData(): Promise<MockData> {
  if (!mockDataLoader) throw new Error('Mock 数据未启用')
  // 动态加载保证真实模式不会在页面初始化时读取 fixture。
  mockDataPromise ??= mockDataLoader()
  return mockDataPromise
}

export interface DatasetUploadInput {
  name: string
  purpose: Dataset['purpose']
  description?: string
}

const clone = <T>(value: T): T => structuredClone(value)

async function mockMutation<T>(fixture: T): Promise<T> {
  await new Promise((resolve) => window.setTimeout(resolve, 420))
  return clone(fixture)
}

async function listModels(): Promise<ModelAsset[]> {
  if (useMocks) return clone((await loadMockData()).modelAssets)
  const { data } = await http.get<BackendModelAsset[]>('/v1/model-assets')
  return data.map(toModelAsset)
}

async function listDeployments(): Promise<Deployment[]> {
  if (useMocks) return clone((await loadMockData()).deployments)
  const { data } = await http.get<BackendDeployment[]>('/v1/deployments')
  return data.map(toDeployment)
}

async function listDatasets(): Promise<Dataset[]> {
  if (useMocks) return clone((await loadMockData()).datasets)
  const { data } = await http.get<BackendDataset[]>('/v1/datasets')
  return data.map(toDataset)
}

async function listTrainingJobs(): Promise<TrainingJob[]> {
  if (useMocks) return clone((await loadMockData()).trainingJobs)
  const [{ data: jobs }, models] = await Promise.all([
    http.get<BackendTrainingJob[]>('/v1/training-jobs'),
    listModels(),
  ])
  const modelNames = new Map(models.map((item) => [item.id, item.name]))
  return jobs.map((item) => toTrainingJob(item, modelNames.get(item.model_asset_id)))
}

async function listGpus(): Promise<GpuDevice[]> {
  if (useMocks) return clone((await loadMockData()).gpuDevices)
  const { data } = await http.get<BackendGpuStatus[]>('/v1/system/gpus')
  return data.map(toGpuDevice)
}

export type GpuHistoryRange = '1h' | '6h' | '24h'

async function listGpuHistory(metric: BackendGpuHistoryMetric, range: GpuHistoryRange, gpuIndexes: number[]): Promise<BackendGpuHistory[]> {
  if (useMocks) return []
  const durationSeconds = { '1h': 3_600, '6h': 21_600, '24h': 86_400 }[range]
  const stepSeconds = { '1h': 15, '6h': 60, '24h': 300 }[range]
  const end = new Date()
  const start = new Date(end.getTime() - durationSeconds * 1_000)
  return Promise.all(gpuIndexes.map(async (index) => (await http.get<BackendGpuHistory>(`/v1/system/gpus/${index}/history`, {
    params: { metric, start: start.toISOString(), end: end.toISOString(), step_seconds: stepSeconds },
  })).data))
}

let dashboardRequest: Promise<BackendDashboardSummary> | undefined

function loadDashboardSummary(): Promise<BackendDashboardSummary> {
  // 总览摘要与活动在同一响应中；并行调用复用同一个在途请求，避免重复聚合数据库。
  dashboardRequest ??= http.get<BackendDashboardSummary>('/v1/dashboard/summary')
    .then(({ data }) => data)
    .finally(() => { dashboardRequest = undefined })
  return dashboardRequest
}

async function evaluationComparison(): Promise<EvaluationSummary[]> {
  if (useMocks) return clone((await loadMockData()).evaluationSummary)
  const { data } = await http.get<BackendEvaluationRun[]>('/v1/evaluation-runs')
  const finished = data.find((run) => Object.keys(run.comparison).length > 0)
  return finished ? toEvaluationSummaries(finished) : []
}

async function listEvaluationRuns(): Promise<EvaluationRunSummary[]> {
  if (useMocks) return clone([
    { id: 'eval-domain', name: 'domain-compare-0520', model: 'ChineseLM-8B-Domain', datasets: 'C-Eval / CMMLU / 领域测试集', progress: 100, status: 'completed', updatedAt: '2024-05-20 11:00:00' },
    { id: 'eval-base', name: 'base-regression-0521', model: 'ChineseLM-8B-Base', datasets: 'C-Eval / CMMLU', progress: 36, status: 'running', updatedAt: '2024-05-20 10:52:00' },
  ] satisfies EvaluationRunSummary[])
  const [{ data: runs }, models, datasets] = await Promise.all([
    http.get<BackendEvaluationRun[]>('/v1/evaluation-runs'),
    listModels(),
    listDatasets(),
  ])
  const modelNames = new Map(models.map((item) => [item.id, item.name]))
  const datasetNames = new Map(datasets.map((item) => [item.id, item.name]))
  return runs.map((run) => toEvaluationRunSummary(run, modelNames.get(run.candidate_model_asset_id), run.custom_dataset_id ? datasetNames.get(run.custom_dataset_id) : undefined))
}

function normalizeModelImport(payload: Record<string, unknown>) {
  const source = String(payload.source ?? 'controlled_directory')
  const repository = String(payload.repository ?? '')
  const sourceDirectory = String(payload.sourceDirectory ?? '')
  const name = String(payload.name || repository.split('/').at(-1) || sourceDirectory || '待导入模型')
  return {
    name,
    source,
    repository: source === 'controlled_directory' ? null : repository,
    revision: source === 'controlled_directory' ? null : payload.revision || null,
    source_directory: source === 'controlled_directory' ? sourceDirectory : null,
    model_kind: payload.modelKind || 'instruct',
  }
}

function normalizeDeployment(payload: Record<string, unknown>) {
  const gpuCount = Number(payload.gpuCount ?? 1)
  return {
    name: payload.name,
    served_model_name: payload.servedModelName || payload.name,
    model_asset_id: payload.modelAssetId,
    task_type: payload.serviceType === 'embedding' ? 'embedding' : 'generate',
    gpu_ids: Array.from({ length: gpuCount }, (_, index) => index),
    tensor_parallel_size: gpuCount,
    simplified_config: {
      max_model_len: payload.maxModelLen,
      gpu_memory_utilization: payload.gpuMemoryUtilization,
      dtype: payload.dtype,
    },
    vllm_args: parseAdvancedArgs(String(payload.advancedArgs ?? '')),
  }
}

function normalizeDeploymentUpdate(payload: Record<string, unknown>) {
  const gpuCount = Number(payload.gpuCount ?? 1)
  return {
    name: payload.name,
    gpu_ids: Array.from({ length: gpuCount }, (_, index) => index),
    tensor_parallel_size: gpuCount,
    simplified_config: {
      max_model_len: payload.maxModelLen,
      gpu_memory_utilization: payload.gpuMemoryUtilization,
      dtype: payload.dtype,
    },
    vllm_args: parseAdvancedArgs(String(payload.advancedArgs ?? '')),
  }
}

function normalizeTraining(payload: Record<string, unknown>) {
  const gpuCount = Number(payload.gpuCount ?? 1)
  return {
    name: String(payload.name),
    model_asset_id: payload.modelAssetId,
    dataset_id: payload.datasetId,
    stage: String(payload.stage).toLowerCase(),
    algorithm: String(payload.algorithm).toLowerCase(),
    gpu_ids: Array.from({ length: gpuCount }, (_, index) => index),
    training_config: {
      num_train_epochs: payload.epochs,
      learning_rate: payload.learningRate,
      cutoff_len: payload.cutoffLen,
      per_device_train_batch_size: payload.batchSize,
      gradient_accumulation_steps: payload.gradientAccumulation,
      logging_steps: payload.loggingSteps,
      save_steps: payload.saveSteps,
      warmup_ratio: payload.warmupRatio,
      lora_rank: payload.loraRank,
      lora_alpha: payload.loraAlpha,
      lora_dropout: payload.loraDropout,
      freeze_trainable_layers: payload.freezeTrainableLayers,
      ...(payload.maxSamples == null ? {} : { max_samples: payload.maxSamples }),
      seed: payload.seed,
      ...(payload.stage === 'SFT' ? { template: payload.template } : {}),
    },
  }
}

function normalizeEvaluation(payload: Record<string, unknown>) {
  const selected = Array.isArray(payload.datasets) ? payload.datasets.map(String) : []
  return {
    name: payload.name,
    base_model_asset_id: payload.baseModelAssetId,
    candidate_model_asset_id: payload.candidateModelAssetId,
    custom_dataset_id: payload.customDatasetId || null,
    builtin_datasets: selected.filter((item) => item === 'ceval' || item === 'cmmlu'),
    gpu_ids: [0],
  }
}

export function parseAdvancedArgs(value: string): Record<string, string | number | boolean | null> {
  const result: Record<string, string | number | boolean | null> = {}
  for (const line of value.split('\n').map((item) => item.trim()).filter(Boolean)) {
    const [rawKey, ...rest] = line.split(/\s+/)
    if (!rawKey?.startsWith('--')) throw new Error(`vLLM 参数必须以 -- 开头：${line}`)
    const key = rawKey.slice(2).replaceAll('-', '_')
    if (!/^[a-z][a-z0-9_]*$/.test(key)) throw new Error(`vLLM 参数名无效：${rawKey}`)
    if (Object.hasOwn(result, key)) throw new Error(`vLLM 参数重复：${rawKey}`)
    if (!rest.length) {
      result[key] = true
      continue
    }
    const rawValue = rest.join(' ')
    if (/^(true|false|null)$/.test(rawValue)) result[key] = JSON.parse(rawValue) as boolean | null
    else if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(rawValue)) result[key] = Number(rawValue)
    else if (/^"(?:[^"\\]|\\.)*"$/.test(rawValue)) result[key] = JSON.parse(rawValue) as string
    else result[key] = rawValue
  }
  return result
}

export const api = {
  auth: {
    async login(username: string, password: string): Promise<AdminIdentity> {
      if (useMocks) return { username: username || 'admin', authMethod: 'disabled' }
      const { data } = await http.post<BackendAdminIdentity>('/v1/auth/login', { username, password })
      setCsrfToken(data.csrf_token)
      return { username: data.username, authMethod: data.auth_method, expiresAt: data.expires_at ?? undefined }
    },
    async me(): Promise<AdminIdentity> {
      if (useMocks) return { username: '管理员', authMethod: 'disabled' }
      const { data } = await http.get<BackendAdminIdentity>('/v1/auth/me')
      setCsrfToken(data.csrf_token)
      return { username: data.username, authMethod: data.auth_method, expiresAt: data.expires_at ?? undefined }
    },
    async logout(): Promise<void> {
      if (useMocks) return
      try { await http.post('/v1/auth/logout') }
      finally { setCsrfToken(null) }
    },
  },
  dashboard: {
    async summary(): Promise<DashboardSummary> {
      if (useMocks) return clone((await loadMockData()).dashboardSummary)
      const data = await loadDashboardSummary()
      return {
        modelCount: data.models.total,
        runningDeployments: data.deployments.running,
        runningTrainingJobs: data.training_jobs.running,
        availableGpus: data.gpus.free,
        totalGpus: data.gpus.total,
      }
    },
    async activities(): Promise<DashboardActivity[]> {
      if (useMocks) return clone([
        { id: 'activity-deploy', time: '11:00:12', text: '部署任务已启动', detail: 'chatglm3-6b 服务部署到 GPU 0', tone: 'success' },
        { id: 'activity-eval', time: '10:58:41', text: '测评任务已创建', detail: 'llama3-8b 评测已加入队列', tone: 'warning' },
        { id: 'activity-failed', time: '10:38:07', text: '训练任务失败', detail: 'internlm2-20b 显存不足', tone: 'danger' },
      ] satisfies DashboardActivity[])
      const data = await loadDashboardSummary()
      const resourceLabels = { model_asset: '模型资产', model_import: '模型导入', deployment: '推理部署', training_job: '训练任务', evaluation_run: '测评任务' }
      return data.recent_activity.map((item) => ({
        id: `${item.resource_type}-${item.resource_id}`,
        time: formatDateTime(item.occurred_at).slice(11),
        text: `${resourceLabels[item.resource_type]} · ${item.status}`,
        detail: item.name,
        tone: ['failed', 'invalid'].includes(item.status) ? 'danger' : ['ready', 'running', 'succeeded'].includes(item.status) ? 'success' : ['pending', 'queued', 'transferring', 'validating'].includes(item.status) ? 'warning' : 'info',
      }))
    },
  },
  models: {
    list: listModels,
    import: async (payload: Record<string, unknown>) => useMocks
      ? mockMutation({ id: crypto.randomUUID() })
      : (await http.post<BackendModelImport>('/v1/model-imports', normalizeModelImport(payload))).data,
    async imports(): Promise<BackendModelImport[]> {
      if (useMocks) return []
      return (await http.get<BackendModelImport[]>('/v1/model-imports')).data
    },
    async cancelImport(id: string): Promise<BackendModelImport> {
      if (useMocks) return mockMutation({ id, status: 'canceled' } as BackendModelImport)
      return (await http.post<BackendModelImport>(`/v1/model-imports/${id}/cancel`)).data
    },
    async scanInbox(): Promise<BackendInboxCandidate[]> {
      if (useMocks) return [
        { name: 'Qwen2-7B-Instruct', path: '/inbox/Qwen2-7B-Instruct', file_count: 12, size_bytes: 15_032_385_536, ready_for_import: true, reason: null },
        { name: 'BGE-M3', path: '/inbox/BGE-M3', file_count: 9, size_bytes: 2_274_918_400, ready_for_import: true, reason: null },
      ]
      return (await http.get<BackendInboxCandidate[]>('/v1/model-inbox')).data
    },
    remove: async (id: string) => useMocks ? mockMutation(true) : (await http.delete(`/v1/model-assets/${id}`), true),
  },
  deployments: {
    list: listDeployments,
    create: async (payload: Record<string, unknown>) => useMocks
      ? mockMutation({ id: crypto.randomUUID() })
      : (await http.post<BackendDeployment>('/v1/deployments', normalizeDeployment(payload))).data,
    start: async (id: string) => useMocks ? mockMutation(true) : (await http.post(`/v1/deployments/${id}/start`), true),
    stop: async (id: string) => useMocks ? mockMutation(true) : (await http.post(`/v1/deployments/${id}/stop`), true),
    update: async (id: string, payload: Record<string, unknown>) => useMocks ? mockMutation(payload) : (await http.patch(`/v1/deployments/${id}`, normalizeDeploymentUpdate(payload))).data,
    remove: async (id: string) => useMocks ? mockMutation(true) : (await http.delete(`/v1/deployments/${id}`), true),
  },
  datasets: {
    list: listDatasets,
    upload: async (payload: DatasetUploadInput, file?: File) => {
      if (useMocks) return mockMutation({ id: crypto.randomUUID() })
      if (!file) throw new Error('请选择 JSONL 文件')
      const body = new FormData()
      body.append('name', payload.name)
      body.append('dataset_type', payload.purpose.toLowerCase())
      body.append('description', payload.description ?? '')
      body.append('file', file)
      // 不手写 Content-Type，让浏览器为 FormData 自动生成带 boundary 的 multipart 请求头。
      return (await http.post<BackendDataset>('/v1/datasets/upload', body)).data
    },
    preview: async (id: string) => useMocks ? [] : (await http.get<Array<Record<string, unknown>>>(`/v1/datasets/${id}/preview`)).data,
    remove: async (id: string) => useMocks ? mockMutation(true) : (await http.delete(`/v1/datasets/${id}`), true),
  },
  training: {
    list: listTrainingJobs,
    create: async (payload: Record<string, unknown>) => useMocks
      ? mockMutation({ id: crypto.randomUUID() })
      : (await http.post<BackendTrainingJob>('/v1/training-jobs', normalizeTraining(payload))).data,
    stop: async (id: string) => useMocks ? mockMutation(true) : (await http.post(`/v1/training-jobs/${id}/terminate`), true),
    remove: async (id: string) => useMocks ? mockMutation(true) : (await http.delete(`/v1/training-jobs/${id}`), true),
    async artifacts(id: string): Promise<BackendTrainingArtifact[]> {
      if (useMocks) return clone([
        { kind: 'checkpoint', path: '/checkpoints/demo/checkpoint-1000', file_count: 8, size_bytes: 38_400_000, archive_filename: 'training-demo-checkpoint.tar.gz' },
        { kind: 'adapter', path: '/checkpoints/demo', file_count: 5, size_bytes: 24_200_000, archive_filename: 'training-demo-adapter.tar.gz' },
        { kind: 'merged', path: '/checkpoints/demo/merged', file_count: 12, size_bytes: 15_032_385_536, archive_filename: 'training-demo-merged.tar.gz' },
      ] satisfies BackendTrainingArtifact[])
      return (await http.get<BackendTrainingArtifactManifest>(`/v1/training-jobs/${id}/artifacts`)).data.artifacts
    },
    artifactDownloadUrl(id: string, kind: BackendTrainingArtifactKind): string {
      const base = String(http.defaults.baseURL ?? '/api').replace(/\/$/, '')
      return `${base}/v1/training-jobs/${encodeURIComponent(id)}/artifacts/${kind}/download`
    },
    async publishModel(id: string): Promise<ModelAsset> {
      if (useMocks) {
        return mockMutation({ id: crypto.randomUUID(), name: 'trained-demo', version: '训练产物', type: 'generation', source: '训练产物', format: 'Safetensors', size: '14.0 GB', status: 'available', updatedAt: '刚刚' })
      }
      const { data } = await http.post<BackendModelAsset>(`/v1/training-jobs/${id}/publish-model`)
      return toModelAsset(data)
    },
  },
  evaluations: {
    list: listEvaluationRuns,
    comparison: evaluationComparison,
    create: async (payload: Record<string, unknown>) => useMocks
      ? mockMutation({ id: crypto.randomUUID() })
      : (await http.post<BackendEvaluationRun>('/v1/evaluation-runs', normalizeEvaluation(payload))).data,
    cancel: async (id: string) => useMocks ? mockMutation(true) : (await http.post(`/v1/evaluation-runs/${id}/cancel`), true),
    remove: async (id: string) => useMocks ? mockMutation(true) : (await http.delete(`/v1/evaluation-runs/${id}`), true),
  },
  apiKeys: {
    async list(): Promise<ApiKeySummary[]> {
      if (useMocks) return clone([
        { id: 'key-playground', name: 'playground', prefix: 'sk-8H3…', active: true, createdAt: '2024-05-20 10:00:00', lastUsedAt: '刚刚' },
        { id: 'key-evaluation', name: 'evaluation-runner', prefix: 'sk-KD9…', active: true, createdAt: '2024-05-18 10:00:00', lastUsedAt: '3 分钟前' },
      ])
      const { data } = await http.get<BackendApiKey[]>('/v1/api-keys')
      return data.map(toApiKey)
    },
    async create(name: string): Promise<CreatedApiKey> {
      if (useMocks) return { id: crypto.randomUUID(), name, prefix: 'sk-demo…', key: 'sk-demo-not-a-real-secret', active: true, createdAt: '刚刚', lastUsedAt: '从未使用' }
      const { data } = await http.post<BackendCreatedApiKey>('/v1/api-keys', { name })
      return { ...toApiKey(data), key: data.key }
    },
    async revoke(id: string): Promise<ApiKeySummary> {
      if (useMocks) return { id, name: '已撤销', prefix: 'sk-…', active: false, createdAt: '—', lastUsedAt: '—' }
      return toApiKey((await http.post<BackendApiKey>(`/v1/api-keys/${id}/revoke`)).data)
    },
  },
  resources: { gpus: listGpus, history: listGpuHistory },
}

interface StreamChatOptions {
  messages: ChatMessage[]
  params: PlaygroundParams
  signal: AbortSignal
  onToken: (token: string) => void
}

export async function streamChat({ messages, params, signal, onToken }: StreamChatOptions) {
  if (useMocks) {
    const mockText = '这是来自 OpenLLMOps 演示服务的流式响应。当前模型部署健康，您可以通过右侧参数面板继续调整采样策略。'
    for (const char of mockText) {
      if (signal.aborted) throw new DOMException('生成已取消', 'AbortError')
      await new Promise((resolve) => window.setTimeout(resolve, 24))
      onToken(char)
    }
    return
  }

  // OpenAI Compatible 网关保留标准根路径，不拼接控制面 /api 前缀。
  const openaiBase = (import.meta.env.VITE_OPENAI_BASE_URL || '').replace(/\/$/, '')
  const apiKey = sessionStorage.getItem('openllmops_api_key')
  const response = await fetch(`${openaiBase}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-API-Key': apiKey } : {}) },
    body: JSON.stringify({
      model: params.model,
      messages: messages.map(({ role, content }) => ({ role, content })),
      temperature: params.temperature,
      top_p: params.topP,
      max_tokens: params.maxTokens,
      repetition_penalty: params.repetitionPenalty,
      seed: params.seed,
      stream: true,
    }),
    signal,
  })

  if (!response.ok || !response.body) throw new Error(`请求失败：HTTP ${response.status}`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = createOpenAIStreamParser(onToken)
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    parser.push(decoder.decode(value, { stream: true }))
  }
  parser.push(decoder.decode())
  parser.finish()
}
