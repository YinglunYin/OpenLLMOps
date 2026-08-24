export type StatusTone = 'success' | 'primary' | 'warning' | 'danger' | 'info'

export interface GpuDevice {
  index: number
  name: string
  utilization: number
  memoryUsed: number
  memoryTotal: number
  temperature: number
  power: number
  powerLimit: number
  state: 'idle' | 'inference' | 'training' | 'reserved' | 'unmanaged' | 'unknown'
  task?: string
  telemetryAvailable?: boolean
  telemetryReason?: string
}

export interface ModelAsset {
  id: string
  name: string
  version: string
  type: 'generation' | 'embedding'
  source: 'Hugging Face' | 'ModelScope' | 'SFTP' | '受控目录' | '训练产物'
  format: 'Safetensors'
  size: string
  status: 'available' | 'validating' | 'importing' | 'failed'
  updatedAt: string
  contextLength?: number
  path?: string
  sourceUri?: string
  revision?: string
  requestedRevision?: string
  resolvedRevision?: string
  family?: string
  architecture?: string
  parameterCount?: number
  weightDtypes?: string[]
  checksum?: string
  errorMessage?: string
  manifest?: ModelManifestSummary
}

/**
 * 模型导入器生成的不可变校验清单。字段保持可选，因为训练产物与旧资产可能没有
 * 导入清单；页面必须明确显示“未提供”，不能从文件大小猜测模型参数量。
 */
export interface ModelManifestFile {
  path: string
  sizeBytes: number
  sha256: string
}

export interface ModelManifestSummary {
  modelType?: string
  architecture?: string
  totalSizeBytes?: number
  fileCount?: number
  parameterCount?: number
  weightDtypes?: string[]
  checksum?: string
  requestedRevision?: string
  resolvedRevision?: string
  files: ModelManifestFile[]
}

export type ModelImportStatus = 'pending' | 'transferring' | 'validating' | 'ready' | 'failed' | 'canceling' | 'canceled'

export interface ModelImportTask {
  id: string
  name: string
  source: 'Hugging Face' | 'ModelScope' | '受控目录'
  sourceKey: 'huggingface' | 'modelscope' | 'controlled_directory'
  repository?: string
  sourceDirectory?: string
  modelKind: 'base' | 'instruct' | 'embedding'
  status: ModelImportStatus
  progressCompleted: number
  progressTotal?: number
  progressPercent?: number
  createdAt: string
  updatedAt: string
  startedAt?: string
  finishedAt?: string
  requestedRevision?: string
  resolvedRevision?: string
  resultAssetId?: string
  manifest?: ModelManifestSummary
  errorMessage?: string
}

export interface Deployment {
  id: string
  name: string
  model: string
  modelAssetId: string
  serviceType: 'generation' | 'embedding'
  gpuIds: number[]
  gpuLabel: string
  parallelism: string
  status: 'running' | 'stopped' | 'queued' | 'error' | 'starting' | 'stopping'
  desiredState: 'running' | 'stopped'
  healthStatus: 'starting' | 'healthy' | 'unhealthy' | null
  startedAt?: string
  errorMessage?: string | null
  qps?: number
  ttft?: number
  kvHitRate?: number
  simplifiedConfig?: Record<string, unknown>
  vllmArgs?: Record<string, unknown>
  createdAt?: string
  updatedAt?: string
}

export interface TrainingJob {
  id: string
  name: string
  stage: 'CPT' | 'SFT'
  algorithm: 'LoRA' | 'QLoRA' | 'Freeze'
  baseModel: string
  gpuIds: number[]
  gpuLabel: string
  progress: number
  status: 'running' | 'queued' | 'completed' | 'failed' | 'stopping' | 'terminated'
  step: number
  totalSteps: number
  epoch: string
  eta?: string
  metrics?: Record<string, unknown>
  outputDir?: string
  checkpointPath?: string
  adapterPath?: string
  mergedModelPath?: string
  publishedModelAssetId?: string
  updatedAt?: string
}

export interface Dataset {
  id: string
  name: string
  version: string
  purpose: 'CPT' | 'SFT' | 'Evaluation'
  format: 'JSONL'
  samples: number
  tokens: number
  status: 'available' | 'validating' | 'failed'
  updatedAt: string
  fileName?: string
  size?: string
  sha256?: string
  validationErrors?: Array<Record<string, unknown>>
  schemaSummary?: Record<string, unknown>
}

export interface EvaluationSummary {
  dataset: string
  samples: number
  beforeCorrect: number | null
  beforeTotal: number | null
  beforeInvalid: number | null
  afterCorrect: number | null
  afterTotal: number | null
  afterInvalid: number | null
  before: number
  after: number
  pointChange: number
  // 基线为 0 时相对变化没有数学定义，不能伪装成 0%。
  relativeChange: number | null
}

export interface EvaluationCategorySummary extends EvaluationSummary {
  category: string
}

export interface EvaluationModelMetric {
  template: 'base' | 'instruct'
  score: number
  total: number
  correct: number
  invalid: number
  averageLatencyMs: number
}

export interface EvaluationComparisonScore {
  before: number
  after: number
  pointChange: number
  relativeChange: number | null
}

export interface EvaluationRunSummary {
  id: string
  name: string
  baseModel: string
  candidateModel: string
  // 保留 Dashboard 现有消费者使用的候选模型别名。
  model: string
  datasetNames: string[]
  datasets: string
  progress: number
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopping' | 'terminated'
  hasResult: boolean
  gpuIds: number[]
  warnings: string[]
  errorMessage?: string
  startedAt?: string
  finishedAt?: string
  updatedAt: string
}

export interface EvaluationRunDetail extends EvaluationRunSummary {
  baseTemplate: 'base' | 'instruct'
  candidateTemplate: 'base' | 'instruct'
  baselineMetric?: EvaluationModelMetric
  candidateMetric?: EvaluationModelMetric
  overall?: EvaluationComparisonScore
  results: EvaluationSummary[]
  categories: EvaluationCategorySummary[]
  tensorParallelSize: number
  concurrency: number
  maxTokens: number
  gpuMemoryUtilization: number
}

export interface DashboardSummary {
  modelCount: number
  runningDeployments: number
  runningTrainingJobs: number
  availableGpus: number
  totalGpus: number
}

export interface DashboardActivity {
  id: string
  time: string
  text: string
  detail: string
  tone: StatusTone
}

export interface ApiKeySummary {
  id: string
  name: string
  prefix: string
  active: boolean
  createdAt: string
  lastUsedAt: string
}

export interface CreatedApiKey extends ApiKeySummary {
  key: string
}

export interface AdminIdentity {
  username: string
  authMethod: 'session' | 'bootstrap_key' | 'disabled'
  expiresAt?: string
}

export interface ChatMessage {
  id: string
  role: 'system' | 'user' | 'assistant'
  content: string
  createdAt?: string
}

export interface PlaygroundParams {
  model: string
  temperature: number
  topP: number
  maxTokens: number
  repetitionPenalty: number
  seed?: number
  stream: boolean
}

/**
 * Playground 只展示客户端能够可靠观测到的指标。兼容服务没有返回 usage 时，
 * Token 数和依赖 Token 数计算的速度保持 null，避免用字符数冒充 Token 数。
 */
export interface PlaygroundMetrics {
  totalDurationMs: number
  ttftMs: number | null
  inputTokens: number | null
  outputTokens: number | null
  outputTokensPerSecond: number | null
}
