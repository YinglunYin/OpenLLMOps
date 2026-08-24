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
  state: 'idle' | 'inference' | 'training' | 'reserved'
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
  endpoint?: string
  qps?: number
  ttft?: number
  kvHitRate?: number
  simplifiedConfig?: Record<string, unknown>
  vllmArgs?: Record<string, unknown>
  updatedAt?: string
}

export interface TrainingJob {
  id: string
  name: string
  stage: 'CPT' | 'SFT'
  algorithm: 'LoRA' | 'QLoRA' | 'Freeze'
  baseModel: string
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
  before: number
  after: number
  pointChange: number
  relativeChange: number
}

export interface EvaluationRunSummary {
  id: string
  name: string
  model: string
  datasets: string
  progress: number
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopping' | 'terminated'
  updatedAt: string
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
