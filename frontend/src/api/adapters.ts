import type {
  BackendApiKey,
  BackendAuditLog,
  BackendDataset,
  BackendDeployment,
  BackendEvaluationRun,
  BackendGpuLease,
  BackendGpuStatus,
  BackendModelAsset,
  BackendTrainingJob,
} from './contracts'
import type { ApiKeySummary, DashboardActivity, Dataset, Deployment, EvaluationRunSummary, EvaluationSummary, GpuDevice, ModelAsset, TrainingJob } from '@/types/domain'

export function formatBytes(value: number | null): string {
  if (value === null) return '—'
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`
  return `${(value / 1024 ** 2).toFixed(1)} MB`
}

export function formatDateTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const pad = (part: number) => String(part).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

export function toModelAsset(item: BackendModelAsset): ModelAsset {
  const sourceMap: Record<BackendModelAsset['source_type'], ModelAsset['source']> = {
    huggingface: 'Hugging Face',
    modelscope: 'ModelScope',
    sftp: 'SFTP',
    manual: '受控目录',
    trained: '训练产物',
  }
  return {
    id: item.id,
    name: item.name,
    version: String(item.metadata_json.version ?? item.revision ?? '—'),
    type: item.model_kind === 'embedding' ? 'embedding' : 'generation',
    source: sourceMap[item.source_type],
    format: 'Safetensors',
    size: formatBytes(item.size_bytes),
    status: item.status === 'ready' ? 'available' : item.status,
    updatedAt: formatDateTime(item.updated_at),
    contextLength: Number(item.metadata_json.context_length ?? 0) || undefined,
    path: item.local_path,
  }
}

export function toDeployment(item: BackendDeployment): Deployment {
  const stateMap: Record<BackendDeployment['actual_state'], Deployment['status']> = {
    created: 'stopped',
    queued: 'queued',
    starting: 'starting',
    running: 'running',
    stopping: 'stopping',
    stopped: 'stopped',
    failed: 'error',
  }
  return {
    id: item.id,
    name: item.name,
    model: item.served_model_name,
    modelAssetId: item.model_asset_id,
    serviceType: item.task_type === 'generate' ? 'generation' : 'embedding',
    gpuIds: item.gpu_ids,
    gpuLabel: item.gpu_ids.length ? item.gpu_ids.map((gpu) => `GPU ${gpu}`).join('、') : '等待资源',
    parallelism: item.tensor_parallel_size > 1 ? `TP ×${item.tensor_parallel_size}` : '单卡',
    status: stateMap[item.actual_state],
    endpoint: item.internal_url ?? undefined,
    qps: numberFrom(item.simplified_config.qps),
    ttft: numberFrom(item.simplified_config.ttft),
    kvHitRate: numberFrom(item.simplified_config.kv_hit_rate),
    simplifiedConfig: item.simplified_config,
    vllmArgs: item.vllm_args,
    updatedAt: formatDateTime(item.updated_at),
  }
}

export function toDataset(item: BackendDataset): Dataset {
  const purposeMap: Record<BackendDataset['dataset_type'], Dataset['purpose']> = {
    cpt: 'CPT',
    sft: 'SFT',
    evaluation: 'Evaluation',
  }
  return {
    id: item.id,
    name: item.name,
    version: String(item.schema_summary.version ?? 'v1.0.0'),
    purpose: purposeMap[item.dataset_type],
    format: 'JSONL',
    samples: item.record_count ?? 0,
    tokens: numberFrom(item.schema_summary.token_count) ?? 0,
    status: item.status === 'ready' ? 'available' : item.status === 'invalid' ? 'failed' : 'validating',
    updatedAt: formatDateTime(item.updated_at),
    fileName: item.file_name,
    size: formatBytes(item.size_bytes),
    sha256: item.sha256 ?? undefined,
    validationErrors: item.validation_errors,
    schemaSummary: item.schema_summary,
  }
}

export function toTrainingJob(item: BackendTrainingJob, modelName?: string): TrainingJob {
  const stateMap: Record<BackendTrainingJob['actual_state'], TrainingJob['status']> = {
    created: 'queued',
    queued: 'queued',
    starting: 'queued',
    running: 'running',
    canceling: 'stopping',
    canceled: 'terminated',
    succeeded: 'completed',
    failed: 'failed',
  }
  return {
    id: item.id,
    name: item.name,
    stage: item.stage.toUpperCase() as TrainingJob['stage'],
    algorithm: ({ freeze: 'Freeze', lora: 'LoRA', qlora: 'QLoRA' } as const)[item.algorithm],
    baseModel: modelName ?? item.model_asset_id,
    gpuLabel: item.gpu_ids.length ? item.gpu_ids.map((gpu) => `GPU ${gpu}`).join('、') : '等待资源',
    progress: Math.round(item.progress),
    status: stateMap[item.actual_state],
    step: item.current_step ?? 0,
    totalSteps: item.total_steps ?? 0,
    epoch: String(item.metrics.epoch ?? '—'),
    eta: typeof item.metrics.eta === 'string' ? item.metrics.eta : undefined,
    metrics: item.metrics,
    outputDir: item.output_dir,
    checkpointPath: item.checkpoint_path ?? undefined,
    adapterPath: item.adapter_path ?? undefined,
    mergedModelPath: item.merged_model_path ?? undefined,
    publishedModelAssetId: item.published_model_asset_id ?? undefined,
    updatedAt: formatDateTime(item.updated_at),
  }
}

export function toApiKey(item: BackendApiKey): ApiKeySummary {
  return {
    id: item.id,
    name: item.name,
    prefix: item.prefix,
    active: item.is_active,
    createdAt: formatDateTime(item.created_at),
    lastUsedAt: item.last_used_at ? formatDateTime(item.last_used_at) : '从未使用',
  }
}

export function toDashboardActivity(item: BackendAuditLog): DashboardActivity {
  const operation = item.action.includes('.') ? item.action.split('.').at(-1) : item.action
  return {
    id: item.id,
    time: formatDateTime(item.occurred_at).slice(11),
    text: operation || `${item.method} 请求`,
    detail: `${item.method} ${item.path} · HTTP ${item.status_code}`,
    tone: item.succeeded ? 'success' : 'danger',
  }
}

export function toEvaluationSummaries(run: BackendEvaluationRun): EvaluationSummary[] {
  const baseline = isRecord(run.metrics.baseline) ? run.metrics.baseline : undefined
  const candidateMetric = isRecord(run.metrics.candidate) ? run.metrics.candidate : undefined
  const baselineCategories = Array.isArray(baseline?.categories) ? baseline.categories.filter(isRecord) : []
  const candidateCategories = Array.isArray(candidateMetric?.categories) ? candidateMetric.categories.filter(isRecord) : []
  if (baselineCategories.length && candidateCategories.length) {
    const candidateByCategory = new Map(candidateCategories.map((item) => [String(item.category), item]))
    const grouped = new Map<string, { total: number; baselineCorrect: number; candidateCorrect: number }>()
    for (const category of baselineCategories) {
      const categoryName = String(category.category ?? '')
      const candidateCategory = candidateByCategory.get(categoryName)
      const total = numberFrom(category.total)
      const baselineCorrect = numberFrom(category.correct)
      const candidateCorrect = numberFrom(candidateCategory?.correct)
      if (!categoryName || total === undefined || baselineCorrect === undefined || candidateCorrect === undefined) continue
      const source = categoryName.split('/', 1)[0] || 'unknown'
      const current = grouped.get(source) ?? { total: 0, baselineCorrect: 0, candidateCorrect: 0 }
      current.total += total
      current.baselineCorrect += baselineCorrect
      current.candidateCorrect += candidateCorrect
      grouped.set(source, current)
    }
    const order = (name: string) => name === 'ceval' ? 0 : name === 'cmmlu' ? 1 : 2
    return [...grouped.entries()].sort(([left], [right]) => order(left) - order(right) || left.localeCompare(right)).map(([source, value]) => {
      const before = roundScore(value.baselineCorrect * 100 / value.total)
      const after = roundScore(value.candidateCorrect * 100 / value.total)
      const pointChange = roundScore(after - before)
      return {
        dataset: source === 'ceval' ? 'C-Eval' : source === 'cmmlu' ? 'CMMLU' : '自定义领域集',
        samples: value.total,
        before,
        after,
        pointChange,
        relativeChange: before ? roundScore(pointChange * 100 / before) : 0,
      }
    })
  }

  const candidate = run.comparison.results ?? run.comparison.datasets ?? run.comparison.summary
  const entries: Array<[string, Record<string, unknown>]> = Array.isArray(candidate)
    ? candidate.map((value, index) => [String((value as Record<string, unknown>).dataset ?? index), value as Record<string, unknown>])
    : isRecord(candidate)
      ? Object.entries(candidate).filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]))
      : []

  return entries.map(([dataset, value]) => {
    const before = numberFrom(value.before ?? value.base_score) ?? 0
    const after = numberFrom(value.after ?? value.candidate_score) ?? 0
    const pointChange = numberFrom(value.point_change) ?? after - before
    return {
      dataset,
      samples: numberFrom(value.samples ?? value.sample_count) ?? 0,
      before,
      after,
      pointChange,
      relativeChange: numberFrom(value.relative_change) ?? (before ? (pointChange / before) * 100 : 0),
    }
  })
}

function roundScore(value: number): number {
  return Math.round(value * 10_000) / 10_000
}

export function toEvaluationRunSummary(run: BackendEvaluationRun, modelName?: string, customDatasetName?: string): EvaluationRunSummary {
  const statusMap: Record<string, EvaluationRunSummary['status']> = {
    created: 'queued', queued: 'queued', starting: 'queued', running: 'running',
    canceling: 'stopping', canceled: 'terminated', succeeded: 'completed', failed: 'failed',
  }
  const datasets = [
    ...run.builtin_datasets.map((item) => ({ ceval: 'C-Eval', cmmlu: 'CMMLU' })[item] ?? item),
    ...(run.custom_dataset_id ? [customDatasetName ?? run.custom_dataset_id] : []),
  ]
  return {
    id: run.id,
    name: run.name,
    model: modelName ?? run.candidate_model_asset_id,
    datasets: datasets.join(' / '),
    progress: Math.round(numberFrom(run.metrics.progress) ?? (run.actual_state === 'succeeded' ? 100 : 0)),
    status: statusMap[run.actual_state] ?? 'queued',
    updatedAt: formatDateTime(run.updated_at),
  }
}

export function toGpuDevices(gpuCount: number, leases: BackendGpuLease[]): GpuDevice[] {
  return Array.from({ length: gpuCount }, (_, index) => {
    const lease = leases.find((item) => item.gpu_index === index)
    return {
      index,
      name: 'RTX 4090D',
      utilization: 0,
      memoryUsed: 0,
      memoryTotal: 24,
      temperature: 0,
      power: 0,
      powerLimit: 425,
      state: lease?.owner_type === 'training' ? 'training' : lease?.owner_type === 'deployment' ? 'inference' : lease ? 'reserved' : 'idle',
      task: lease?.owner_name,
      telemetryAvailable: false,
    }
  })
}

export function toGpuDevice(item: BackendGpuStatus): GpuDevice {
  const completeTelemetry = item.telemetry_available
    && item.utilization_percent !== null
    && item.memory_used_mib !== null
    && item.memory_total_mib !== null
    && item.temperature_celsius !== null
    && item.power_watts !== null
  return {
    index: item.index,
    name: item.name ?? `NVIDIA GPU ${item.index}`,
    utilization: item.utilization_percent ?? 0,
    memoryUsed: item.memory_used_mib === null ? 0 : Number((item.memory_used_mib / 1024).toFixed(2)),
    memoryTotal: item.memory_total_mib === null ? 0 : Number((item.memory_total_mib / 1024).toFixed(2)),
    temperature: item.temperature_celsius ?? 0,
    power: item.power_watts ?? 0,
    powerLimit: 425,
    state: item.owner_type === 'training' ? 'training' : item.owner_type === 'deployment' ? 'inference' : item.owner_type ? 'reserved' : 'idle',
    task: item.owner_name ?? undefined,
    telemetryAvailable: completeTelemetry,
    telemetryReason: item.degraded_reason ?? undefined,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function numberFrom(value: unknown): number | undefined {
  if (value === null || value === undefined || value === '') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}
