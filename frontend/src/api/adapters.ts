import type {
  BackendApiKey,
  BackendAuditLog,
  BackendDataset,
  BackendDeployment,
  BackendEvaluationRun,
  BackendGpuLease,
  BackendGpuStatus,
  BackendModelAsset,
  BackendModelImport,
  BackendTrainingJob,
} from './contracts'
import type {
  ApiKeySummary,
  DashboardActivity,
  Dataset,
  Deployment,
  EvaluationCategorySummary,
  EvaluationModelMetric,
  EvaluationRunDetail,
  EvaluationRunSummary,
  EvaluationSummary,
  GpuDevice,
  ModelAsset,
  ModelImportTask,
  ModelManifestSummary,
  TrainingJob,
} from '@/types/domain'

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return '—'
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`
  if (value > 0) return `${Math.round(value)} B`
  if (value === 0) return '0 B'
  return `${(value / 1024 ** 2).toFixed(1)} MB`
}

export function formatBytesPerSecond(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) || value < 0 ? '—' : `${formatBytes(value)}/s`
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
  const manifest = toModelManifest(item.metadata_json.manifest)
  const requestedRevision = stringFrom(item.metadata_json.requested_revision) ?? manifest?.requestedRevision
  const resolvedRevision = stringFrom(item.metadata_json.resolved_revision) ?? manifest?.resolvedRevision ?? item.revision ?? undefined
  return {
    id: item.id,
    name: item.name,
    version: stringFrom(item.metadata_json.version) ?? item.revision ?? '—',
    type: item.model_kind === 'embedding' ? 'embedding' : 'generation',
    source: sourceMap[item.source_type],
    format: 'Safetensors',
    size: formatBytes(item.size_bytes),
    status: item.status === 'ready' ? 'available' : item.status,
    updatedAt: formatDateTime(item.updated_at),
    contextLength: Number(item.metadata_json.context_length ?? 0) || undefined,
    path: item.local_path,
    sourceUri: item.source_uri ?? undefined,
    revision: item.revision ?? undefined,
    requestedRevision,
    resolvedRevision,
    family: item.family ?? manifest?.modelType,
    architecture: stringFrom(item.metadata_json.architecture) ?? manifest?.architecture,
    parameterCount: item.parameter_count ?? manifest?.parameterCount,
    weightDtypes: stringArrayFrom(item.metadata_json.weight_dtypes) ?? manifest?.weightDtypes,
    checksum: item.checksum ?? manifest?.checksum,
    errorMessage: item.error_message ?? undefined,
    manifest,
  }
}

/** 将导入器的 JSON 清单收窄为页面可安全使用的字段，异常字段直接忽略。 */
export function toModelManifest(value: unknown): ModelManifestSummary | undefined {
  if (!isRecord(value)) return undefined
  const files = Array.isArray(value.files)
    ? value.files.flatMap((entry) => {
      if (!isRecord(entry)) return []
      const path = stringFrom(entry.path)
      const sizeBytes = nonNegativeIntegerFrom(entry.size_bytes)
      const sha256 = sha256From(entry.sha256)
      return path && sizeBytes !== undefined && sha256 ? [{ path, sizeBytes, sha256 }] : []
    })
    : []
  const fileCount = nonNegativeIntegerFrom(value.file_count)
  return {
    modelType: stringFrom(value.model_type),
    architecture: stringFrom(value.architecture),
    totalSizeBytes: nonNegativeIntegerFrom(value.total_size_bytes),
    fileCount,
    parameterCount: nonNegativeIntegerFrom(value.parameter_count),
    weightDtypes: stringArrayFrom(value.weight_dtypes),
    checksum: sha256From(value.checksum),
    requestedRevision: stringFrom(value.requested_revision),
    resolvedRevision: stringFrom(value.resolved_revision),
    files,
  }
}

export function toModelImportTask(item: BackendModelImport): ModelImportTask {
  const sourceMap: Record<BackendModelImport['source'], ModelImportTask['source']> = {
    huggingface: 'Hugging Face',
    modelscope: 'ModelScope',
    controlled_directory: '受控目录',
  }
  const manifest = toModelManifest(item.manifest_json)
  return {
    id: item.id,
    name: item.name,
    source: sourceMap[item.source],
    sourceKey: item.source,
    repository: item.repository ?? undefined,
    sourceDirectory: item.source_directory ?? undefined,
    modelKind: item.model_kind,
    status: item.status,
    progressCompleted: item.progress_completed,
    progressTotal: item.progress_total ?? undefined,
    progressPercent: item.progress_percent ?? undefined,
    createdAt: item.created_at,
    updatedAt: item.updated_at,
    startedAt: item.started_at ?? undefined,
    finishedAt: item.finished_at ?? undefined,
    requestedRevision: item.revision ?? manifest?.requestedRevision,
    resolvedRevision: manifest?.resolvedRevision,
    resultAssetId: item.result_asset_id ?? undefined,
    manifest,
    errorMessage: item.error_message ?? undefined,
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
    desiredState: item.desired_state,
    healthStatus: item.health_status,
    startedAt: item.started_at ? formatDateTime(item.started_at) : undefined,
    errorMessage: item.error_message,
    qps: numberFrom(item.simplified_config.qps),
    ttft: numberFrom(item.simplified_config.ttft),
    kvHitRate: numberFrom(item.simplified_config.kv_hit_rate),
    simplifiedConfig: item.simplified_config,
    vllmArgs: item.vllm_args,
    createdAt: formatDateTime(item.created_at),
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
    version: stringFrom(item.schema_summary.version) ?? '未标注',
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
    gpuIds: item.gpu_ids,
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

export interface EvaluationNameContext {
  baseModelName?: string
  candidateModelName?: string
  customDatasetName?: string
}

const evaluationStateMap: Record<string, EvaluationRunSummary['status']> = {
  created: 'queued',
  queued: 'queued',
  starting: 'queued',
  running: 'running',
  canceling: 'stopping',
  canceled: 'terminated',
  succeeded: 'completed',
  failed: 'failed',
}

function evaluationDatasetName(source: string, customDatasetName?: string): string {
  if (source === 'ceval') return 'C-Eval'
  if (source === 'cmmlu') return 'CMMLU'
  if (source.startsWith('custom-')) return customDatasetName ?? '自定义领域集'
  return source || '未知数据集'
}

function splitEvaluationCategory(value: string, customDatasetName?: string): { dataset: string; category: string } {
  const [source = '', ...categoryParts] = value.split('/')
  return {
    dataset: evaluationDatasetName(source, customDatasetName),
    category: categoryParts.join('/') || '默认分类',
  }
}

function changeFromScores(before: number, after: number): { pointChange: number; relativeChange: number | null } {
  const pointChange = roundScore(after - before)
  return {
    pointChange,
    relativeChange: before === 0 ? null : roundScore(pointChange * 100 / before),
  }
}

/**
 * 将严格评测报告里的科目计数完整映射到页面，避免只显示百分比后无法判断样本质量。
 */
export function toEvaluationCategorySummaries(run: BackendEvaluationRun, customDatasetName?: string): EvaluationCategorySummary[] {
  const baseline = isRecord(run.metrics.baseline) ? run.metrics.baseline : undefined
  const candidateMetric = isRecord(run.metrics.candidate) ? run.metrics.candidate : undefined
  const baselineCategories = Array.isArray(baseline?.categories) ? baseline.categories.filter(isRecord) : []
  const candidateCategories = Array.isArray(candidateMetric?.categories) ? candidateMetric.categories.filter(isRecord) : []
  if (!baselineCategories.length || !candidateCategories.length) return []

  const candidateByCategory = new Map(candidateCategories.map((item) => [String(item.category), item]))
  return baselineCategories.flatMap((baselineCategory) => {
    const categoryName = String(baselineCategory.category ?? '')
    const candidateCategory = candidateByCategory.get(categoryName)
    const beforeTotal = numberFrom(baselineCategory.total)
    const beforeCorrect = numberFrom(baselineCategory.correct)
    const beforeInvalid = numberFrom(baselineCategory.invalid)
    const afterTotal = numberFrom(candidateCategory?.total)
    const afterCorrect = numberFrom(candidateCategory?.correct)
    const afterInvalid = numberFrom(candidateCategory?.invalid)
    if (
      !categoryName
      || beforeTotal === undefined
      || beforeCorrect === undefined
      || beforeInvalid === undefined
      || afterTotal === undefined
      || afterCorrect === undefined
      || afterInvalid === undefined
      || beforeTotal <= 0
      || afterTotal <= 0
    ) return []

    const before = roundScore(beforeCorrect * 100 / beforeTotal)
    const after = roundScore(afterCorrect * 100 / afterTotal)
    const change = changeFromScores(before, after)
    const labels = splitEvaluationCategory(categoryName, customDatasetName)
    return [{
      ...labels,
      samples: Math.max(beforeTotal, afterTotal),
      beforeCorrect,
      beforeTotal,
      beforeInvalid,
      afterCorrect,
      afterTotal,
      afterInvalid,
      before,
      after,
      ...change,
    }]
  })
}

export function toEvaluationSummaries(run: BackendEvaluationRun, customDatasetName?: string): EvaluationSummary[] {
  const categoryRows = toEvaluationCategorySummaries(run, customDatasetName)
  if (categoryRows.length) {
    const grouped = new Map<string, {
      beforeCorrect: number
      beforeTotal: number
      beforeInvalid: number
      afterCorrect: number
      afterTotal: number
      afterInvalid: number
    }>()
    for (const row of categoryRows) {
      const current = grouped.get(row.dataset) ?? {
        beforeCorrect: 0,
        beforeTotal: 0,
        beforeInvalid: 0,
        afterCorrect: 0,
        afterTotal: 0,
        afterInvalid: 0,
      }
      current.beforeCorrect += row.beforeCorrect ?? 0
      current.beforeTotal += row.beforeTotal ?? 0
      current.beforeInvalid += row.beforeInvalid ?? 0
      current.afterCorrect += row.afterCorrect ?? 0
      current.afterTotal += row.afterTotal ?? 0
      current.afterInvalid += row.afterInvalid ?? 0
      grouped.set(row.dataset, current)
    }
    const order = (name: string) => name === 'C-Eval' ? 0 : name === 'CMMLU' ? 1 : 2
    return [...grouped.entries()]
      .sort(([left], [right]) => order(left) - order(right) || left.localeCompare(right))
      .map(([dataset, value]) => {
        const before = roundScore(value.beforeCorrect * 100 / value.beforeTotal)
        const after = roundScore(value.afterCorrect * 100 / value.afterTotal)
        return {
          dataset,
          samples: Math.max(value.beforeTotal, value.afterTotal),
          ...value,
          before,
          after,
          ...changeFromScores(before, after),
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
    const pointChange = roundScore(numberFrom(value.point_change) ?? after - before)
    const samples = numberFrom(value.samples ?? value.sample_count) ?? 0
    return {
      dataset,
      samples,
      beforeCorrect: numberFrom(value.before_correct ?? value.base_correct) ?? null,
      beforeTotal: numberFrom(value.before_total ?? value.base_total) ?? (samples || null),
      beforeInvalid: numberFrom(value.before_invalid ?? value.base_invalid) ?? null,
      afterCorrect: numberFrom(value.after_correct ?? value.candidate_correct) ?? null,
      afterTotal: numberFrom(value.after_total ?? value.candidate_total) ?? (samples || null),
      afterInvalid: numberFrom(value.after_invalid ?? value.candidate_invalid) ?? null,
      before,
      after,
      pointChange,
      relativeChange: before === 0
        ? null
        : roundScore(numberFrom(value.relative_change ?? value.relative_change_percent) ?? (pointChange / before) * 100),
    }
  })
}

function roundScore(value: number): number {
  return Math.round(value * 10_000) / 10_000
}

export function toEvaluationRunSummary(run: BackendEvaluationRun, names: EvaluationNameContext = {}): EvaluationRunSummary {
  const datasets = [
    ...run.builtin_datasets.map((item) => ({ ceval: 'C-Eval', cmmlu: 'CMMLU' })[item] ?? item),
    ...(run.custom_dataset_id ? [names.customDatasetName ?? run.custom_dataset_id] : []),
  ]
  const candidateModel = names.candidateModelName ?? run.candidate_model_asset_id
  return {
    id: run.id,
    name: run.name,
    baseModel: names.baseModelName ?? run.base_model_asset_id,
    candidateModel,
    model: candidateModel,
    datasetNames: datasets,
    datasets: datasets.join(' / '),
    progress: Math.min(100, Math.max(0, Math.round(numberFrom(run.metrics.progress) ?? (run.actual_state === 'succeeded' ? 100 : 0)))),
    status: evaluationStateMap[run.actual_state] ?? 'queued',
    hasResult: Object.keys(run.comparison).length > 0,
    gpuIds: [...run.gpu_ids],
    warnings: [...run.warnings],
    errorMessage: run.error_message ?? undefined,
    startedAt: run.started_at ? formatDateTime(run.started_at) : undefined,
    finishedAt: run.finished_at ? formatDateTime(run.finished_at) : undefined,
    updatedAt: formatDateTime(run.updated_at),
  }
}

function toEvaluationModelMetric(value: unknown, fallbackTemplate: 'base' | 'instruct'): EvaluationModelMetric | undefined {
  if (!isRecord(value)) return undefined
  const total = numberFrom(value.total)
  const correct = numberFrom(value.correct)
  const invalid = numberFrom(value.invalid)
  const score = numberFrom(value.accuracy_percent)
  const averageLatencyMs = numberFrom(value.average_latency_ms)
  if ([total, correct, invalid, score, averageLatencyMs].some((item) => item === undefined)) return undefined
  return {
    template: value.template === 'base' || value.template === 'instruct' ? value.template : fallbackTemplate,
    total: total!,
    correct: correct!,
    invalid: invalid!,
    score: score!,
    averageLatencyMs: averageLatencyMs!,
  }
}

export function toEvaluationRunDetail(run: BackendEvaluationRun, names: EvaluationNameContext = {}): EvaluationRunDetail {
  const summary = toEvaluationRunSummary(run, names)
  const baselineMetric = toEvaluationModelMetric(run.metrics.baseline, run.base_template)
  const candidateMetric = toEvaluationModelMetric(run.metrics.candidate, run.candidate_template)
  const before = numberFrom(run.comparison.baseline_percent)
  const after = numberFrom(run.comparison.candidate_percent)
  const rawPointChange = numberFrom(run.comparison.percentage_point_change)
  const overall = before === undefined || after === undefined
    ? undefined
    : {
        before,
        after,
        pointChange: roundScore(rawPointChange ?? after - before),
        relativeChange: before === 0
          ? null
          : roundScore(numberFrom(run.comparison.relative_change_percent) ?? ((rawPointChange ?? after - before) * 100 / before)),
      }
  return {
    ...summary,
    baseTemplate: run.base_template,
    candidateTemplate: run.candidate_template,
    baselineMetric,
    candidateMetric,
    overall,
    results: toEvaluationSummaries(run, names.customDatasetName),
    categories: toEvaluationCategorySummaries(run, names.customDatasetName),
    tensorParallelSize: run.tensor_parallel_size,
    concurrency: run.concurrency,
    maxTokens: run.max_tokens,
    gpuMemoryUtilization: run.gpu_memory_utilization,
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
  const state: GpuDevice['state'] = item.owner_type === 'training'
    ? 'training'
    : item.owner_type === 'deployment'
      ? 'inference'
      : item.owner_type
        ? 'reserved'
        : item.resource_state === 'unmanaged'
          ? 'unmanaged'
          : item.resource_state === 'unknown'
            ? 'unknown'
            : 'idle'
  return {
    index: item.index,
    name: item.name ?? `NVIDIA GPU ${item.index}`,
    utilization: item.utilization_percent ?? 0,
    memoryUsed: item.memory_used_mib === null ? 0 : Number((item.memory_used_mib / 1024).toFixed(2)),
    memoryTotal: item.memory_total_mib === null ? 0 : Number((item.memory_total_mib / 1024).toFixed(2)),
    temperature: item.temperature_celsius ?? 0,
    power: item.power_watts ?? 0,
    powerLimit: 425,
    state,
    task: item.owner_name ?? (state === 'unmanaged' ? '检测到未纳管 GPU 活动' : undefined),
    telemetryAvailable: completeTelemetry,
    telemetryReason: item.degraded_reason ?? undefined,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringArrayFrom(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined
  const normalized = value.filter(
    (item): item is string => typeof item === 'string' && item.length > 0 && item.length <= 32,
  )
  return normalized.length === value.length ? normalized : undefined
}

function numberFrom(value: unknown): number | undefined {
  if (value === null || value === undefined || value === '') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

function nonNegativeNumberFrom(value: unknown): number | undefined {
  const number = numberFrom(value)
  return number !== undefined && number >= 0 ? number : undefined
}

function nonNegativeIntegerFrom(value: unknown): number | undefined {
  if (typeof value !== 'number') return undefined
  const number = nonNegativeNumberFrom(value)
  return number !== undefined && Number.isSafeInteger(number) ? number : undefined
}

function stringFrom(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function sha256From(value: unknown): string | undefined {
  const digest = stringFrom(value)
  return digest && /^[0-9a-f]{64}$/i.test(digest) ? digest.toLowerCase() : undefined
}
