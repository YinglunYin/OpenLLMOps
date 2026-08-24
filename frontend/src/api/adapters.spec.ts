import { describe, expect, it } from 'vitest'

import { formatBytes, formatBytesPerSecond, formatDateTime, toApiKey, toDashboardActivity, toDataset, toDeployment, toEvaluationRunDetail, toEvaluationRunSummary, toEvaluationSummaries, toGpuDevice, toGpuDevices, toModelAsset, toModelImportTask, toModelManifest, toTrainingJob } from './adapters'
import type {
  BackendApiKey,
  BackendAuditLog,
  BackendDataset,
  BackendDeployment,
  BackendEvaluationRun,
  BackendGpuStatus,
  BackendModelAsset,
  BackendModelImport,
  BackendTrainingJob,
} from './contracts'

const timestamps = { id: '00000000-0000-0000-0000-000000000001', created_at: '2026-08-24T10:00:00Z', updated_at: '2026-08-24T11:00:00Z' }

describe('FastAPI 资源适配器', () => {
  it('将模型资产 snake_case 响应转换为页面字段', () => {
    const raw: BackendModelAsset = {
      ...timestamps,
      name: 'Qwen-Test', source_type: 'huggingface', source_uri: 'Qwen/Test', revision: 'abc123',
      local_path: '/models/qwen', model_kind: 'instruct', format: 'safetensors', status: 'ready',
      family: 'qwen', parameter_count: 7_000_000_000, size_bytes: 15 * 1024 ** 3, checksum: 'sha',
      error_message: null,
      metadata_json: {
        version: 'v2.0.0', context_length: 32768, architecture: 'Qwen2ForCausalLM',
        requested_revision: 'main', resolved_revision: 'abc123',
        manifest: {
          model_type: 'qwen2', architecture: 'Qwen2ForCausalLM', total_size_bytes: 15 * 1024 ** 3,
          file_count: 2, parameter_count: 7_000_000_000, weight_dtypes: ['BF16'], checksum: 'd'.repeat(64), requested_revision: 'main', resolved_revision: 'abc123',
          files: [{ path: 'model.safetensors', size_bytes: 15 * 1024 ** 3, sha256: 'a'.repeat(64) }],
        },
      },
    }
    expect(toModelAsset(raw)).toMatchObject({
      name: 'Qwen-Test', version: 'v2.0.0', type: 'generation', source: 'Hugging Face',
      size: '15.0 GB', status: 'available', contextLength: 32768, path: '/models/qwen', sourceUri: 'Qwen/Test',
      requestedRevision: 'main', resolvedRevision: 'abc123', architecture: 'Qwen2ForCausalLM', parameterCount: 7_000_000_000, weightDtypes: ['BF16'],
      manifest: { fileCount: 2, totalSizeBytes: 15 * 1024 ** 3, parameterCount: 7_000_000_000, weightDtypes: ['BF16'], checksum: 'd'.repeat(64), files: [{ path: 'model.safetensors', sha256: 'a'.repeat(64) }] },
    })
  })

  it('防御性解析导入 manifest，并保留字节进度与固定 revision', () => {
    const raw: BackendModelImport = {
      ...timestamps,
      name: 'Qwen-Online', source: 'huggingface', repository: 'Qwen/Test', revision: 'main', source_directory: null,
      model_kind: 'instruct', status: 'transferring', progress_completed: 2 * 1024 ** 3, progress_total: 4 * 1024 ** 3,
      progress_percent: 50, started_at: '2026-08-24T10:01:00Z', finished_at: null, result_asset_id: null,
      manifest_json: {
        model_type: 'qwen2', architecture: 'Qwen2ForCausalLM', total_size_bytes: 4 * 1024 ** 3, file_count: 2,
        requested_revision: 'main', resolved_revision: 'b'.repeat(40),
        files: [
          { path: 'config.json', size_bytes: 1024, sha256: 'c'.repeat(64) },
          { path: '../invalid', size_bytes: -1, sha256: '' },
        ],
      },
      error_message: null,
    }
    expect(toModelImportTask(raw)).toMatchObject({
      source: 'Hugging Face', sourceKey: 'huggingface', repository: 'Qwen/Test', requestedRevision: 'main',
      resolvedRevision: 'b'.repeat(40), progressCompleted: 2 * 1024 ** 3, progressTotal: 4 * 1024 ** 3,
      manifest: { modelType: 'qwen2', architecture: 'Qwen2ForCausalLM', fileCount: 2, files: [{ path: 'config.json' }] },
    })
    expect(toModelManifest({ files: 'invalid' })).toEqual({ files: [] })
    expect(formatBytes(1024)).toBe('1.0 KB')
    expect(formatBytesPerSecond(2 * 1024 ** 2)).toBe('2.0 MB/s')
  })

  it('映射部署状态、服务类型和多卡并行字段', () => {
    const raw: BackendDeployment = {
      ...timestamps, name: 'embed-service', served_model_name: 'bge-m3', model_asset_id: 'asset',
      task_type: 'embedding', desired_state: 'running', actual_state: 'queued', gpu_ids: [1, 2],
      tensor_parallel_size: 2, simplified_config: { max_model_len: 8192 }, vllm_args: { enable_prefix_caching: true },
      health_status: 'starting', started_at: '2026-08-24T10:30:00Z', error_message: '等待 readiness',
    }
    expect(toDeployment(raw)).toMatchObject({
      modelAssetId: 'asset', serviceType: 'embedding', gpuIds: [1, 2], status: 'queued',
      desiredState: 'running', healthStatus: 'starting', startedAt: formatDateTime(raw.started_at!),
      errorMessage: '等待 readiness', gpuLabel: 'GPU 1、GPU 2', parallelism: 'TP ×2',
      simplifiedConfig: { max_model_len: 8192 }, vllmArgs: { enable_prefix_caching: true },
    })
  })

  it('映射数据集校验信息与训练状态', () => {
    const dataset: BackendDataset = {
      ...timestamps, name: '金融 SFT', dataset_type: 'sft', status: 'invalid', file_name: 'data.jsonl',
      local_path: '/datasets/data.jsonl', record_count: 120, size_bytes: 1024, sha256: 'a'.repeat(64),
      schema_summary: { token_count: 6400, version: 'v1.2.0' }, validation_errors: [{ line: 3 }], description: null,
    }
    expect(toDataset(dataset)).toMatchObject({ purpose: 'SFT', status: 'failed', samples: 120, tokens: 6400, version: 'v1.2.0', fileName: 'data.jsonl' })

    const training: BackendTrainingJob = {
      ...timestamps, name: 'job', model_asset_id: 'asset', dataset_id: 'dataset', stage: 'sft', algorithm: 'qlora',
      desired_state: 'terminated', actual_state: 'canceled', gpu_ids: [0], progress: 42.7, current_step: 427,
      total_steps: 1000, metrics: { epoch: '1 / 3', eta: '00:12:30' }, training_config: {}, output_dir: '/checkpoints/job',
      checkpoint_path: '/checkpoints/job/checkpoint-427', adapter_path: '/checkpoints/job/adapter', merged_model_path: null,
      published_model_asset_id: 'published-asset', error_message: null,
    }
    expect(toTrainingJob(training, 'Qwen-Test')).toMatchObject({ stage: 'SFT', algorithm: 'QLoRA', status: 'terminated', progress: 43, baseModel: 'Qwen-Test', gpuIds: [0], checkpointPath: '/checkpoints/job/checkpoint-427', adapterPath: '/checkpoints/job/adapter', publishedModelAssetId: 'published-asset' })
  })

  it('把评测 comparison 和 GPU 租约转换为量化结果及无遥测设备', () => {
    const run = {
      ...timestamps, name: 'compare', base_model_asset_id: 'base', candidate_model_asset_id: 'candidate',
      custom_dataset_id: null, builtin_datasets: ['ceval'], base_template: 'base', candidate_template: 'instruct',
      output_dir: '/evaluation/1', tensor_parallel_size: 1, gpu_memory_utilization: 0.9, concurrency: 4, max_tokens: 128,
      desired_state: 'running', actual_state: 'succeeded', gpu_ids: [0], metrics: {},
      comparison: { datasets: { 'C-Eval': { samples: 100, base_score: 60, candidate_score: 63 } } },
      result_path: '/evaluation/1/pair-report.json', dataset_manifest_path: '/runtime/manifest.json', warnings: [], error_message: null,
      queued_at: null, state_version: 3, runtime_generation: 1, started_at: timestamps.created_at, finished_at: timestamps.updated_at,
    } satisfies BackendEvaluationRun
    expect(toEvaluationSummaries(run)[0]).toMatchObject({ dataset: 'C-Eval', samples: 100, before: 60, after: 63, pointChange: 3, relativeChange: 5 })
    expect(toEvaluationRunSummary(run, { baseModelName: '基础模型', candidateModelName: '领域模型' })).toMatchObject({ name: 'compare', baseModel: '基础模型', model: '领域模型', datasets: 'C-Eval', progress: 100, status: 'completed', hasResult: true })

    const realRun = {
      ...run,
      metrics: {
        baseline: { template: 'base', total: 100, correct: 50, invalid: 2, accuracy_percent: 50, average_latency_ms: 10, categories: [{ category: 'ceval/law', total: 40, correct: 20, invalid: 1 }, { category: 'ceval/math', total: 60, correct: 30, invalid: 1 }] },
        candidate: { template: 'instruct', total: 100, correct: 60, invalid: 1, accuracy_percent: 60, average_latency_ms: 12, categories: [{ category: 'ceval/law', total: 40, correct: 24, invalid: 0 }, { category: 'ceval/math', total: 60, correct: 36, invalid: 1 }] },
      },
      comparison: { baseline_percent: 50, candidate_percent: 60, percentage_point_change: 10, relative_change_percent: 20, category_changes: [] },
    } satisfies BackendEvaluationRun
    expect(toEvaluationSummaries(realRun)).toEqual([{ dataset: 'C-Eval', samples: 100, beforeCorrect: 50, beforeTotal: 100, beforeInvalid: 2, afterCorrect: 60, afterTotal: 100, afterInvalid: 1, before: 50, after: 60, pointChange: 10, relativeChange: 20 }])
    expect(toEvaluationRunDetail(realRun, { baseModelName: '基础模型', candidateModelName: '候选模型' })).toMatchObject({
      baseModel: '基础模型', candidateModel: '候选模型',
      baselineMetric: { total: 100, correct: 50, invalid: 2, averageLatencyMs: 10 },
      candidateMetric: { total: 100, correct: 60, invalid: 1, averageLatencyMs: 12 },
      overall: { before: 50, after: 60, pointChange: 10, relativeChange: 20 },
      categories: [
        { dataset: 'C-Eval', category: 'law', beforeCorrect: 20, afterCorrect: 24 },
        { dataset: 'C-Eval', category: 'math', beforeCorrect: 30, afterCorrect: 36 },
      ],
    })

    const zeroBaseline = {
      ...run,
      comparison: { datasets: { '零基线集': { samples: 10, base_score: 0, candidate_score: 20, relative_change: 0 } } },
    } satisfies BackendEvaluationRun
    expect(toEvaluationSummaries(zeroBaseline)[0]?.relativeChange).toBeNull()

    const devices = toGpuDevices(2, [{ id: 'lease', gpu_index: 1, owner_type: 'training', owner_id: 'job', owner_name: 'sft-job', acquired_at: timestamps.created_at }])
    expect(devices).toHaveLength(2)
    expect(devices[0]).toMatchObject({ state: 'idle', telemetryAvailable: false })
    expect(devices[1]).toMatchObject({ state: 'training', task: 'sft-job', telemetryAvailable: false })

    const status = {
      index: 0, name: 'NVIDIA RTX 4090 D', memory_total_mib: 24_576, memory_used_mib: 12_288,
      memory_free_mib: 12_288, utilization_percent: 51, temperature_celsius: 62, power_watts: 318,
      telemetry_available: true, degraded_reason: null, resource_state: 'leased', owner_type: 'deployment', owner_id: 'deploy',
      owner_name: 'qwen-chat', lease_expires_at: '2026-08-24T11:01:00Z',
    } satisfies BackendGpuStatus
    expect(toGpuDevice(status)).toMatchObject({ memoryTotal: 24, memoryUsed: 12, utilization: 51, state: 'inference', task: 'qwen-chat', telemetryAvailable: true })

    expect(toGpuDevice({ ...status, resource_state: 'unmanaged', owner_type: null, owner_id: null, owner_name: null, lease_expires_at: null })).toMatchObject({ state: 'unmanaged', task: '检测到未纳管 GPU 活动' })
  })

  it('将 API Key 的布尔状态与时间字段映射为视图模型', () => {
    const raw: BackendApiKey = { ...timestamps, created_at: '2026-08-24T10:00:00', name: 'playground', prefix: 'ollm_abc', is_active: true, last_used_at: null }
    expect(toApiKey(raw)).toMatchObject({ name: 'playground', prefix: 'ollm_abc', active: true, createdAt: '2026-08-24 10:00:00', lastUsedAt: '从未使用' })

    const audit: BackendAuditLog = { id: 'audit', request_id: 'request', actor: 'admin', auth_method: 'session', action: 'deployment.start', method: 'POST', path: '/api/v1/deployments/1/start', status_code: 202, succeeded: true, source_ip: '127.0.0.1', occurred_at: '2026-08-24T10:01:00' }
    expect(toDashboardActivity(audit)).toMatchObject({ text: 'start', detail: 'POST /api/v1/deployments/1/start · HTTP 202', tone: 'success' })
  })
})
