import { describe, expect, it } from 'vitest'

import { toApiKey, toDashboardActivity, toDataset, toDeployment, toEvaluationRunSummary, toEvaluationSummaries, toGpuDevices, toModelAsset, toTrainingJob } from './adapters'
import type {
  BackendApiKey,
  BackendAuditLog,
  BackendDataset,
  BackendDeployment,
  BackendEvaluationRun,
  BackendModelAsset,
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
      error_message: null, metadata_json: { version: 'v2.0.0', context_length: 32768 },
    }
    expect(toModelAsset(raw)).toMatchObject({
      name: 'Qwen-Test', version: 'v2.0.0', type: 'generation', source: 'Hugging Face',
      size: '15.0 GB', status: 'available', contextLength: 32768, path: '/models/qwen',
    })
  })

  it('映射部署状态、服务类型和多卡并行字段', () => {
    const raw: BackendDeployment = {
      ...timestamps, name: 'embed-service', served_model_name: 'bge-m3', model_asset_id: 'asset',
      task_type: 'embedding', desired_state: 'running', actual_state: 'queued', gpu_ids: [1, 2],
      tensor_parallel_size: 2, port: null, internal_url: null, simplified_config: { max_model_len: 8192 }, vllm_args: { enable_prefix_caching: true }, error_message: null,
    }
    expect(toDeployment(raw)).toMatchObject({ serviceType: 'embedding', status: 'queued', gpuLabel: 'GPU 1、GPU 2', parallelism: 'TP ×2', simplifiedConfig: { max_model_len: 8192 }, vllmArgs: { enable_prefix_caching: true } })
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
      checkpoint_path: '/checkpoints/job/checkpoint-427', adapter_path: '/checkpoints/job/adapter', merged_model_path: null, error_message: null,
    }
    expect(toTrainingJob(training, 'Qwen-Test')).toMatchObject({ stage: 'SFT', algorithm: 'QLoRA', status: 'terminated', progress: 43, baseModel: 'Qwen-Test', checkpointPath: '/checkpoints/job/checkpoint-427', adapterPath: '/checkpoints/job/adapter' })
  })

  it('把评测 comparison 和 GPU 租约转换为量化结果及无遥测设备', () => {
    const run = {
      ...timestamps, name: 'compare', base_model_asset_id: 'base', candidate_model_asset_id: 'candidate',
      custom_dataset_id: null, builtin_datasets: ['ceval'], actual_state: 'succeeded', gpu_ids: [0], metrics: {},
      comparison: { datasets: { 'C-Eval': { samples: 100, base_score: 60, candidate_score: 63 } } }, error_message: null,
    } satisfies BackendEvaluationRun
    expect(toEvaluationSummaries(run)[0]).toMatchObject({ dataset: 'C-Eval', samples: 100, before: 60, after: 63, pointChange: 3, relativeChange: 5 })
    expect(toEvaluationRunSummary(run, '领域模型')).toMatchObject({ name: 'compare', model: '领域模型', datasets: 'C-Eval', progress: 100, status: 'completed' })

    const devices = toGpuDevices(2, [{ id: 'lease', gpu_index: 1, owner_type: 'training', owner_id: 'job', owner_name: 'sft-job', acquired_at: timestamps.created_at }])
    expect(devices).toHaveLength(2)
    expect(devices[0]).toMatchObject({ state: 'idle', telemetryAvailable: false })
    expect(devices[1]).toMatchObject({ state: 'training', task: 'sft-job', telemetryAvailable: false })
  })

  it('将 API Key 的布尔状态与时间字段映射为视图模型', () => {
    const raw: BackendApiKey = { ...timestamps, created_at: '2026-08-24T10:00:00', name: 'playground', prefix: 'ollm_abc', is_active: true, last_used_at: null }
    expect(toApiKey(raw)).toMatchObject({ name: 'playground', prefix: 'ollm_abc', active: true, createdAt: '2026-08-24 10:00:00', lastUsedAt: '从未使用' })

    const audit: BackendAuditLog = { id: 'audit', request_id: 'request', actor: 'admin', auth_method: 'session', action: 'deployment.start', method: 'POST', path: '/api/v1/deployments/1/start', status_code: 202, succeeded: true, source_ip: '127.0.0.1', occurred_at: '2026-08-24T10:01:00' }
    expect(toDashboardActivity(audit)).toMatchObject({ text: 'start', detail: 'POST /api/v1/deployments/1/start · HTTP 202', tone: 'success' })
  })
})
