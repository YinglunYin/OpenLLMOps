/** FastAPI 控制面使用 snake_case；这些类型只描述线上响应，不直接泄漏到页面。 */
export interface TimestampedResource {
  id: string
  created_at: string
  updated_at: string
}

export interface BackendModelAsset extends TimestampedResource {
  name: string
  source_type: 'huggingface' | 'modelscope' | 'sftp' | 'manual' | 'trained'
  source_uri: string | null
  revision: string | null
  local_path: string
  model_kind: 'base' | 'instruct' | 'embedding'
  format: string
  status: 'importing' | 'ready' | 'failed'
  family: string | null
  parameter_count: number | null
  size_bytes: number | null
  checksum: string | null
  error_message: string | null
  metadata_json: Record<string, unknown>
}

export interface BackendModelImport extends TimestampedResource {
  name: string
  source: 'huggingface' | 'modelscope' | 'controlled_directory'
  repository: string | null
  revision: string | null
  source_directory: string | null
  model_kind: 'base' | 'instruct' | 'embedding'
  status: 'pending' | 'transferring' | 'validating' | 'ready' | 'failed' | 'canceling' | 'canceled'
  progress_completed: number
  progress_total: number | null
  progress_percent: number | null
  started_at: string | null
  finished_at: string | null
  result_asset_id: string | null
  manifest_json: Record<string, unknown> | null
  error_message: string | null
}

export interface BackendInboxCandidate {
  name: string
  path: string
  file_count: number
  size_bytes: number
  ready_for_import: boolean
  reason: string | null
}

export interface BackendDeployment extends TimestampedResource {
  name: string
  served_model_name: string
  model_asset_id: string
  task_type: 'generate' | 'embedding'
  desired_state: 'running' | 'stopped'
  actual_state: 'created' | 'queued' | 'starting' | 'running' | 'stopping' | 'stopped' | 'failed'
  gpu_ids: number[]
  tensor_parallel_size: number
  port: number | null
  internal_url: string | null
  simplified_config: Record<string, unknown>
  vllm_args: Record<string, unknown>
  error_message: string | null
}

export interface BackendDataset extends TimestampedResource {
  name: string
  dataset_type: 'cpt' | 'sft' | 'evaluation'
  status: 'validating' | 'ready' | 'invalid'
  file_name: string
  local_path: string
  record_count: number | null
  size_bytes: number | null
  sha256: string | null
  schema_summary: Record<string, unknown>
  validation_errors: Array<Record<string, unknown>>
  description: string | null
}

export interface BackendTrainingJob extends TimestampedResource {
  name: string
  model_asset_id: string
  dataset_id: string
  stage: 'cpt' | 'sft'
  algorithm: 'freeze' | 'lora' | 'qlora'
  desired_state: 'running' | 'terminated'
  actual_state: 'created' | 'queued' | 'starting' | 'running' | 'canceling' | 'canceled' | 'succeeded' | 'failed'
  gpu_ids: number[]
  progress: number
  current_step: number | null
  total_steps: number | null
  metrics: Record<string, unknown>
  training_config: Record<string, unknown>
  output_dir: string
  checkpoint_path: string | null
  adapter_path: string | null
  merged_model_path: string | null
  error_message: string | null
}

export interface BackendEvaluationRun extends TimestampedResource {
  name: string
  base_model_asset_id: string
  candidate_model_asset_id: string
  custom_dataset_id: string | null
  builtin_datasets: string[]
  actual_state: string
  gpu_ids: number[]
  metrics: Record<string, unknown>
  comparison: Record<string, unknown>
  error_message: string | null
}

export interface BackendGpuLease {
  id: string
  gpu_index: number
  owner_type: 'deployment' | 'training' | 'evaluation'
  owner_id: string
  owner_name: string
  acquired_at: string
}

export interface BackendGpuStatus {
  index: number
  name: string | null
  memory_total_mib: number | null
  memory_used_mib: number | null
  memory_free_mib: number | null
  utilization_percent: number | null
  temperature_celsius: number | null
  power_watts: number | null
  telemetry_available: boolean
  degraded_reason: string | null
  owner_type: 'deployment' | 'training' | 'evaluation' | null
  owner_id: string | null
  owner_name: string | null
  lease_expires_at: string | null
}

export type BackendGpuHistoryMetric = 'utilization' | 'memory_used_mib' | 'memory_free_mib' | 'temperature_celsius' | 'power_watts'

export interface BackendGpuHistory {
  gpu_index: number
  metric: BackendGpuHistoryMetric
  unit: string
  start: string
  end: string
  step_seconds: number
  telemetry_available: boolean
  degraded_reason: string | null
  points: Array<{ timestamp: string; value: number }>
}

interface BackendDashboardWorkloadSummary {
  total: number
  running: number
  queued: number
  failed: number
}

export interface BackendDashboardSummary {
  generated_at: string
  models: { total: number; ready: number; importing: number; failed: number }
  deployments: BackendDashboardWorkloadSummary
  training_jobs: BackendDashboardWorkloadSummary
  evaluation_runs: BackendDashboardWorkloadSummary
  queue: { total: number; deployments: number; training_jobs: number; evaluation_runs: number; model_imports: number }
  gpus: {
    total: number
    leased: number
    free: number
    leases: Array<{ gpu_index: number; owner_type: 'deployment' | 'training' | 'evaluation'; owner_id: string; owner_name: string; expires_at: string }>
  }
  recent_activity: Array<{
    resource_type: 'model_asset' | 'model_import' | 'deployment' | 'training_job' | 'evaluation_run'
    resource_id: string
    name: string
    status: string
    occurred_at: string
  }>
}

export interface BackendCapabilities {
  gpu_count: number
  gpu_policy: string
  model_format: string[]
  trust_remote_code: boolean
  openai_endpoints: string[]
}

export interface BackendApiKey extends TimestampedResource {
  name: string
  prefix: string
  is_active: boolean
  last_used_at: string | null
}

export interface BackendCreatedApiKey extends BackendApiKey {
  key: string
}

export interface BackendAdminIdentity {
  username: string
  auth_method: 'session' | 'bootstrap_key' | 'disabled'
  expires_at: string | null
  csrf_token: string | null
}

export interface BackendAuditLog {
  id: string
  request_id: string
  actor: string
  auth_method: string | null
  action: string
  method: string
  path: string
  status_code: number
  succeeded: boolean
  source_ip: string
  occurred_at: string
}
