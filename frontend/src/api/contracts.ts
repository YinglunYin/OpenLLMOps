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
