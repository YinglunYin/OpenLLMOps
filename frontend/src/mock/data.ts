import type {
  DashboardSummary,
  Dataset,
  Deployment,
  EvaluationSummary,
  GpuDevice,
  ModelAsset,
  TrainingJob,
} from '@/types/domain'

export const dashboardSummary: DashboardSummary = {
  modelCount: 12,
  runningDeployments: 2,
  runningTrainingJobs: 1,
  availableGpus: 1,
  totalGpus: 4,
}

export const gpuDevices: GpuDevice[] = [
  { index: 0, name: 'RTX 4090D', utilization: 32, memoryUsed: 7.6, memoryTotal: 24, temperature: 52, power: 98, powerLimit: 425, state: 'inference', task: 'chatglm3-6b 服务' },
  { index: 1, name: 'RTX 4090D', utilization: 78, memoryUsed: 18.3, memoryTotal: 24, temperature: 68, power: 264, powerLimit: 425, state: 'training', task: 'qwen2-7b 微调' },
  { index: 2, name: 'RTX 4090D', utilization: 15, memoryUsed: 3.2, memoryTotal: 24, temperature: 46, power: 82, powerLimit: 425, state: 'reserved', task: 'llama3-8b 测评' },
  { index: 3, name: 'RTX 4090D', utilization: 1, memoryUsed: 0.2, memoryTotal: 24, temperature: 38, power: 41, powerLimit: 425, state: 'idle' },
]

export const modelAssets: ModelAsset[] = [
  { id: 'm-001', name: 'ChineseLM-8B-Instruct', version: 'v1.0.0', type: 'generation', source: 'Hugging Face', format: 'Safetensors', size: '15.6 GB', status: 'available', updatedAt: '2024-05-20 10:35', contextLength: 32768, path: '/models/ChineseLM-8B-Instruct' },
  { id: 'm-002', name: 'ChineseLM-14B-Base', version: 'v1.0.0', type: 'generation', source: 'ModelScope', format: 'Safetensors', size: '28.7 GB', status: 'available', updatedAt: '2024-05-20 09:12', contextLength: 32768, path: '/models/ChineseLM-14B-Base' },
  { id: 'm-003', name: 'Qwen2-7B-Instruct', version: 'v2.0.1', type: 'generation', source: 'ModelScope', format: 'Safetensors', size: '13.4 GB', status: 'validating', updatedAt: '2024-05-20 08:58' },
  { id: 'm-004', name: 'Baichuan2-13B-Chat', version: 'v1.0.0', type: 'generation', source: 'Hugging Face', format: 'Safetensors', size: '25.1 GB', status: 'available', updatedAt: '2024-05-19 16:20' },
  { id: 'm-005', name: 'Mixtral-8x7B-Instruct-v0.1', version: 'v1.0.0', type: 'generation', source: 'SFTP', format: 'Safetensors', size: '46.8 GB', status: 'available', updatedAt: '2024-05-19 11:05' },
  { id: 'm-006', name: 'BGE-M3', version: 'v1.1.0', type: 'embedding', source: 'Hugging Face', format: 'Safetensors', size: '2.0 GB', status: 'available', updatedAt: '2024-05-18 15:42' },
  { id: 'm-007', name: 'Embed-ZH-v2', version: 'v2.0.0', type: 'embedding', source: 'ModelScope', format: 'Safetensors', size: '1.1 GB', status: 'available', updatedAt: '2024-05-18 18:03' },
  { id: 'm-008', name: 'Qwen1.5-7B-Chat', version: 'v1.1.0', type: 'generation', source: '受控目录', format: 'Safetensors', size: '14.4 GB', status: 'failed', updatedAt: '2024-05-17 22:47' },
]

export const deployments: Deployment[] = [
  { id: 'd-001', name: 'chatglm3-6b-generation', model: 'chatglm3-6b', serviceType: 'generation', gpuLabel: 'GPU 0', parallelism: '单卡', status: 'running', endpoint: 'http://10.0.0.10:8000', qps: 18.6, ttft: 186, kvHitRate: 92.3 },
  { id: 'd-002', name: 'bge-large-zh-embedding', model: 'bge-large-zh', serviceType: 'embedding', gpuLabel: 'GPU 1', parallelism: '单卡', status: 'stopped', endpoint: 'http://10.0.0.20:8000' },
  { id: 'd-003', name: 'qwen2-7b-wait', model: 'qwen2-7b', serviceType: 'generation', gpuLabel: '请求 2 卡', parallelism: 'PP ×2', status: 'queued' },
]

export const trainingJobs: TrainingJob[] = [
  { id: 't-001', name: 'sft-qlora-domain', stage: 'SFT', algorithm: 'QLoRA', baseModel: 'ChineseLM-8B', gpuLabel: 'GPU 0–3', progress: 63, status: 'running', step: 6330, totalSteps: 10000, epoch: '2 / 3', eta: '00:32:45' },
  { id: 't-002', name: 'cpt-lora-general', stage: 'CPT', algorithm: 'LoRA', baseModel: 'ChineseLM-8B', gpuLabel: '等待资源', progress: 0, status: 'queued', step: 0, totalSteps: 12000, epoch: '0 / 3' },
  { id: 't-003', name: 'sft-lora-instruct', stage: 'SFT', algorithm: 'LoRA', baseModel: 'ChineseLM-7B', gpuLabel: 'GPU 0', progress: 100, status: 'completed', step: 8000, totalSteps: 8000, epoch: '3 / 3' },
  { id: 't-004', name: 'sft-freeze-domain', stage: 'SFT', algorithm: 'Freeze', baseModel: 'ChineseLM-14B', gpuLabel: 'GPU 1–2', progress: 100, status: 'completed', step: 9600, totalSteps: 9600, epoch: '3 / 3' },
]

export const datasets: Dataset[] = [
  { id: 'ds-001', name: '通用领域百科问答数据集', version: 'v1.0.0', purpose: 'CPT', format: 'JSONL', samples: 250000, tokens: 128450000, status: 'available', updatedAt: '2024-05-20 10:22:11' },
  { id: 'ds-002', name: '金融领域指令跟随数据集', version: 'v2.1.0', purpose: 'SFT', format: 'JSONL', samples: 18423, tokens: 25781230, status: 'available', updatedAt: '2024-05-20 09:58:32' },
  { id: 'ds-003', name: '编程代码生成数据集', version: 'v1.2.0', purpose: 'SFT', format: 'JSONL', samples: 32105, tokens: 42990114, status: 'validating', updatedAt: '2024-05-20 09:35:18' },
  { id: 'ds-004', name: '医疗领域问答测评集', version: 'v1.0.0', purpose: 'Evaluation', format: 'JSONL', samples: 5000, tokens: 6230987, status: 'available', updatedAt: '2024-05-19 22:14:05' },
  { id: 'ds-005', name: '法律法规测评集', version: 'v1.0.0', purpose: 'Evaluation', format: 'JSONL', samples: 3200, tokens: 3890442, status: 'failed', updatedAt: '2024-05-19 18:07:26' },
  { id: 'ds-006', name: '客服多轮对话数据集', version: 'v1.1.0', purpose: 'SFT', format: 'JSONL', samples: 12560, tokens: 18225770, status: 'available', updatedAt: '2024-05-19 16:40:33' },
]

export const evaluationSummary: EvaluationSummary[] = [
  { dataset: 'C-Eval', samples: 5000, before: 62.4, after: 63.5, pointChange: 1.1, relativeChange: 1.76 },
  { dataset: 'CMMLU', samples: 5000, before: 58.1, after: 59.3, pointChange: 1.2, relativeChange: 2.07 },
  { dataset: '领域测试集', samples: 11800, before: 54.2, after: 71.8, pointChange: 17.6, relativeChange: 32.47 },
]

export const trendTimes = ['10:05', '10:15', '10:25', '10:35', '10:45', '10:55', '11:05']
export const utilizationSeries = [
  [31, 36, 35, 38, 35, 37, 34],
  [74, 78, 72, 76, 75, 79, 78],
  [14, 12, 15, 11, 13, 17, 15],
  [1, 0, 2, 1, 1, 0, 1],
]
