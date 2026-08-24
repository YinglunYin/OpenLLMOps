import { afterEach, describe, expect, it, vi } from 'vitest'

import { http } from './client'
import { api, normalizeDeployment, normalizeGpuIds, normalizeTraining, parseAdvancedArgs, streamChat } from './services'
import type { ChatMessage, PlaygroundParams } from '@/types/domain'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('数据集大文件上传', () => {
  it('覆盖普通请求超时并透传取消与进度', async () => {
    const post = vi.spyOn(http, 'post').mockResolvedValue({ data: { id: 'dataset' } } as never)
    const controller = new AbortController()
    const progress = vi.fn()

    await api.datasets.upload(
      { name: 'large-sft', purpose: 'SFT', version: 'v2.0.0' },
      new File(['{}\n'], 'large.jsonl', { type: 'application/jsonl' }),
      { signal: controller.signal, onProgress: progress },
    )

    expect(post).toHaveBeenCalledOnce()
    const body = post.mock.calls[0]?.[1]
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('version')).toBe('v2.0.0')
    const config = post.mock.calls[0]?.[2]
    expect(config?.timeout).toBe(0)
    expect(config?.signal).toBe(controller.signal)
    expect(config?.onUploadProgress).toBeTypeOf('function')
    config?.onUploadProgress?.({ loaded: 5, total: 10 } as never)
    expect(progress).toHaveBeenCalledWith(50)
  })
})

describe('vLLM 详细参数解析', () => {
  it('保留布尔、数值、null 与字符串类型', () => {
    expect(parseAdvancedArgs([
      '--enable-prefix-caching',
      '--max-num-seqs 64',
      '--gpu-memory-utilization 0.9',
      '--enforce-eager false',
      '--kv-cache-dtype null',
      '--dtype bfloat16',
    ].join('\n'))).toEqual({
      enable_prefix_caching: true,
      max_num_seqs: 64,
      gpu_memory_utilization: 0.9,
      enforce_eager: false,
      kv_cache_dtype: null,
      dtype: 'bfloat16',
    })
  })

  it('拒绝重复或不带双横线的参数', () => {
    expect(() => parseAdvancedArgs('--max-num-seqs 8\n--max-num-seqs 16')).toThrow('参数重复')
    expect(() => parseAdvancedArgs('max-num-seqs 8')).toThrow('必须以 -- 开头')
  })

  it('简化配置按节点安全边界发送显存利用率', () => {
    expect(normalizeDeployment({
      name: 'safe-memory', modelAssetId: 'asset', serviceType: 'generation',
      gpuIds: [1, 3], maxModelLen: 32768, gpuMemoryUtilization: 0.98,
      dtype: 'auto', advancedArgs: '',
    })).toMatchObject({
      gpu_ids: [1, 3], tensor_parallel_size: 2,
      simplified_config: { max_model_len: 32768, gpu_memory_utilization: 0.98, dtype: 'auto' },
    })
  })
})

describe('训练参数合同', () => {
  it('保留管理员选择的非连续 GPU，不再固定从 GPU 0 开始', () => {
    expect(normalizeGpuIds({ gpuIds: [1, 3] })).toEqual([1, 3])
    expect(normalizeGpuIds({ gpuCount: 2 })).toEqual([0, 1])
  })

  it('只发送服务端允许的白名单字段且不接受浏览器输出路径', () => {
    const payload = normalizeTraining({
      name: 'finance-sft', modelAssetId: 'asset', datasetId: 'dataset', stage: 'SFT',
      algorithm: 'QLoRA', gpuIds: [1, 3], epochs: 3, learningRate: 0.0002, cutoffLen: 4096,
      batchSize: 2, gradientAccumulation: 8, loggingSteps: 5, saveSteps: 50,
      warmupRatio: 0.03, loraRank: 16, loraAlpha: 32, loraDropout: 0.05,
      freezeTrainableLayers: 2, maxSamples: null, seed: 42, template: 'qwen',
      outputDir: '/tmp/attacker-controlled',
    })

    expect(payload).toEqual({
      name: 'finance-sft', model_asset_id: 'asset', dataset_id: 'dataset', stage: 'sft',
      algorithm: 'qlora', gpu_ids: [1, 3], training_config: {
        num_train_epochs: 3, learning_rate: 0.0002, cutoff_len: 4096,
        per_device_train_batch_size: 2, gradient_accumulation_steps: 8,
        logging_steps: 5, save_steps: 50, warmup_ratio: 0.03, lora_rank: 16,
        lora_alpha: 32, lora_dropout: 0.05, freeze_trainable_layers: 2,
        seed: 42, template: 'qwen',
      },
    })
    expect(payload).not.toHaveProperty('output_dir')
  })
})

describe('测评任务详情', () => {
  it('按用户选择的任务 ID 获取详情，并保留基线为零时未定义的相对变化', async () => {
    const get = vi.spyOn(http, 'get').mockImplementation(async (url) => {
      if (url === '/v1/evaluation-runs/eval-zero') {
        return { data: {
          id: 'eval-zero', created_at: '2026-08-24T10:00:00Z', updated_at: '2026-08-24T11:00:00Z',
          name: 'zero-baseline', base_model_asset_id: 'base', candidate_model_asset_id: 'candidate', custom_dataset_id: null,
          builtin_datasets: ['ceval'], base_template: 'base', candidate_template: 'instruct', output_dir: '/evaluation/eval-zero',
          tensor_parallel_size: 1, gpu_memory_utilization: 0.9, concurrency: 4, max_tokens: 128, desired_state: 'running',
          actual_state: 'succeeded', gpu_ids: [0], metrics: {}, comparison: {
            baseline_percent: 0, candidate_percent: 20, percentage_point_change: 20, relative_change_percent: null,
            datasets: { 'C-Eval': { samples: 10, base_score: 0, candidate_score: 20 } },
          }, result_path: '/evaluation/eval-zero/pair-report.json', dataset_manifest_path: '/runtime/manifest.json', warnings: [],
          error_message: null, queued_at: null, state_version: 2, runtime_generation: 1,
          started_at: '2026-08-24T10:00:00Z', finished_at: '2026-08-24T11:00:00Z',
        } } as never
      }
      if (url === '/v1/model-assets' || url === '/v1/datasets') return { data: [] } as never
      throw new Error(`unexpected URL: ${url}`)
    })

    const detail = await api.evaluations.get('eval-zero')

    expect(get).toHaveBeenCalledWith('/v1/evaluation-runs/eval-zero')
    expect(detail.overall).toMatchObject({ before: 0, after: 20, pointChange: 20, relativeChange: null })
    expect(detail.results[0]?.relativeChange).toBeNull()
  })
})

const playgroundMessages: ChatMessage[] = [
  { id: 'system', role: 'system', content: '你是助手' },
  { id: 'user', role: 'user', content: '你好' },
]

function playgroundParams(stream: boolean): PlaygroundParams {
  return {
    model: 'qwen-test', temperature: 0.7, topP: 0.9, maxTokens: 128,
    repetitionPenalty: 1.05, seed: 42, stream,
  }
}

function stubPlaygroundGlobals(nowValues: number[]) {
  vi.stubGlobal('sessionStorage', { getItem: vi.fn(() => 'sk-test') })
  vi.stubGlobal('performance', { now: vi.fn(() => nowValues.shift() ?? 0) })
}

describe('Playground OpenAI Chat', () => {
  it('流式增量回填内容、解析 usage 并且不发送管理 Cookie', async () => {
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"你"}}]}\n\n'))
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\ndata: [DONE]\n\n'))
        controller.close()
      },
    })
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    stubPlaygroundGlobals([100, 150, 300])
    const tokens: string[] = []

    const metrics = await streamChat({
      messages: playgroundMessages,
      params: playgroundParams(true),
      signal: new AbortController().signal,
      onToken: (token) => tokens.push(token),
    })

    expect(tokens).toEqual(['你', '好'])
    expect(metrics).toEqual({
      totalDurationMs: 200,
      ttftMs: 50,
      inputTokens: 4,
      outputTokens: 2,
      outputTokensPerSecond: 2 / 0.15,
    })
    const options = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(options.credentials).toBe('omit')
    expect(options.headers).toMatchObject({ 'X-API-Key': 'sk-test' })
    expect(JSON.parse(String(options.body))).toMatchObject({
      stream: true,
      stream_options: { include_usage: true },
    })
  })

  it('非流式请求一次回填完整内容并返回标准 usage', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      choices: [{ message: { role: 'assistant', content: '一次返回的完整回答' } }],
      usage: { prompt_tokens: 9, completion_tokens: 5 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    vi.stubGlobal('fetch', fetchMock)
    stubPlaygroundGlobals([20, 140, 160])
    const tokens: string[] = []

    const metrics = await streamChat({
      messages: playgroundMessages,
      params: playgroundParams(false),
      signal: new AbortController().signal,
      onToken: (token) => tokens.push(token),
    })

    expect(tokens).toEqual(['一次返回的完整回答'])
    expect(metrics).toEqual({
      totalDurationMs: 140,
      ttftMs: 120,
      inputTokens: 9,
      outputTokens: 5,
      outputTokensPerSecond: null,
    })
    const options = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(options.credentials).toBe('omit')
    const request = JSON.parse(String(options.body)) as Record<string, unknown>
    expect(request.stream).toBe(false)
    expect(request).not.toHaveProperty('stream_options')
  })

  it('优先展示 OpenAI 根 error.message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { message: 'API Key 已失效', type: 'authentication_error' },
    }), { status: 401, headers: { 'Content-Type': 'application/json' } })))
    stubPlaygroundGlobals([0])

    await expect(streamChat({
      messages: playgroundMessages,
      params: playgroundParams(false),
      signal: new AbortController().signal,
      onToken: () => undefined,
    })).rejects.toThrow('API Key 已失效')
  })

  it.each([true, false])('stream=%s 时都把取消可靠传给 fetch', async (stream) => {
    vi.stubGlobal('sessionStorage', { getItem: vi.fn(() => null) })
    vi.stubGlobal('fetch', vi.fn((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
      const signal = options.signal as AbortSignal
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    })))
    const controller = new AbortController()
    const request = streamChat({
      messages: playgroundMessages,
      params: playgroundParams(stream),
      signal: controller.signal,
      onToken: () => undefined,
    })

    controller.abort()
    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
  })
})
