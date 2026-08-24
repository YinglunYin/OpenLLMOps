import { describe, expect, it } from 'vitest'

import { parseAdvancedArgs } from './services'

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
})
