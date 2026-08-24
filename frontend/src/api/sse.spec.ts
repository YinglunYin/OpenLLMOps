import { describe, expect, it } from 'vitest'

import { createOpenAIStreamParser } from './sse'

describe('OpenAI SSE 增量解析器', () => {
  it('能够处理 data 与 JSON 被任意网络分片切断的情况', () => {
    const tokens: string[] = []
    const parser = createOpenAIStreamParser((token) => tokens.push(token))
    parser.push('da')
    parser.push('ta: {"choices":[{"delta":{"content":"你')
    parser.push('好"}}]}\n\ndata: {"choices":[{"delta":{"content":"！"}}]}\n')
    parser.push('\ndata: [DO')
    parser.push('NE]\n\n')
    parser.finish()
    expect(tokens).toEqual(['你好', '！'])
  })

  it('兼容 completions 的 choices[].text 并忽略注释与空事件', () => {
    const tokens: string[] = []
    const parser = createOpenAIStreamParser((token) => tokens.push(token))
    parser.push(': keep-alive\n\ndata:\n\ndata: {"choices":[{"text":"A"}]}\r\n')
    parser.finish()
    expect(tokens).toEqual(['A'])
  })
})
