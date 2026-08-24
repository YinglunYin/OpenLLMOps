export interface OpenAIStreamParser {
  push(chunk: string): void
  finish(): void
}

/**
 * 创建可增量消费的 OpenAI SSE 解析器。
 * 网络分片可能切在 `data:`、UTF-8 字符或 JSON 任意位置，因此解析器只处理完整行，
 * 并把未完成尾部保留到下一次 push。
 */
export function createOpenAIStreamParser(onToken: (token: string) => void): OpenAIStreamParser {
  let buffer = ''

  const consumeLine = (line: string) => {
    const normalized = line.replace(/\r$/, '').trim()
    if (!normalized.startsWith('data:')) return
    const payload = normalized.slice(5).trimStart()
    if (!payload || payload === '[DONE]') return
    const event = JSON.parse(payload) as { choices?: Array<{ delta?: { content?: string }; text?: string }> }
    const choice = event.choices?.[0]
    const token = choice?.delta?.content ?? choice?.text
    if (token) onToken(token)
  }

  return {
    push(chunk: string) {
      buffer += chunk
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      lines.forEach(consumeLine)
    },
    finish() {
      if (buffer) consumeLine(buffer)
      buffer = ''
    },
  }
}
