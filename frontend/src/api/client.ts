import axios from 'axios'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api'
let csrfToken: string | null = null

// Mock 必须显式开启；本地开发默认同样连接真实接口，避免开发态掩盖契约问题。
export const useMocks = import.meta.env.VITE_USE_MOCKS === 'true'

export const http = axios.create({
  baseURL: apiBaseUrl,
  timeout: 30_000,
  withCredentials: true,
})

export function setCsrfToken(value: string | null) {
  // CSRF 只保存在当前 JS 运行时；刷新后通过 /auth/me 重新恢复。
  csrfToken = value
}

http.interceptors.request.use((config) => {
  const method = config.method?.toUpperCase()
  if (csrfToken && method && ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    config.headers['X-CSRF-Token'] = csrfToken
  }
  return config
})

http.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 401 && window.location.pathname !== '/login') {
        setCsrfToken(null)
        const redirect = `${window.location.pathname}${window.location.search}${window.location.hash}`
        // 会话过期时硬导航到登录页，同时清空当前内存中的 CSRF，避免继续发送失效令牌。
        window.location.assign(`/login?redirect=${encodeURIComponent(redirect)}`)
      }
      const detail = error.response?.data?.detail
      if (typeof detail === 'string') return Promise.reject(new Error(detail))
      if (Array.isArray(detail)) return Promise.reject(new Error(detail.map((item) => item?.msg).filter(Boolean).join('；') || error.message))
    }
    return Promise.reject(error)
  },
)
