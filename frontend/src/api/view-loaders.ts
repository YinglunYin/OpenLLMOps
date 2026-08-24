import type { Deployment, GpuDevice } from '@/types/domain'

export type LatestLoadResult<T> =
  | { status: 'success'; value: T }
  | { status: 'error'; error: unknown }
  | { status: 'stale' }

/**
 * 为详情页保留“最后一次选择生效”语义。旧请求即使更晚返回，也不能覆盖管理员
 * 当前选择；当前请求失败则显式返回 error，页面据此清空旧详情。
 */
export function createLatestKeyedLoader<T>() {
  let requestVersion = 0

  return {
    async run(
      key: string,
      currentKey: () => string,
      request: () => Promise<T>,
    ): Promise<LatestLoadResult<T>> {
      const version = ++requestVersion
      try {
        const value = await request()
        if (version !== requestVersion || currentKey() !== key) return { status: 'stale' }
        return { status: 'success', value }
      } catch (error) {
        if (version !== requestVersion || currentKey() !== key) return { status: 'stale' }
        return { status: 'error', error }
      }
    },
    invalidate() {
      requestVersion += 1
    },
  }
}

export interface DeploymentRefreshResult {
  deployments?: Deployment[]
  gpus?: GpuDevice[]
  deploymentError?: unknown
  gpuError?: unknown
}

/** GPU 遥测是辅助信息；它失败时，部署实际状态仍必须独立刷新。 */
export async function loadDeploymentRefresh(
  loadDeployments: () => Promise<Deployment[]>,
  loadGpus: () => Promise<GpuDevice[]>,
): Promise<DeploymentRefreshResult> {
  const [deploymentResult, gpuResult] = await Promise.allSettled([
    loadDeployments(),
    loadGpus(),
  ])
  return {
    ...(deploymentResult.status === 'fulfilled'
      ? { deployments: deploymentResult.value }
      : { deploymentError: deploymentResult.reason }),
    ...(gpuResult.status === 'fulfilled'
      ? { gpus: gpuResult.value }
      : { gpuError: gpuResult.reason }),
  }
}
