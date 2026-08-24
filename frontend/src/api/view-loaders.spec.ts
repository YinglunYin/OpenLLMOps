import { describe, expect, it, vi } from 'vitest'

import { createLatestKeyedLoader, loadDeploymentRefresh } from './view-loaders'
import type { Deployment, GpuDevice } from '@/types/domain'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('详情请求竞态协调', () => {
  it('当前选择加载失败时返回 error，旧选择的迟到响应返回 stale', async () => {
    const loader = createLatestKeyedLoader<{ id: string }>()
    let selectedId = 'old'
    const oldRequest = deferred<{ id: string }>()
    const oldResult = loader.run('old', () => selectedId, () => oldRequest.promise)

    selectedId = 'new'
    const failure = new Error('详情不可用')
    const newResult = loader.run('new', () => selectedId, async () => { throw failure })
    oldRequest.resolve({ id: 'old' })

    await expect(newResult).resolves.toEqual({ status: 'error', error: failure })
    await expect(oldResult).resolves.toEqual({ status: 'stale' })
  })
})

describe('部署与 GPU 独立刷新', () => {
  it('GPU 加载失败时仍返回最新部署列表', async () => {
    const deployments = [{ id: 'deployment-new' }] as Deployment[]
    const gpuError = new Error('GPU 遥测暂不可用')

    const result = await loadDeploymentRefresh(
      vi.fn().mockResolvedValue(deployments),
      vi.fn().mockRejectedValue(gpuError),
    )

    expect(result.deployments).toBe(deployments)
    expect(result.gpus).toBeUndefined()
    expect(result.gpuError).toBe(gpuError)
    expect(result.deploymentError).toBeUndefined()
  })

  it('部署加载失败时仍保留可用的 GPU 刷新结果', async () => {
    const gpus = [{ index: 0 }] as GpuDevice[]
    const deploymentError = new Error('部署接口不可用')

    const result = await loadDeploymentRefresh(
      vi.fn().mockRejectedValue(deploymentError),
      vi.fn().mockResolvedValue(gpus),
    )

    expect(result.gpus).toBe(gpus)
    expect(result.deployments).toBeUndefined()
    expect(result.deploymentError).toBe(deploymentError)
  })
})
