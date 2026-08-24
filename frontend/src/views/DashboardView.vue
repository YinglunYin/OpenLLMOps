<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Box, Cpu, DataLine, Promotion, Upload } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import GpuCard from '@/components/GpuCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { DashboardActivity, DashboardSummary, GpuDevice, StatusTone } from '@/types/domain'

const emptySummary: DashboardSummary = { modelCount: 0, runningDeployments: 0, runningTrainingJobs: 0, availableGpus: 0, totalGpus: 0 }
const summary = ref<DashboardSummary>(emptySummary)
const gpus = ref<GpuDevice[]>([])
const trendTimes = ref<string[]>([])
const utilizationSeries = ref<number[][]>([])

const chartOption = computed(() => ({
  color: ['#12a865', '#1769f5', '#ed8a16', '#7c3aed'],
  tooltip: { trigger: 'axis' },
  legend: { top: 2, right: 6, itemWidth: 8, itemHeight: 8, textStyle: { color: '#657187', fontSize: 11 } },
  grid: { top: 38, left: 42, right: 16, bottom: 28 },
  xAxis: { type: 'category', boundaryGap: false, data: trendTimes.value, axisLine: { lineStyle: { color: '#dfe6ef' } }, axisLabel: { color: '#748094', fontSize: 11 } },
  yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%', color: '#748094', fontSize: 11 }, splitLine: { lineStyle: { color: '#edf1f6', type: 'dashed' } } },
  series: utilizationSeries.value.map((data, index) => ({
    name: `GPU ${index}`,
    type: 'line',
    data,
    smooth: true,
    showSymbol: false,
    lineStyle: { width: 2 },
  })),
}))

interface DashboardTask { type: string; name: string; target: string; status: string; tone: StatusTone; time: string }
const tasks = ref<DashboardTask[]>(useMocks ? [
  { type: '部署', name: 'chatglm3-6b 服务', target: 'GPU 0', status: '运行中', tone: 'success' as const, time: '11:00:12' },
  { type: '训练', name: 'qwen2-7b 微调', target: 'GPU 1', status: '运行中', tone: 'warning' as const, time: '10:58:41' },
  { type: '测评', name: 'llama3-8b 评测', target: 'GPU 2', status: '排队中', tone: 'primary' as const, time: '10:52:33' },
 ] : [])

const activities = ref<DashboardActivity[]>([])

onMounted(async () => {
  try {
    if (import.meta.env.VITE_USE_MOCKS === 'true') {
      const mockData = await import('@/mock/data')
      trendTimes.value = mockData.trendTimes
      utilizationSeries.value = mockData.utilizationSeries
    }
    const [summaryResult, gpuResult, activityResult] = await Promise.allSettled([api.dashboard.summary(), api.resources.gpus(), api.dashboard.activities()])
    if (summaryResult.status === 'fulfilled') summary.value = summaryResult.value
    if (gpuResult.status === 'fulfilled') {
      gpus.value = gpuResult.value
      if (!useMocks) {
        const histories = await api.resources.history('utilization', '1h', gpus.value.map((gpu) => gpu.index))
        utilizationSeries.value = histories.map((history) => history.points.map((point) => point.value))
        const timeline = histories.find((history) => history.points.length)?.points ?? []
        trendTimes.value = timeline.map((point) => new Date(point.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false }))
      }
    }
    if (activityResult.status === 'fulfilled') activities.value = activityResult.value
    if (summaryResult.status === 'rejected' && gpuResult.status === 'rejected') ElMessage.error('总览核心数据加载失败，请检查控制面连接')
    if (!useMocks) {
      const [deploymentResult, trainingResult, evaluationResult] = await Promise.allSettled([api.deployments.list(), api.training.list(), api.evaluations.list()])
      const deployments = deploymentResult.status === 'fulfilled' ? deploymentResult.value : []
      const trainingJobs = trainingResult.status === 'fulfilled' ? trainingResult.value : []
      const evaluationRuns = evaluationResult.status === 'fulfilled' ? evaluationResult.value : []
      const deploymentStatus = { running: ['运行中', 'success'], starting: ['启动中', 'primary'], stopping: ['停止中', 'info'], queued: ['等待 GPU', 'warning'], stopped: ['已停止', 'info'], error: ['异常', 'danger'] } as const
      const trainingStatus = { running: ['训练中', 'success'], queued: ['等待 GPU', 'warning'], completed: ['已完成', 'success'], failed: ['失败', 'danger'], stopping: ['终止中', 'info'], terminated: ['已终止', 'info'] } as const
      const evaluationStatus = { running: ['测评中', 'primary'], queued: ['等待 GPU', 'warning'], completed: ['已完成', 'success'], failed: ['失败', 'danger'], stopping: ['取消中', 'info'], terminated: ['已取消', 'info'] } as const
      tasks.value = [
        ...deployments.map((item) => ({ type: '部署', name: item.name, target: item.gpuLabel, status: deploymentStatus[item.status][0], tone: deploymentStatus[item.status][1], time: item.updatedAt?.slice(11) ?? '—' })),
        ...trainingJobs.map((item) => ({ type: '训练', name: item.name, target: item.gpuLabel, status: trainingStatus[item.status][0], tone: trainingStatus[item.status][1], time: item.updatedAt?.slice(11) ?? '—' })),
        ...evaluationRuns.map((item) => ({ type: '测评', name: item.name, target: '整卡队列', status: evaluationStatus[item.status][0], tone: evaluationStatus[item.status][1], time: item.updatedAt.slice(11) })),
      ].slice(0, 6)
    }
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '总览数据加载失败')
  }
})
</script>

<template>
  <div>
    <PageHeader title="总览" subtitle="单机多卡资源与大模型任务实时态势">
      <el-button :icon="Upload">导入模型</el-button>
      <el-button type="primary" :icon="Promotion">创建部署</el-button>
    </PageHeader>

    <div class="stats-grid dashboard-stats">
      <StatCard label="模型资产" :value="summary.modelCount" :icon="Box" tone="blue" hint="已纳管 Safetensors 模型" />
      <StatCard label="运行服务" :value="summary.runningDeployments" :icon="Promotion" tone="green" hint="OpenAI Compatible 接口" />
      <StatCard label="训练任务" :value="summary.runningTrainingJobs" :icon="DataLine" tone="orange" hint="非抢占式整卡调度" />
      <StatCard label="可用 GPU" :value="`${summary.availableGpus} / ${summary.totalGpus}`" :icon="Cpu" tone="purple" hint="整卡独占调度" />
    </div>

    <PanelCard title="GPU 资源" class="section-gap">
      <div class="gpu-grid"><GpuCard v-for="gpu in gpus" :key="gpu.index" :gpu="gpu" compact /></div>
      <el-empty v-if="!gpus.length" description="暂无 GPU 能力或租约数据" :image-size="64" />
    </PanelCard>

    <div class="dashboard-middle section-gap">
      <PanelCard title="GPU 利用率"><BaseChart v-if="utilizationSeries.some((series) => series.length)" :option="chartOption" height="213px" /><el-empty v-else description="最近一小时暂无 GPU 遥测" :image-size="62" /></PanelCard>
      <PanelCard title="任务队列" flush>
        <el-table :data="tasks" size="small">
          <el-table-column prop="type" label="类型" width="70">
            <template #default="{ row }"><StatusPill :text="row.type" :tone="row.type === '部署' ? 'primary' : row.type === '训练' ? 'warning' : 'info'" /></template>
          </el-table-column>
          <el-table-column prop="name" label="名称" min-width="150" show-overflow-tooltip />
          <el-table-column prop="target" label="目标" width="82" />
          <el-table-column label="状态" width="86"><template #default="{ row }"><StatusPill :text="row.status" :tone="row.tone" /></template></el-table-column>
          <el-table-column prop="time" label="更新时间" width="90" />
        </el-table>
      </PanelCard>
    </div>

    <PanelCard title="最近活动" class="section-gap">
      <div class="activity-list">
        <div v-for="activity in activities" :key="activity.id" class="activity-item">
          <i :class="`dot-${activity.tone}`" />
          <time>{{ activity.time }}</time>
          <strong>{{ activity.text }}</strong>
          <span>{{ activity.detail }}</span>
        </div>
      </div>
      <el-empty v-if="!activities.length" description="暂无最近审计活动" :image-size="62" />
    </PanelCard>
  </div>
</template>

<style scoped lang="scss">
.dashboard-stats { margin-bottom: 0; }
.dashboard-middle { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(480px, .95fr); gap: 14px; }
.activity-list { display: grid; gap: 11px; }
.activity-item { display: grid; grid-template-columns: 8px 72px 160px 1fr; align-items: center; gap: 12px; font-size: 13px; }
.activity-item > i { width: 8px; height: 8px; border-radius: 50%; }
.activity-item time, .activity-item span { color: #768195; }
.activity-item strong { font-weight: 560; }
.dot-success { background: #12a865; }.dot-warning { background: #ed8a16; }.dot-primary { background: #1769f5; }.dot-danger { background: #e5484d; }
@media (max-width: 1200px) { .dashboard-middle { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .activity-item { grid-template-columns: 8px 65px 1fr; }.activity-item span { grid-column: 3; }.activity-item strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } }
</style>
