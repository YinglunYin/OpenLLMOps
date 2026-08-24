<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Connection, Cpu, Files, Monitor, Refresh, Stopwatch } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import BaseChart from '@/components/BaseChart.vue'
import GpuCard from '@/components/GpuCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import PanelCard from '@/components/PanelCard.vue'
import StatCard from '@/components/StatCard.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api/services'
import { useMocks } from '@/api/client'
import type { GpuDevice } from '@/types/domain'

const gpus = ref<GpuDevice[]>([])
const trendTimes = ref<string[]>([])
const utilizationSeries = ref<number[][]>([])
const range = ref('1h')
const refreshedAt = ref(new Date())
const telemetryAvailable = computed(() => useMocks && gpus.value.some((gpu) => gpu.telemetryAvailable !== false))

const utilizationOption = computed(() => makeLineOption(utilizationSeries.value, '%'))
const memoryOption = computed(() => makeLineOption([[7.2,7.8,7.6,7.7,7.5,7.8,7.6],[17.2,18.2,18.5,18.1,18.4,18.1,18.3],[3,3.1,3.2,3,3.3,3.1,3.2],[.2,.2,.2,.2,.2,.2,.2]], ' GB', 24))
const thermalOption = computed(() => makeLineOption([[51,52,54,53,52,54,52],[65,68,69,68,70,69,68],[44,45,46,46,45,47,46],[37,38,38,39,38,38,38]], '°C', 100))

function makeLineOption(series: number[][], suffix: string, max = 100) {
  const colors = ['#12a865','#1769f5','#ed8a16','#7c3aed']
  return { color: colors, tooltip: { trigger: 'axis' }, legend: { top: 0, right: 5, itemWidth: 8, itemHeight: 8, textStyle: { fontSize: 10 } }, grid: { top: 36, left: 42, right: 12, bottom: 27 }, xAxis: { type: 'category', boundaryGap: false, data: trendTimes.value, axisLabel: { fontSize: 10, color: '#758195' }, axisLine: { lineStyle: { color: '#dfe6ef' } } }, yAxis: { type: 'value', min: 0, max, axisLabel: { fontSize: 10, color: '#758195', formatter: `{value}${suffix}` }, splitLine: { lineStyle: { color: '#edf1f5', type: 'dashed' } } }, series: series.map((data,index) => ({ name:`GPU ${index}`,type:'line',data,smooth:true,showSymbol:false,lineStyle:{width:2} })) }
}

async function refresh() {
  try {
    if (import.meta.env.VITE_USE_MOCKS === 'true' && !trendTimes.value.length) {
      const mockData = await import('@/mock/data')
      trendTimes.value = mockData.trendTimes
      utilizationSeries.value = mockData.utilizationSeries
    }
    gpus.value = await api.resources.gpus()
    refreshedAt.value = new Date()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'GPU 资源加载失败')
  }
}
onMounted(refresh)
</script>

<template>
  <div>
    <PageHeader title="资源监控" subtitle="NVIDIA GPU、主机资源、独占租约与硬件健康状态">
      <el-select v-model="range" style="width:145px"><el-option label="最近 1 小时" value="1h"/><el-option label="最近 6 小时" value="6h"/><el-option label="最近 24 小时" value="24h"/></el-select>
      <el-button :icon="Refresh" @click="refresh">5 秒刷新</el-button>
    </PageHeader>

    <div class="stats-grid five">
      <StatCard label="GPU" :value="gpus.length ? `${gpus.length} × RTX 4090D` : '—'" :icon="Cpu" tone="green" />
      <StatCard label="总显存" :value="gpus.length ? gpus.reduce((sum, gpu) => sum + gpu.memoryTotal, 0) : '—'" :suffix="gpus.length ? 'GB' : ''" :icon="Monitor" tone="blue" />
      <StatCard label="系统内存" :value="useMocks ? 128 : '—'" :suffix="useMocks ? 'GB' : ''" :icon="Cpu" tone="purple" />
      <StatCard label="磁盘" :value="useMocks ? 2 : '—'" :suffix="useMocks ? 'TB' : ''" :icon="Files" tone="orange" />
      <StatCard label="拓扑" value="PCIe" :icon="Connection" tone="slate" hint="无 NVLink" />
    </div>

    <div class="gpu-grid section-gap"><GpuCard v-for="gpu in gpus" :key="gpu.index" :gpu="gpu" /></div>
    <PanelCard v-if="!gpus.length" class="section-gap"><el-empty description="暂无 GPU 能力或租约数据" /></PanelCard>

    <div class="monitor-charts section-gap">
      <PanelCard title="GPU 利用率"><BaseChart v-if="telemetryAvailable" :option="utilizationOption" height="210px"/><el-empty v-else description="实时遥测端点尚未接入" :image-size="54"/></PanelCard>
      <PanelCard title="显存占用"><BaseChart v-if="telemetryAvailable" :option="memoryOption" height="210px"/><el-empty v-else description="实时遥测端点尚未接入" :image-size="54"/></PanelCard>
      <PanelCard title="温度趋势"><BaseChart v-if="telemetryAvailable" :option="thermalOption" height="210px"/><el-empty v-else description="实时遥测端点尚未接入" :image-size="54"/></PanelCard>
    </div>

    <div class="monitor-bottom section-gap">
      <PanelCard title="GPU 资源租约" flush>
        <el-table :data="gpus" size="small"><el-table-column label="GPU" width="70"><template #default="{row}">GPU {{ row.index }}</template></el-table-column><el-table-column prop="task" label="占用任务" min-width="180"><template #default="{row}">{{ row.task ?? '—' }}</template></el-table-column><el-table-column label="任务类型" width="95"><template #default="{row}"><StatusPill :text="row.state==='training'?'训练':row.state==='inference'?'推理':row.state==='reserved'?'预留':'空闲'" :tone="row.state==='training'?'warning':row.state==='inference'?'primary':row.state==='reserved'?'info':'success'"/></template></el-table-column><el-table-column label="显存" width="105"><template #default="{row}">{{ row.telemetryAvailable === false ? '未接入' : `${row.memoryUsed} / 24 GB` }}</template></el-table-column><el-table-column label="状态" width="90"><template #default="{row}"><StatusPill :text="row.state==='idle'?'空闲':'运行中'" :tone="row.state==='idle'?'info':'success'"/></template></el-table-column></el-table>
      </PanelCard>
      <PanelCard title="系统资源">
        <div v-if="useMocks" class="host-metrics"><div><span><el-icon><Cpu/></el-icon>CPU 利用率<b>28%</b></span><el-progress :percentage="28" :show-text="false" :stroke-width="7"/></div><div><span><el-icon><Monitor/></el-icon>系统内存<b>71.7 / 128 GB</b></span><el-progress :percentage="56" :show-text="false" :stroke-width="7"/></div><div><span><el-icon><Files/></el-icon>磁盘（/）<b>842 / 2000 GB</b></span><el-progress :percentage="42" :show-text="false" :stroke-width="7"/></div><div><span><el-icon><Connection/></el-icon>网络（eno1）<b>↑ 1.2 ↓ 1.5 Gbps</b></span><el-progress :percentage="52" :show-text="false" :stroke-width="7"/></div></div><el-empty v-else description="主机资源端点尚未接入" :image-size="54"/>
      </PanelCard>
      <PanelCard title="健康状态">
        <div v-if="useMocks" class="health-list"><div><span class="check">✓</span><span>ECC</span><b>0</b></div><div><span class="check">✓</span><span>XID</span><b>0</b></div><div><span class="check">✓</span><span>状态</span><b>正常</b></div><div class="refresh-time"><el-icon><Stopwatch/></el-icon>{{ refreshedAt.toLocaleTimeString('zh-CN',{hour12:false}) }}</div></div><el-empty v-else description="硬件健康端点尚未接入" :image-size="54"/>
      </PanelCard>
    </div>
  </div>
</template>

<style scoped lang="scss">
.monitor-charts{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.monitor-bottom{display:grid;grid-template-columns:1.65fr .75fr .45fr;gap:13px}.host-metrics{display:grid;gap:15px}.host-metrics span{display:flex;align-items:center;gap:7px;margin-bottom:6px;font-size:12px}.host-metrics b{margin-left:auto;font-weight:550}.health-list{display:grid;gap:9px}.health-list>div:not(.refresh-time){display:grid;grid-template-columns:24px 1fr auto;align-items:center;gap:8px;padding:11px;border:1px solid #e2e8ef;border-radius:6px;font-size:12px}.health-list b{font-weight:550}.check{width:20px;height:20px;display:grid;place-items:center;border-radius:50%;color:#fff;background:#12a865}.refresh-time{display:flex;align-items:center;gap:6px;color:#8792a3;font-size:10px}
@media(max-width:1300px){.monitor-charts{grid-template-columns:1fr 1fr}.monitor-charts>*:last-child{grid-column:1/-1}.monitor-bottom{grid-template-columns:1fr 1fr}.monitor-bottom>*:first-child{grid-column:1/-1}}@media(max-width:720px){.monitor-charts,.monitor-bottom{grid-template-columns:1fr}.monitor-charts>*:last-child,.monitor-bottom>*:first-child{grid-column:auto}}
</style>
