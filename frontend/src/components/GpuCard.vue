<script setup lang="ts">
import { computed } from 'vue'
import type { GpuDevice } from '@/types/domain'

const props = withDefaults(defineProps<{ gpu: GpuDevice; compact?: boolean }>(), { compact: false })

const stateText = computed(() => ({ idle: '空闲', inference: '推理', training: '训练', reserved: '预留' })[props.gpu.state])
const stateClass = computed(() => `state-${props.gpu.state}`)
const memoryPercent = computed(() => props.gpu.memoryTotal > 0 ? Math.round((props.gpu.memoryUsed / props.gpu.memoryTotal) * 100) : 0)
</script>

<template>
  <article class="gpu-card" :class="{ compact }">
    <div class="gpu-title">
      <div><strong>GPU {{ gpu.index }}</strong><span v-if="!compact">{{ gpu.name }}</span></div>
      <span :class="['gpu-state', stateClass]"><i />{{ stateText }}</span>
    </div>
    <div v-if="gpu.telemetryAvailable === false" class="telemetry-unavailable">
      {{ gpu.telemetryReason ?? '当前仅有租约状态，实时遥测不可用' }}
    </div>
    <template v-else>
    <div class="metric-row"><span>利用率</span><b>{{ gpu.utilization }}%</b></div>
    <el-progress :percentage="gpu.utilization" :show-text="false" :stroke-width="6" color="#16a66a" />
    <div class="metric-row"><span>显存</span><b>{{ gpu.memoryUsed }} / {{ gpu.memoryTotal }} GB</b></div>
    <el-progress :percentage="memoryPercent" :show-text="false" :stroke-width="6" color="#1769f5" />
    <template v-if="!compact">
      <div class="metric-row"><span>温度</span><b>{{ gpu.temperature }}°C</b></div>
      <el-progress :percentage="gpu.temperature" :show-text="false" :stroke-width="6" color="#f39a20" />
      <div class="metric-row"><span>功耗</span><b>{{ gpu.power }} W</b></div>
      <el-progress :percentage="Math.round((gpu.power / gpu.powerLimit) * 100)" :show-text="false" :stroke-width="6" color="#16a66a" />
    </template>
    </template>
  </article>
</template>

<style scoped>
.telemetry-unavailable { min-height: 62px; display: grid; place-items: center; padding: 10px; border-radius: 6px; color: #7a8698; background: #f5f7fa; font-size: 11px; text-align: center; line-height: 1.5; }
</style>
