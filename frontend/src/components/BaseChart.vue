<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'

echarts.use([LineChart, BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const props = withDefaults(defineProps<{ option: EChartsCoreOption; height?: string }>(), { height: '220px' })
const host = ref<HTMLDivElement>()
const chart = shallowRef<EChartsType>()
let observer: ResizeObserver | undefined

const render = () => chart.value?.setOption(props.option, { notMerge: true })

onMounted(() => {
  if (!host.value) return
  chart.value = echarts.init(host.value)
  render()
  // 页面侧栏折叠与响应式布局都会改变容器宽度，观察容器比只监听 window 更可靠。
  observer = new ResizeObserver(() => chart.value?.resize())
  observer.observe(host.value)
})

watch(() => props.option, render, { deep: true })

onBeforeUnmount(() => {
  observer?.disconnect()
  chart.value?.dispose()
})
</script>

<template>
  <div ref="host" class="base-chart" :style="{ height }" role="img" aria-label="数据趋势图" />
</template>
