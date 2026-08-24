<script setup lang="ts">
import { computed } from 'vue'
import type { StatusTone } from '@/types/domain'

const props = withDefaults(
  defineProps<{
    text: string
    tone?: StatusTone
    dot?: boolean
  }>(),
  { tone: 'info', dot: false },
)

const className = computed(() => [`status-pill`, `is-${props.tone}`, { 'with-dot': props.dot }])
</script>

<template>
  <span :class="className"><i v-if="dot" />{{ text }}</span>
</template>
