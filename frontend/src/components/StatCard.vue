<script setup lang="ts">
import type { Component } from 'vue'

withDefaults(
  defineProps<{
    label: string
    value: string | number
    icon: Component
    tone?: 'blue' | 'green' | 'orange' | 'purple' | 'red' | 'slate'
    suffix?: string
    hint?: string
  }>(),
  { tone: 'blue', suffix: '', hint: '' },
)
</script>

<template>
  <article class="stat-card">
    <div class="stat-icon" :class="`tone-${tone}`"><el-icon :size="25"><component :is="icon" /></el-icon></div>
    <div class="stat-content">
      <span class="stat-label">{{ label }}</span>
      <strong>{{ value }}<small v-if="suffix"> {{ suffix }}</small></strong>
      <span v-if="hint" class="stat-hint">{{ hint }}</span>
    </div>
  </article>
</template>
