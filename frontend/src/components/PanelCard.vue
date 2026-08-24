<script setup lang="ts">
withDefaults(defineProps<{ title?: string; flush?: boolean }>(), { title: '', flush: false })
</script>

<template>
  <section class="panel-card" :class="{ flush }">
    <header v-if="title || $slots.actions" class="panel-header">
      <h2>{{ title }}</h2>
      <div><slot name="actions" /></div>
    </header>
    <div class="panel-body"><slot /></div>
  </section>
</template>
