<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  Box,
  ChatDotRound,
  Connection,
  DataAnalysis,
  DataLine,
  Files,
  Fold,
  House,
  Monitor,
  Operation,
  Promotion,
  Setting,
  User,
} from '@element-plus/icons-vue'

import { useAppStore } from '@/stores/app'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const appStore = useAppStore()
const auth = useAuthStore()

const menuItems = [
  { label: '总览', path: '/', icon: House },
  { label: '模型资产', path: '/models', icon: Box },
  { label: '模型部署', path: '/deployments', icon: Promotion },
  { label: '训练数据集', path: '/datasets', icon: Files },
  { label: '训练任务', path: '/training', icon: DataLine },
  { label: '模型测评', path: '/evaluations', icon: DataAnalysis },
  { label: 'Playground', path: '/playground', icon: ChatDotRound },
  { label: '资源监控', path: '/monitoring', icon: Monitor },
  { label: '系统设置', path: '/settings', icon: Setting },
]

const activePath = computed(() => route.path)

async function handleAdminCommand(command: string) {
  if (command !== 'logout') return
  try {
    await auth.logout()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '退出失败')
  } finally {
    await router.replace('/login')
  }
}
</script>

<template>
  <div class="app-shell" :class="{ 'is-collapsed': appStore.sidebarCollapsed }">
    <header class="topbar">
      <div class="brand" :class="{ compact: appStore.sidebarCollapsed }">
        <span class="brand-mark">O</span>
        <span class="brand-text">OpenLLMOps</span>
      </div>
      <button class="icon-button collapse-button" aria-label="切换侧边栏" @click="appStore.toggleSidebar">
        <el-icon :size="21"><Fold /></el-icon>
      </button>
      <div class="topbar-spacer" />
      <div class="topbar-item hide-on-mobile"><el-icon><Connection /></el-icon>{{ appStore.clusterName }}</div>
      <div class="topbar-item hide-on-mobile"><el-icon><Operation /></el-icon>{{ appStore.networkLabel }}</div>
      <el-divider direction="vertical" class="hide-on-mobile" />
      <el-dropdown trigger="click" @command="handleAdminCommand">
        <button class="admin-button"><el-icon><User /></el-icon><span>{{ auth.identity?.username ?? '管理员' }}</span><span class="chevron">⌄</span></button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item disabled>单管理员会话</el-dropdown-item>
            <el-dropdown-item command="logout" divided>安全退出</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </header>

    <aside class="sidebar" :class="{ 'mobile-visible': appStore.mobileSidebarVisible }">
      <nav aria-label="主导航">
        <RouterLink
          v-for="item in menuItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: activePath === item.path }"
          :title="appStore.sidebarCollapsed ? item.label : undefined"
          @click="appStore.closeMobileSidebar"
        >
          <el-icon :size="21"><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
      <div class="sidebar-footer">
        <el-icon><Monitor /></el-icon>
        <span>控制面运行正常</span>
        <i />
      </div>
    </aside>

    <div v-if="appStore.mobileSidebarVisible" class="sidebar-mask" @click="appStore.closeMobileSidebar" />

    <main class="main-content">
      <RouterView v-slot="{ Component }">
        <Transition name="page" mode="out-in">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </main>
  </div>
</template>
