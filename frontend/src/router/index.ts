import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import AppLayout from '@/layouts/AppLayout.vue'
import { pinia } from '@/stores'
import { useAuthStore } from '@/stores/auth'

const routes: RouteRecordRaw[] = [
  { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue'), meta: { title: '管理员登录' } },
  {
    path: '/',
    component: AppLayout,
    children: [
      { path: '', name: 'dashboard', component: () => import('@/views/DashboardView.vue'), meta: { title: '总览' } },
      { path: 'models', name: 'models', component: () => import('@/views/ModelAssetsView.vue'), meta: { title: '模型资产' } },
      { path: 'deployments', name: 'deployments', component: () => import('@/views/DeploymentsView.vue'), meta: { title: '模型部署' } },
      { path: 'datasets', name: 'datasets', component: () => import('@/views/DatasetsView.vue'), meta: { title: '训练数据集' } },
      { path: 'training', name: 'training', component: () => import('@/views/TrainingView.vue'), meta: { title: '训练任务' } },
      { path: 'evaluations', name: 'evaluations', component: () => import('@/views/EvaluationView.vue'), meta: { title: '模型测评' } },
      { path: 'playground', name: 'playground', component: () => import('@/views/PlaygroundView.vue'), meta: { title: 'Playground' } },
      { path: 'monitoring', name: 'monitoring', component: () => import('@/views/MonitoringView.vue'), meta: { title: '资源监控' } },
      { path: 'settings', name: 'settings', component: () => import('@/views/SettingsView.vue'), meta: { title: '系统设置' } },
    ],
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach(async (to) => {
  const auth = useAuthStore(pinia)
  const authenticated = await auth.restore()
  if (to.name === 'login') return authenticated ? { path: '/' } : true
  if (authenticated) return true
  return { name: 'login', query: { redirect: to.fullPath } }
})

router.afterEach((to) => {
  document.title = `${String(to.meta.title ?? '控制台')} · OpenLLMOps`
})

export default router
