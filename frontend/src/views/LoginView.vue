<script setup lang="ts">
import { reactive } from 'vue'
import { Lock, User } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const form = reactive({ username: 'admin', password: '' })

async function submit() {
  if (!form.username || !form.password) { ElMessage.warning('请输入管理员用户名和密码'); return }
  try {
    await auth.login(form.username, form.password)
    const redirect = typeof route.query.redirect === 'string' && route.query.redirect.startsWith('/') ? route.query.redirect : '/'
    await router.replace(redirect)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '登录失败')
  }
}
</script>

<template>
  <main class="login-page">
    <section class="login-card">
      <div class="login-brand"><span>O</span><div><h1>OpenLLMOps</h1><p>单机多卡大模型服务综合管理系统</p></div></div>
      <el-alert title="管理员会话登录" description="管理接口使用 HttpOnly Cookie 与内存 CSRF 保护；推理接口继续使用独立 API Key。" type="info" :closable="false" show-icon />
      <el-form label-position="top" @submit.prevent="submit">
        <el-form-item label="管理员用户名"><el-input v-model="form.username" :prefix-icon="User" autocomplete="username" /></el-form-item>
        <el-form-item label="密码"><el-input v-model="form.password" type="password" :prefix-icon="Lock" show-password autocomplete="current-password" /></el-form-item>
        <el-button native-type="submit" type="primary" size="large" :loading="auth.loading">登录控制台</el-button>
      </el-form>
      <p class="login-footnote">凭证不会写入浏览器持久存储</p>
    </section>
  </main>
</template>

<style scoped lang="scss">
.login-page{min-height:100vh;display:grid;place-items:center;padding:24px;background:radial-gradient(circle at 20% 15%,#e8f2ff 0,transparent 34%),linear-gradient(145deg,#f7faff,#eef3f9)}.login-card{width:min(440px,100%);padding:32px;border:1px solid #dce4ef;border-radius:14px;background:#fff;box-shadow:0 22px 60px #25456e18}.login-brand{display:flex;align-items:center;gap:14px;margin-bottom:24px}.login-brand>span{width:48px;height:48px;display:grid;place-items:center;border-radius:12px;color:#fff;background:#1769f5;font-size:25px;font-weight:750}.login-brand h1{margin:0;color:#172133;font-size:24px}.login-brand p{margin:5px 0 0;color:#778398;font-size:12px}.el-alert{margin-bottom:22px}.el-button{width:100%;margin-top:4px}.login-footnote{margin:18px 0 0;color:#8a95a7;font-size:11px;text-align:center}
</style>
