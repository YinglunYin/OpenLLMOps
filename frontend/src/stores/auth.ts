import { defineStore } from 'pinia'

import { api } from '@/api/services'
import type { AdminIdentity } from '@/types/domain'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    identity: null as AdminIdentity | null,
    restored: false,
    loading: false,
  }),
  actions: {
    async restore(): Promise<boolean> {
      if (this.restored) return Boolean(this.identity)
      this.loading = true
      try {
        this.identity = await api.auth.me()
        return true
      } catch {
        this.identity = null
        return false
      } finally {
        this.restored = true
        this.loading = false
      }
    },
    async login(username: string, password: string) {
      this.loading = true
      try {
        this.identity = await api.auth.login(username, password)
        this.restored = true
      } finally {
        this.loading = false
      }
    },
    async logout() {
      try { await api.auth.logout() }
      finally {
        this.identity = null
        this.restored = true
      }
    },
  },
})
