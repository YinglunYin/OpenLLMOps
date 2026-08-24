import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', {
  state: () => ({
    sidebarCollapsed: false,
    mobileSidebarVisible: false,
    clusterName: '单机集群',
    networkLabel: '内网',
  }),
  actions: {
    toggleSidebar() {
      if (window.innerWidth <= 900) {
        this.mobileSidebarVisible = !this.mobileSidebarVisible
        return
      }
      this.sidebarCollapsed = !this.sidebarCollapsed
    },
    closeMobileSidebar() {
      this.mobileSidebarVisible = false
    },
  },
})
