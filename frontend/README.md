# OpenLLMOps Web

Vue 3 + TypeScript + Vite + Element Plus 构建的单机多卡大模型运维控制台。

## 本地开发

```bash
cp .env.example .env.local
npm install
npm run dev
```

默认连接真实 FastAPI 控制面，并使用管理员 Cookie + 内存 CSRF 会话。只有显式设置 `VITE_USE_MOCKS=true` 时才加载演示数据；`VITE_API_BASE_URL` 默认使用同域 `/api`。

## 生产构建

```bash
npm run typecheck
npm run test:unit
npm run build
docker build -t openllmops-web .
```

API Key 仅保存在 `sessionStorage`，浏览器关闭后自动清除。在线仓库 Token、SFTP 密钥等服务端凭证不会由前端持久化。
