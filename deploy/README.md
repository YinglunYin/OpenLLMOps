# OpenLLMOps 单机部署

控制面由 PostgreSQL、FastAPI、Vue/Nginx、Prometheus、DCGM Exporter 和 node-agent 组成。推理与训练工作负载由 node-agent 按需创建，不在 Compose 中预先固化。

## 准备

宿主机需要 Ubuntu 22.04、NVIDIA 驱动、Docker Engine 28+、Docker Compose 2.33.1+ 和 NVIDIA Container Toolkit。Compose 的最低版本用于显式指定 API 容器的外联默认网关。正式机按 4 张 RTX 4090D 配置，开发机可将 `GPU_COUNT` 改为 `2`。

```bash
cp deploy/.env.example deploy/.env
sudo install -d -o 1000 -g 1000 \
  /srv/openllmops/{models,inbox,model-staging,datasets,checkpoints,training-configs,runtime}
chmod 0600 deploy/.env
```

在已安装 `argon2-cffi` 的 Python 环境中运行 `python3 scripts/generate_secrets.py`。脚本会在终端交互读取管理员密码，并生成六个互不复用的值；把输出填入 `.env`。其中 `ADMIN_PASSWORD_HASH` 必须保留单引号，否则 Argon2 哈希里的 `$` 会被 Compose 当成变量插值。脚本不写文件，真实密码、哈希和密钥都不要提交到 Git。

生产环境还需要把 `CORS_ORIGINS` 改为实际 HTTPS 来源。`WEB_PROXY_IP` 必须位于 `CONTROL_SUBNET`，`TRUSTED_PROXY_CIDRS` 则必须是该地址对应的 `/32`；API 只信任这一台 Nginx 的转发头，不信任控制网络中的其他容器。

## 私有模型仓库凭证

公开 Hugging Face/ModelScope 仓库不需要令牌。需要私有仓库时，把令牌分别写入宿主机普通文件，不要写入 `.env`、接口参数或日志：

```bash
mkdir -p deploy/secrets/model-sources
# 使用本地编辑器创建所需文件，然后移除所有写权限。
chmod 0444 deploy/secrets/model-sources/huggingface.token
chmod 0444 deploy/secrets/model-sources/modelscope.token
```

再在 `.env` 中按需配置容器内的固定只读路径：

```dotenv
HUGGINGFACE_TOKEN_FILE=/run/secrets/model-sources/huggingface.token
MODELSCOPE_TOKEN_FILE=/run/secrets/model-sources/modelscope.token
```

未使用的项保持空值。后端会拒绝软链接、可写文件、超大文件和相对路径；预检脚本也会在启动前检查宿主机映射。人工拷入的离线模型放在 `/srv/openllmops/inbox`，在线下载先进入 `/srv/openllmops/model-staging`，校验成功后再原子移入 models；`model-staging` 和 `models` 必须位于同一文件系统。

## 训练镜像安全基线

截至 2026-08-24，上游[最新正式版 `v0.9.5`](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.9.5) 仍受 [GHSA-mwc7-mf87-v3mf / CVE-2026-58116](https://github.com/advisories/GHSA-mwc7-mf87-v3mf) 影响，没有可直接采用的官方修复版本。上游 [`c4e09c7cbe18844816af9e18a97fe465515edbcd`](https://github.com/hiyouga/LLaMA-Factory/commit/c4e09c7cbe18844816af9e18a97fe465515edbcd) 源码标记为 `0.9.6.dev0`，审计时仍包含公告涉及的硬编码 `trust_remote_code=True` 路径，因此禁止直接使用任何 `hiyouga/llamafactory` 镜像或 `latest`。

项目提供固定安全衍生版 `openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1`。它以官方 OCI index digest 为基础，并在构建期校验七个相关源码文件的 SHA-256、永久禁用远程模型代码，再写入安全标签；任何上游源码漂移或补丁匹配数量变化都会让构建失败。该名称不是上游发行版本。

首次部署先显式构建训练运行时：

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml \
  --profile runtime-image build llamafactory-secure-image
```

生产环境应将构建结果推送到内部仓库，取得 registry 返回的 manifest digest：

```bash
docker tag openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1 \
  registry.internal/openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1
docker push registry.internal/openllmops/llamafactory-secure:0.9.6.dev0-c4e09c7-rcefix1
# 将下列 64 个 0 替换为 registry 返回的真实摘要。
docker pull registry.internal/openllmops/llamafactory-secure@sha256:0000000000000000000000000000000000000000000000000000000000000000
```

随后把 `LLAMAFACTORY_ALLOWED_IMAGES` 改为上述完整 digest 引用。`LLAMAFACTORY_SECURE_IMAGE` 只是本地构建输出 tag，不要把 tag 当作生产白名单。镜像在内部仓库复制时必须保留以下标签：

- `com.openllmops.security.ghsa-mwc7-mf87-v3mf=mitigated`
- `com.openllmops.security.trust-remote-code=disabled`
- `org.opencontainers.image.revision=c4e09c7cbe18844816af9e18a97fe465515edbcd`

启动前执行只读预检，确认训练镜像、GPU 数量、NVIDIA Container Toolkit、受控目录、代理网段、可选令牌文件和 Compose 配置一致：

```bash
sh deploy/scripts/preflight.sh deploy/.env
```

## 启动与检查

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml config
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
curl --insecure https://127.0.0.1/_gateway/health
```

首次联调默认生成 30 天自签名证书。正式环境应把组织 CA 签发的 `tls.crt` 和 `tls.key` 放入 `TLS_DIR`，设置 `TLS_AUTO_GENERATE=false` 后重建 Web 容器。

## 安全边界

- Web 是唯一映射到宿主机的业务服务；PostgreSQL、Prometheus、node-agent 和 Docker socket 代理均只在内部网络中可见。
- node-agent 的工作负载命令采用带时间戳和 nonce 的双向 HMAC；请求与响应都签名，并持久化 request_id/generation 水位以拒绝重放和迟到命令。旧的 token 直传写端点已移除。
- 训练镜像除命中白名单外还必须已存在于节点、通过安全引用策略并携带构建标签；启动时转为不可变 image ID，避免校验后的 tag 替换。
- 推理参数采用白名单，并永久拒绝 `trust_remote_code`、宿主机网络、特权容器和任意挂载。
- Docker socket 由只开放必要端点的代理隔离；它仍是高权限组件，应限制 Compose 文件和 `.env` 的宿主机访问权限。
- 动态容器以非 root 用户、只读根文件系统、全部 capabilities 丢弃和 `no-new-privileges` 运行。模型/数据只读，checkpoint 与任务缓存按任务单独可写。

## 停止

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml stop
```

`stop` 会保留 runtime 网络和动态工作负载容器。若要彻底执行 `down`，应先在界面停止并删除所有推理/训练容器；否则 Docker 会因动态容器仍连接 runtime 网络而拒绝删除该网络。数据库与 Prometheus 数据卷默认保留，不要在未备份时添加 `--volumes`。
