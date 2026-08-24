# 该 digest 是官方 2026-08-20 docker workflow 对 c4e09c7 上游提交生成的 amd64 OCI index。
FROM hiyouga/llamafactory@sha256:b96fd8dde0a5b5f177cf6cff70e782ebe0fc240b17c3b85e95c20b0b98fedc0a

USER root
ENV PYTHONDONTWRITEBYTECODE=1
COPY deploy/security/harden_llamafactory.py /usr/local/lib/openllmops/harden_llamafactory.py

# 哈希不匹配、补丁匹配数量变化或仍存在危险字面量时，镜像构建立即失败。
RUN python /usr/local/lib/openllmops/harden_llamafactory.py apply --root /app \
    && python /usr/local/lib/openllmops/harden_llamafactory.py verify --root /app \
    && chmod 0444 /usr/local/share/openllmops/llamafactory-hardening.json

LABEL org.opencontainers.image.title="OpenLLMOps hardened LLaMAFactory runtime" \
      org.opencontainers.image.version="0.9.6.dev0-c4e09c7-rcefix1" \
      org.opencontainers.image.revision="c4e09c7cbe18844816af9e18a97fe465515edbcd" \
      org.opencontainers.image.source="https://github.com/hiyouga/LLaMA-Factory" \
      org.opencontainers.image.base.digest="sha256:b96fd8dde0a5b5f177cf6cff70e782ebe0fc240b17c3b85e95c20b0b98fedc0a" \
      com.openllmops.security.ghsa-mwc7-mf87-v3mf="mitigated" \
      com.openllmops.security.trust-remote-code="disabled"

# node-agent 还会显式传入同一非 root 身份；这里也提供安全的直接运行默认值。
USER 1000:1000
