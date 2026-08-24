import os

# main 模块坚持生产无默认密钥；测试收集阶段显式注入独立固定值。
os.environ.setdefault("NODE_AGENT_TOKEN", "agent-contract-test-secret-at-least-32-bytes")
