"""节点与训练镜像共享的固定容器内路径。"""

from pathlib import Path

WORKSPACE_CONFIG = Path("/workspace/config/training.json")
WORKSPACE_MODEL = Path("/workspace/model")
WORKSPACE_DATASET = Path("/workspace/dataset")
WORKSPACE_DATA_FILE = Path("/workspace/data/training.jsonl")
WORKSPACE_OUTPUT = Path("/workspace/output")
WORKSPACE_CACHE = Path("/workspace/cache")
