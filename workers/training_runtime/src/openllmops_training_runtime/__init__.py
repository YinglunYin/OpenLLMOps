"""安全训练运行时公开接口。"""

from .artifacts import (
    TrainingArtifactError,
    validate_adapter_directory,
    validate_checkpoint_directory,
    validate_full_model_directory,
)
from .contract import (
    WORKSPACE_CACHE,
    WORKSPACE_CONFIG,
    WORKSPACE_DATA_FILE,
    WORKSPACE_DATASET,
    WORKSPACE_MODEL,
    WORKSPACE_OUTPUT,
)
from .runtime import (
    ProcessSupervisor,
    TrainingInterrupted,
    TrainingRuntimeError,
    TrainingSpec,
    run_training,
    validate_training_config,
)

__all__ = [
    "WORKSPACE_CACHE",
    "WORKSPACE_CONFIG",
    "WORKSPACE_DATASET",
    "WORKSPACE_DATA_FILE",
    "WORKSPACE_MODEL",
    "WORKSPACE_OUTPUT",
    "ProcessSupervisor",
    "TrainingArtifactError",
    "TrainingInterrupted",
    "TrainingRuntimeError",
    "TrainingSpec",
    "run_training",
    "validate_adapter_directory",
    "validate_checkpoint_directory",
    "validate_full_model_directory",
    "validate_training_config",
]
