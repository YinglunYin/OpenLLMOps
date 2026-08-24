from enum import StrEnum


class StringEnum(StrEnum):
    """API、数据库与任务执行器共享的小写字符串枚举。"""


class ModelSourceType(StringEnum):
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    SFTP = "sftp"
    MANUAL = "manual"
    TRAINED = "trained"


class ModelKind(StringEnum):
    BASE = "base"
    INSTRUCT = "instruct"
    EMBEDDING = "embedding"


class ModelImportSource(StringEnum):
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    CONTROLLED_DIRECTORY = "controlled_directory"


class ModelImportStatus(StringEnum):
    PENDING = "pending"
    TRANSFERRING = "transferring"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    CANCELING = "canceling"
    CANCELED = "canceled"


class AssetStatus(StringEnum):
    IMPORTING = "importing"
    READY = "ready"
    FAILED = "failed"


class DatasetType(StringEnum):
    CPT = "cpt"
    SFT = "sft"
    EVALUATION = "evaluation"


class DatasetStatus(StringEnum):
    VALIDATING = "validating"
    READY = "ready"
    INVALID = "invalid"


class DeploymentTaskType(StringEnum):
    GENERATE = "generate"
    EMBEDDING = "embedding"


class DesiredServiceState(StringEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class DeploymentState(StringEnum):
    CREATED = "created"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class TrainingStage(StringEnum):
    CPT = "cpt"
    SFT = "sft"


class TrainingAlgorithm(StringEnum):
    FREEZE = "freeze"
    LORA = "lora"
    QLORA = "qlora"


class DesiredJobState(StringEnum):
    RUNNING = "running"
    TERMINATED = "terminated"


class JobState(StringEnum):
    CREATED = "created"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELING = "canceling"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LeaseOwnerType(StringEnum):
    DEPLOYMENT = "deployment"
    TRAINING = "training"
    EVALUATION = "evaluation"
