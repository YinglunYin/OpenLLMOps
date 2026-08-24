"""把控制面表单转换为严格白名单的 LLaMA-Factory 配置。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfigError(ValueError):
    """训练组合或超参数违反首版约束。"""


class Stage(StrEnum):
    CPT = "cpt"
    SFT = "sft"


class Algorithm(StrEnum):
    FREEZE = "freeze"
    LORA = "lora"
    QLORA = "qlora"


class DatasetFormat(StrEnum):
    CPT_TEXT = "cpt_text"
    ALPACA = "alpaca"
    MESSAGES = "messages"


class TrainingHyperparameters(BaseModel):
    """控制面可写的完整训练参数集合；任何额外键都必须在边界处失败。"""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    template: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$",
    )
    num_train_epochs: float = Field(default=3.0, gt=0, le=100)
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1)
    cutoff_len: int = Field(default=2048, ge=128, le=65_536)
    per_device_train_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=8, ge=1, le=4096)
    logging_steps: int = Field(default=10, ge=1, le=100_000)
    save_steps: int = Field(default=100, ge=1, le=1_000_000)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    lora_rank: int = Field(default=16, ge=1, le=1024)
    lora_alpha: int = Field(default=32, ge=1, le=4096)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    freeze_trainable_layers: int = Field(default=2, ge=1, le=256)
    max_samples: int | None = Field(default=None, ge=1, le=10_000_000)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class TrainingRequest(TrainingHyperparameters):
    """已由节点补齐路径和训练模式的内部请求。"""

    stage: Stage
    algorithm: Algorithm
    model_path: Path
    dataset_dir: Path
    output_dir: Path
    dataset_name: str = Field(default="openllmops_dataset", pattern=r"^[A-Za-z0-9_]{1,64}$")
    dataset_format: DatasetFormat = DatasetFormat.ALPACA


def _validate_combination(request: TrainingRequest) -> None:
    if request.stage == Stage.CPT and request.algorithm != Algorithm.LORA:
        raise TrainingConfigError("继续预训练（CPT）仅支持 LoRA")
    if request.stage == Stage.CPT and request.dataset_format != DatasetFormat.CPT_TEXT:
        raise TrainingConfigError("CPT 必须使用 text/content 数据集")
    if request.stage == Stage.SFT and request.dataset_format == DatasetFormat.CPT_TEXT:
        raise TrainingConfigError("SFT 不能使用 CPT 文本格式")
    if request.stage == Stage.SFT and not request.template:
        raise TrainingConfigError("SFT 必须显式选择与模型匹配的模板")


def build_training_config(request: TrainingRequest) -> dict[str, Any]:
    """返回节点生成的固定配置；调用方不能注入额外 YAML 字段。"""

    _validate_combination(request)
    config: dict[str, Any] = {
        "stage": "pt" if request.stage == Stage.CPT else "sft",
        "do_train": True,
        "model_name_or_path": str(request.model_path),
        "dataset": request.dataset_name,
        "dataset_dir": str(request.dataset_dir),
        "output_dir": str(request.output_dir),
        "overwrite_cache": False,
        "overwrite_output_dir": False,
        "trust_remote_code": False,
        "finetuning_type": "freeze" if request.algorithm == Algorithm.FREEZE else "lora",
        "cutoff_len": request.cutoff_len,
        "preprocessing_num_workers": 8,
        "per_device_train_batch_size": request.per_device_train_batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "learning_rate": request.learning_rate,
        "num_train_epochs": request.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": request.warmup_ratio,
        "bf16": True,
        "ddp_timeout": 18_000_000,
        "logging_steps": request.logging_steps,
        "save_steps": request.save_steps,
        "save_total_limit": 5,
        # 明确禁用 Transformers 的 pickle 权重回退；wrapper 仍会在成功后递归
        # 清理优化器等 Trainer 状态，控制面只上报 Safetensors 产物。
        "save_safetensors": True,
        "plot_loss": True,
        "report_to": "none",
        "seed": request.seed,
    }
    if request.max_samples is not None:
        config["max_samples"] = request.max_samples
    if request.stage == Stage.SFT:
        config["template"] = request.template
        config["packing"] = False
    if request.algorithm in {Algorithm.LORA, Algorithm.QLORA}:
        config.update(
            {
                "lora_target": "all",
                "lora_rank": request.lora_rank,
                "lora_alpha": request.lora_alpha,
                "lora_dropout": request.lora_dropout,
            }
        )
    if request.algorithm == Algorithm.QLORA:
        config.update({"quantization_bit": 4, "quantization_method": "bitsandbytes"})
    if request.algorithm == Algorithm.FREEZE:
        config.update(
            {
                "freeze_trainable_layers": request.freeze_trainable_layers,
                "freeze_trainable_modules": "all",
            }
        )
    return config


def build_dataset_info(
    file_name: str,
    dataset_format: DatasetFormat,
    dataset_name: str = "openllmops_dataset",
) -> dict[str, Any]:
    """生成每个不可变数据集目录内的 ``dataset_info.json``。"""

    if Path(file_name).name != file_name or not file_name.endswith(".jsonl"):
        raise TrainingConfigError("数据文件名必须是当前目录中的 .jsonl 文件")
    if not dataset_name.replace("_", "").isalnum():
        raise TrainingConfigError("dataset_name 只能包含字母、数字和下划线")
    if dataset_format == DatasetFormat.CPT_TEXT:
        descriptor = {"file_name": file_name, "columns": {"prompt": "text"}}
    elif dataset_format == DatasetFormat.ALPACA:
        descriptor = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
    else:
        descriptor = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return {dataset_name: descriptor}
