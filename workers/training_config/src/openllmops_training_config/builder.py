"""把简化表单转换为不接受任意命令的 LLaMA-Factory 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    stage: Stage
    algorithm: Algorithm
    model_path: Path
    dataset_dir: Path
    output_dir: Path
    dataset_name: str = "openllmops_dataset"
    dataset_format: DatasetFormat = DatasetFormat.ALPACA
    template: str | None = None
    epochs: float = 3.0
    learning_rate: float = 2e-4
    cutoff_len: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    freeze_trainable_layers: int = 2
    max_samples: int | None = None
    seed: int = 42


def _validate(request: TrainingRequest) -> None:
    if request.stage == Stage.CPT and request.algorithm != Algorithm.LORA:
        raise TrainingConfigError("继续预训练（CPT）仅支持 LoRA")
    if request.stage == Stage.CPT and request.dataset_format != DatasetFormat.CPT_TEXT:
        raise TrainingConfigError("CPT 必须使用 text/content 数据集")
    if request.stage == Stage.SFT and request.dataset_format == DatasetFormat.CPT_TEXT:
        raise TrainingConfigError("SFT 不能使用 CPT 文本格式")
    if request.stage == Stage.SFT and not request.template:
        raise TrainingConfigError("SFT 必须显式选择与模型匹配的模板")
    if not request.dataset_name.replace("_", "").isalnum():
        raise TrainingConfigError("dataset_name 只能包含字母、数字和下划线")

    ranges: tuple[tuple[bool, str], ...] = (
        (0 < request.epochs <= 100, "epochs 必须位于 0..100"),
        (1e-7 <= request.learning_rate <= 1.0, "learning_rate 超出安全范围"),
        (128 <= request.cutoff_len <= 131072, "cutoff_len 超出安全范围"),
        (1 <= request.per_device_batch_size <= 512, "单卡 batch size 超出安全范围"),
        (1 <= request.gradient_accumulation_steps <= 4096, "梯度累积步数超出安全范围"),
        (1 <= request.logging_steps <= 100000, "logging_steps 超出安全范围"),
        (1 <= request.save_steps <= 1000000, "save_steps 超出安全范围"),
        (0 <= request.warmup_ratio <= 1, "warmup_ratio 必须位于 0..1"),
        (1 <= request.lora_rank <= 1024, "LoRA rank 超出安全范围"),
        (1 <= request.lora_alpha <= 4096, "LoRA alpha 超出安全范围"),
        (0 <= request.lora_dropout < 1, "LoRA dropout 必须位于 0..1"),
        (0 <= request.freeze_trainable_layers <= 256, "Freeze 层数超出安全范围"),
    )
    for valid, message in ranges:
        if not valid:
            raise TrainingConfigError(message)


def build_training_config(request: TrainingRequest) -> dict[str, Any]:
    """返回可序列化配置；调用方可用 JSON 写入 `.yaml`（YAML 兼容 JSON）。"""

    _validate(request)
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
        "finetuning_type": (
            "freeze" if request.algorithm == Algorithm.FREEZE else "lora"
        ),
        "cutoff_len": request.cutoff_len,
        "preprocessing_num_workers": 8,
        "per_device_train_batch_size": request.per_device_batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "learning_rate": request.learning_rate,
        "num_train_epochs": request.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": request.warmup_ratio,
        "bf16": True,
        "ddp_timeout": 18_000_000,
        "logging_steps": request.logging_steps,
        "save_steps": request.save_steps,
        "save_total_limit": 5,
        "plot_loss": True,
        "report_to": "none",
        "seed": request.seed,
    }
    if request.max_samples is not None:
        if not 1 <= request.max_samples <= 1_000_000_000:
            raise TrainingConfigError("max_samples 超出安全范围")
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
    """生成每个不可变数据集目录内的 `dataset_info.json`。"""

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
