from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.models import Dataset, ModelAsset, TrainingJob
from app.models.enums import (
    AssetStatus,
    DatasetStatus,
    DatasetType,
    JobState,
    ModelKind,
    TrainingAlgorithm,
    TrainingStage,
)
from app.schemas.training import (
    TrainingArtifactKind,
    TrainingArtifactRead,
    TrainingObservationMetadata,
    TrainingParameters,
)
from app.services.dataset_files import MAX_DATASET_BYTES, MAX_LINE_BYTES, validate_training_record

MAX_TREE_DEPTH = 64
MAX_JSON_METADATA_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
CHECKPOINT_NAME = re.compile(r"^checkpoint-[0-9]+$")
TOKENIZER_PAYLOADS = {
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
}
ADAPTER_FILE = re.compile(r"^adapter_model(?:-[0-9]+-of-[0-9]+)?\.safetensors$")
ADAPTER_METADATA_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "README.md",
}
UNSAFE_SERIALIZATION_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
}
UNSAFE_MODEL_SUFFIXES = UNSAFE_SERIALIZATION_SUFFIXES | {".py", ".so"}


class TrainingControlError(ValueError):
    """训练执行配置或受控产物没有通过控制面信任边界。"""


def _reject_json_constant(value: str) -> None:
    raise TrainingControlError(f"JSON 不允许非有限数值：{value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingControlError(f"JSON 对象包含重复字段：{key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class _TreeNode:
    relative_path: Path
    is_directory: bool
    size_bytes: int
    signature: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class TrainingArtifactTree:
    kind: TrainingArtifactKind
    root: Path
    nodes: tuple[_TreeNode, ...]
    file_count: int
    size_bytes: int
    root_signature: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class PublishedModelFiles:
    path: Path
    size_bytes: int
    checksum: str
    created: bool
    artifact_kind: str


def _signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _absolute_path(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise TrainingControlError(f"{label}必须是绝对路径")
    return Path(os.path.abspath(path))


def _configured_root(root: Path, label: str) -> tuple[Path, Path]:
    raw_root = _absolute_path(root, label)
    try:
        root_stat = raw_root.lstat()
    except OSError as exc:
        raise TrainingControlError(f"{label}不存在或不可读：{raw_root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise TrainingControlError(f"{label}必须是非软链接目录：{raw_root}")
    try:
        return raw_root, raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingControlError(f"{label}无法解析：{raw_root}") from exc


def _reject_symlink_components(root: Path, candidate: Path, label: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise TrainingControlError(f"{label}越出受控目录") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise TrainingControlError(f"无法检查{label}") from exc
        if stat.S_ISLNK(mode):
            raise TrainingControlError(f"{label}不能包含软链接")


def _strict_existing_path(
    candidate: Path,
    root: Path,
    *,
    directory: bool,
    label: str,
) -> Path:
    raw_root, resolved_root = _configured_root(root, f"{label}受控根目录")
    raw_candidate = _absolute_path(candidate, label)
    _reject_symlink_components(raw_root, raw_candidate, label)
    try:
        candidate_stat = raw_candidate.lstat()
        resolved = raw_candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingControlError(f"{label}不存在、不可读或越出受控目录") from exc
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if stat.S_ISLNK(candidate_stat.st_mode) or not expected(candidate_stat.st_mode):
        kind = "目录" if directory else "普通文件"
        raise TrainingControlError(f"{label}必须是非软链接{kind}")
    try:
        if directory:
            with os.scandir(raw_candidate):
                pass
        else:
            descriptor = os.open(raw_candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            os.close(descriptor)
    except OSError as exc:
        raise TrainingControlError(f"{label}不可读") from exc
    return resolved


def derive_training_output_dir(settings: Settings, job_id: uuid.UUID) -> Path:
    raw_root, _ = _configured_root(settings.checkpoint_root, "训练输出根目录")
    output_dir = raw_root / str(job_id)
    _reject_symlink_components(raw_root, output_dir, "训练输出目录")
    try:
        output_stat = output_dir.lstat()
    except FileNotFoundError:
        return output_dir
    except OSError as exc:
        raise TrainingControlError("无法检查训练输出目录") from exc
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise TrainingControlError("训练输出目录必须是非软链接目录")
    return output_dir


def _require_empty_directory(path: Path, label: str) -> None:
    try:
        if any(path.iterdir()):
            raise TrainingControlError(f"{label}已存在且非空，不能被新任务复用")
    except OSError as exc:
        raise TrainingControlError(f"无法检查{label}") from exc


def validate_training_parameters(
    stage: TrainingStage,
    algorithm: TrainingAlgorithm,
    raw: dict[str, Any] | TrainingParameters,
) -> TrainingParameters:
    try:
        parameters = raw if isinstance(raw, TrainingParameters) else TrainingParameters.model_validate(raw)
    except ValidationError as exc:
        raise TrainingControlError("训练参数结构、类型或范围无效") from exc
    if stage == TrainingStage.CPT and algorithm != TrainingAlgorithm.LORA:
        raise TrainingControlError("继续预训练（CPT）首版仅支持 LoRA")
    if stage == TrainingStage.SFT and parameters.template is None:
        raise TrainingControlError("SFT 训练必须指定受支持的 template")
    return parameters


def build_training_execution(
    row: TrainingJob,
    asset: ModelAsset,
    dataset: Dataset,
    settings: Settings,
) -> dict[str, Any]:
    """只从可信数据库字段和服务端配置构造 node-agent execution。"""

    if asset.status != AssetStatus.READY or dataset.status != DatasetStatus.READY:
        raise TrainingControlError("模型资产和数据集必须处于 ready 状态")
    if asset.model_kind == ModelKind.EMBEDDING:
        raise TrainingControlError("Embedding 模型不支持生成式训练")
    expected_dataset_type = DatasetType.CPT if row.stage == TrainingStage.CPT else DatasetType.SFT
    if dataset.dataset_type != expected_dataset_type:
        raise TrainingControlError(f"训练阶段需要 {expected_dataset_type.value} 数据集")
    parameters = validate_training_parameters(row.stage, row.algorithm, row.training_config)
    model_path = _strict_existing_path(
        Path(asset.local_path),
        settings.model_root,
        directory=True,
        label="训练模型路径",
    )
    model_tree = _scan_tree("full", model_path, settings)
    validate_deployable_model(model_tree)
    dataset_path = _strict_existing_path(
        Path(dataset.local_path),
        settings.dataset_root,
        directory=False,
        label="训练数据集路径",
    )
    if dataset_path.suffix.lower() != ".jsonl":
        raise TrainingControlError("训练数据集必须是 JSONL 文件")
    _verify_training_dataset(dataset_path, dataset, row.stage)
    expected_output = derive_training_output_dir(settings, row.id)
    if Path(row.output_dir) != expected_output:
        raise TrainingControlError("训练输出目录不是控制面按 job UUID 派生的路径")
    if expected_output.exists():
        _require_empty_directory(expected_output, "训练输出目录")
    return {
        "runner": "llamafactory",
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "stage": row.stage.value,
        "algorithm": row.algorithm.value,
        "training_config": parameters.model_dump(mode="json", exclude_none=True),
        "output_dir": str(expected_output),
    }


def _verify_training_dataset(path: Path, dataset: Dataset, stage: TrainingStage) -> None:
    """调度前重算训练集摘要与单一行格式，防止 READY 后文件被替换。"""

    digest = hashlib.sha256()
    record_count = 0
    record_format: str | None = None
    dataset_type = DatasetType.CPT if stage == TrainingStage.CPT else DatasetType.SFT
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_DATASET_BYTES:
                raise TrainingControlError("训练数据集为空或超过 5 GiB")
            line_number = 0
            while True:
                raw_line = source.readline(MAX_LINE_BYTES + 1)
                if not raw_line:
                    break
                line_number += 1
                if len(raw_line) > MAX_LINE_BYTES:
                    raise TrainingControlError(f"训练数据集第 {line_number} 行超过 16 MiB")
                digest.update(raw_line)
                if not raw_line.strip():
                    raise TrainingControlError(f"训练数据集第 {line_number} 行为空")
                try:
                    item = json.loads(
                        raw_line,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_unique_json_object,
                    )
                except TrainingControlError:
                    raise
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
                    raise TrainingControlError(f"训练数据集第 {line_number} 行 JSON 无效") from exc
                shape_error, current_format = validate_training_record(item, dataset_type)
                if shape_error or current_format is None:
                    raise TrainingControlError(
                        f"训练数据集第 {line_number} 行格式无效：{shape_error or '未知格式'}"
                    )
                if record_format is None:
                    record_format = current_format
                elif record_format != current_format:
                    raise TrainingControlError("训练数据集混用了多种行格式")
                record_count += 1
            after = os.fstat(source.fileno())
    except TrainingControlError:
        raise
    except OSError as exc:
        raise TrainingControlError("训练数据集无法安全读取") from exc
    if _signature(before) != _signature(after):
        raise TrainingControlError("训练数据集读取期间发生变化")
    if record_count == 0:
        raise TrainingControlError("训练数据集不含有效记录")
    if (
        dataset.size_bytes != before.st_size
        or dataset.record_count != record_count
        or dataset.sha256 != digest.hexdigest()
    ):
        raise TrainingControlError("训练数据集与数据库大小、记录数或 SHA-256 不一致")
    if dataset.schema_summary.get("record_format") != record_format:
        raise TrainingControlError("训练数据集行格式与数据库 schema 摘要不一致")


def _safe_component(name: str) -> None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TrainingControlError("训练产物文件名不是有效 UTF-8") from exc
    if (
        name in {"", ".", ".."}
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise TrainingControlError("训练产物文件名包含不安全字符")


def _scan_tree(
    kind: TrainingArtifactKind,
    root: Path,
    settings: Settings,
) -> TrainingArtifactTree:
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise TrainingControlError("训练产物根路径必须是非软链接目录")
    root_device = root_stat.st_dev
    nodes: list[_TreeNode] = []
    file_count = 0
    total_bytes = 0

    def visit(directory: Path, relative_directory: Path, depth: int) -> None:
        nonlocal file_count, total_bytes
        if depth > MAX_TREE_DEPTH:
            raise TrainingControlError(f"训练产物目录深度超过 {MAX_TREE_DEPTH}")
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise TrainingControlError("训练产物目录不可读") from exc
        for child in children:
            _safe_component(child.name)
            relative = relative_directory / child.name
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise TrainingControlError("无法检查训练产物节点") from exc
            if child_stat.st_dev != root_device:
                raise TrainingControlError("训练产物不能跨越文件系统挂载点")
            if stat.S_ISLNK(child_stat.st_mode):
                raise TrainingControlError("训练产物不能包含软链接")
            if stat.S_ISDIR(child_stat.st_mode):
                nodes.append(_TreeNode(relative, True, 0, _signature(child_stat)))
                if len(nodes) > settings.training_artifact_max_files:
                    raise TrainingControlError("训练产物节点数量超过安全上限")
                visit(Path(child.path), relative, depth + 1)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise TrainingControlError("训练产物不能包含设备、FIFO、socket 等特殊文件")
            if child_stat.st_nlink != 1:
                raise TrainingControlError("训练产物不能包含硬链接文件")
            file_count += 1
            total_bytes += child_stat.st_size
            if len(nodes) + 1 > settings.training_artifact_max_files:
                raise TrainingControlError("训练产物文件/目录数量超过安全上限")
            if total_bytes > settings.training_artifact_max_bytes:
                raise TrainingControlError("训练产物总大小超过安全上限")
            nodes.append(_TreeNode(relative, False, child_stat.st_size, _signature(child_stat)))

    visit(root, Path(), 0)
    if file_count == 0:
        raise TrainingControlError("训练产物目录不包含普通文件")
    return TrainingArtifactTree(
        kind=kind,
        root=root,
        nodes=tuple(nodes),
        file_count=file_count,
        size_bytes=total_bytes,
        root_signature=_signature(root_stat),
    )


def _filter_adapter_tree(tree: TrainingArtifactTree) -> TrainingArtifactTree:
    selected = tuple(
        node
        for node in tree.nodes
        if not node.is_directory
        and len(node.relative_path.parts) == 1
        and (
            node.relative_path.name in ADAPTER_METADATA_FILES
            or ADAPTER_FILE.fullmatch(node.relative_path.name)
        )
    )
    names = {node.relative_path.name for node in selected}
    if "adapter_config.json" not in names or not any(ADAPTER_FILE.fullmatch(name) for name in names):
        raise TrainingControlError("adapter 产物缺少 adapter_config.json 或 safetensors 权重")
    return TrainingArtifactTree(
        kind="adapter",
        root=tree.root,
        nodes=selected,
        file_count=len(selected),
        size_bytes=sum(node.size_bytes for node in selected),
        root_signature=tree.root_signature,
    )


def _filter_checkpoint_tree(tree: TrainingArtifactTree) -> TrainingArtifactTree:
    """checkpoint 下载仅保留不会触发 Python 反序列化的文件。

    取消/失败发生在 Agent 的成功后清理之前，optimizer、scheduler、rng 等状态常以
    pickle 兼容格式存在。首版不支持 resume，因此这些文件既无合同价值也不能下发。
    """

    safe_files = tuple(
        node
        for node in tree.nodes
        if not node.is_directory and node.relative_path.suffix.lower() not in UNSAFE_SERIALIZATION_SUFFIXES
    )
    if not safe_files:
        raise TrainingControlError("checkpoint 不含可安全导出的文件")
    safe_paths = {node.relative_path for node in safe_files}
    safe_directories = tuple(
        node
        for node in tree.nodes
        if node.is_directory and any(path.is_relative_to(node.relative_path) for path in safe_paths)
    )
    selected = tuple(sorted((*safe_directories, *safe_files), key=lambda node: node.relative_path))
    return TrainingArtifactTree(
        kind="checkpoint",
        root=tree.root,
        nodes=selected,
        file_count=len(safe_files),
        size_bytes=sum(node.size_bytes for node in safe_files),
        root_signature=tree.root_signature,
    )


def _reject_unsafe_export_files(tree: TrainingArtifactTree) -> None:
    unsafe = [
        node.relative_path.as_posix()
        for node in tree.nodes
        if not node.is_directory and node.relative_path.suffix.lower() in UNSAFE_MODEL_SUFFIXES
    ]
    if unsafe:
        raise TrainingControlError(f"训练产物包含不安全的可执行/反序列化文件：{unsafe[:3]}")


def _artifact_path(job: TrainingJob, kind: TrainingArtifactKind, settings: Settings) -> Path | None:
    output = derive_training_output_dir(settings, job.id)
    if Path(job.output_dir) != output:
        raise TrainingControlError("训练任务 output_dir 与 job UUID 不一致")
    raw: str | None
    if kind == "checkpoint":
        raw = job.checkpoint_path
        if raw is None:
            return None
        candidate = Path(raw)
        if candidate.parent != output or not CHECKPOINT_NAME.fullmatch(candidate.name):
            raise TrainingControlError("checkpoint_path 不符合受控 checkpoint-* 路径合同")
    elif kind == "adapter":
        raw = job.adapter_path
        if raw is None:
            return None
        candidate = Path(raw)
        if job.algorithm not in {TrainingAlgorithm.LORA, TrainingAlgorithm.QLORA} or candidate != output:
            raise TrainingControlError("adapter_path 与训练算法或受控输出目录不一致")
    elif kind == "merged":
        raw = job.merged_model_path
        if raw is None:
            return None
        candidate = Path(raw)
        expected = output if job.algorithm == TrainingAlgorithm.FREEZE else output / "merged"
        if candidate != expected:
            raise TrainingControlError("merged_model_path 与训练算法或受控输出目录不一致")
    else:
        # canceled/failed 的 output 是未清理 raw 工作区，只允许通过 Agent 已登记的
        # checkpoint/adapter/merged 路径按各自安全策略导出。
        if job.actual_state != JobState.SUCCEEDED:
            return None
        candidate = output
    try:
        candidate.lstat()
    except FileNotFoundError:
        if kind == "full":
            return None
        raise TrainingControlError(f"训练产物 {kind} 已登记但目录不存在") from None
    except OSError as exc:
        raise TrainingControlError(f"无法检查训练产物 {kind}") from exc
    return _strict_existing_path(
        candidate,
        settings.checkpoint_root,
        directory=True,
        label=f"训练产物 {kind}",
    )


def validate_training_observation_metadata(
    job: TrainingJob,
    raw: dict[str, Any],
    settings: Settings,
    *,
    require_success_artifacts: bool,
) -> TrainingObservationMetadata:
    try:
        metadata = TrainingObservationMetadata.model_validate(raw)
    except ValidationError as exc:
        raise TrainingControlError("node-agent 返回的训练 metadata 结构无效") from exc
    output = derive_training_output_dir(settings, job.id)
    if Path(job.output_dir) != output:
        raise TrainingControlError("训练任务 output_dir 与 job UUID 不一致")
    if metadata.checkpoint_path is not None:
        checkpoint = Path(metadata.checkpoint_path)
        if checkpoint.parent != output or not CHECKPOINT_NAME.fullmatch(checkpoint.name):
            raise TrainingControlError("node-agent 返回的 checkpoint_path 不符合受控合同")
    if job.algorithm in {TrainingAlgorithm.LORA, TrainingAlgorithm.QLORA}:
        if metadata.adapter_path is not None and Path(metadata.adapter_path) != output:
            raise TrainingControlError("node-agent 返回的 adapter_path 不符合受控合同")
        if metadata.merged_model_path is not None and Path(metadata.merged_model_path) != output / "merged":
            raise TrainingControlError("node-agent 返回的 merged_model_path 不符合受控合同")
        if require_success_artifacts and (
            metadata.adapter_path is None or metadata.merged_model_path is None
        ):
            raise TrainingControlError("LoRA/QLoRA 成功响应必须包含 adapter 与 merged 模型路径")
    else:
        if metadata.adapter_path is not None:
            raise TrainingControlError("Freeze 成功响应不能包含 adapter_path")
        if metadata.merged_model_path is not None and Path(metadata.merged_model_path) != output:
            raise TrainingControlError("node-agent 返回的 Freeze 完整模型路径不符合受控合同")
        if require_success_artifacts and metadata.merged_model_path is None:
            raise TrainingControlError("Freeze 成功响应必须包含完整模型路径")
    return metadata


def validate_successful_training_artifacts(job: TrainingJob, settings: Settings) -> None:
    """对 Agent 已确认成功的产物做控制面独立文件系统复核。"""

    if job.actual_state != JobState.SUCCEEDED:
        raise TrainingControlError("训练产物成功校验只能用于 succeeded 任务")
    full = inspect_training_artifact(job, "full", settings)
    if full is None:  # pragma: no cover - 成功 metadata 已要求输出路径存在。
        raise TrainingControlError("成功训练任务缺少完整输出目录")
    if job.checkpoint_path is not None:
        inspect_training_artifact(job, "checkpoint", settings)
    if job.algorithm in {TrainingAlgorithm.LORA, TrainingAlgorithm.QLORA}:
        adapter = inspect_training_artifact(job, "adapter", settings)
        merged = inspect_training_artifact(job, "merged", settings)
        if adapter is None or merged is None:
            raise TrainingControlError("LoRA/QLoRA 成功产物缺少 adapter 或 merged 模型")
        validate_deployable_model(merged)
    else:
        merged = inspect_training_artifact(job, "merged", settings)
        if merged is None:
            raise TrainingControlError("Freeze 成功产物缺少完整模型")
        validate_deployable_model(merged)


def inspect_training_artifact(
    job: TrainingJob,
    kind: TrainingArtifactKind,
    settings: Settings,
) -> TrainingArtifactTree | None:
    if job.actual_state not in {JobState.CANCELED, JobState.SUCCEEDED, JobState.FAILED}:
        raise TrainingControlError("只有终态训练任务可以读取产物")
    artifact_path = _artifact_path(job, kind, settings)
    if artifact_path is None:
        return None
    tree = _scan_tree(kind, artifact_path, settings)
    if kind == "checkpoint":
        return _filter_checkpoint_tree(tree)
    if kind == "adapter":
        return _filter_adapter_tree(tree)
    _reject_unsafe_export_files(tree)
    return tree


def list_training_artifacts(
    job: TrainingJob,
    settings: Settings,
) -> list[TrainingArtifactRead]:
    artifacts: list[TrainingArtifactRead] = []
    for kind in ("checkpoint", "adapter", "merged", "full"):
        tree = inspect_training_artifact(job, kind, settings)  # type: ignore[arg-type]
        if tree is not None:
            artifacts.append(artifact_read(tree, job.id))
    return artifacts


def artifact_read(tree: TrainingArtifactTree, job_id: uuid.UUID) -> TrainingArtifactRead:
    return TrainingArtifactRead(
        kind=tree.kind,
        path=str(tree.root),
        file_count=tree.file_count,
        size_bytes=tree.size_bytes,
        archive_filename=f"training-{job_id}-{tree.kind}.tar.gz",
    )


def _verify_node(root: Path, node: _TreeNode) -> None:
    try:
        current = (root / node.relative_path).lstat()
    except OSError as exc:
        raise TrainingControlError("训练产物在读取期间被删除或替换") from exc
    if _signature(current) != node.signature:
        raise TrainingControlError("训练产物在读取期间发生变化")


def _normalized_tar_info(name: str, *, directory: bool, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.size = 0 if directory else size
    return info


def build_training_archive(
    job: TrainingJob,
    kind: TrainingArtifactKind,
    settings: Settings,
) -> tuple[Path, TrainingArtifactRead]:
    tree = inspect_training_artifact(job, kind, settings)
    if tree is None:
        raise TrainingControlError(f"训练任务没有可下载的 {kind} 产物")
    checkpoint_root, _ = _configured_root(settings.checkpoint_root, "训练输出根目录")
    archive_root = checkpoint_root / ".download-cache"
    try:
        archive_root.mkdir(mode=0o700, exist_ok=True)
        archive_stat = archive_root.lstat()
    except OSError as exc:
        raise TrainingControlError("无法创建训练下载临时目录") from exc
    if stat.S_ISLNK(archive_stat.st_mode) or not stat.S_ISDIR(archive_stat.st_mode):
        raise TrainingControlError("训练下载临时目录必须是非软链接目录")
    descriptor, raw_archive_path = tempfile.mkstemp(
        prefix=f".{job.id}-{kind}-",
        suffix=".tar.gz",
        dir=archive_root,
    )
    archive_path = Path(raw_archive_path)
    try:
        with (
            os.fdopen(descriptor, "wb") as raw_output,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            archive.addfile(_normalized_tar_info("artifact", directory=True))
            for node in tree.nodes:
                _verify_node(tree.root, node)
                archive_name = f"artifact/{node.relative_path.as_posix()}"
                info = _normalized_tar_info(
                    archive_name,
                    directory=node.is_directory,
                    size=node.size_bytes,
                )
                if node.is_directory:
                    archive.addfile(info)
                    continue
                source_path = tree.root / node.relative_path
                file_descriptor = os.open(
                    source_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                with os.fdopen(file_descriptor, "rb") as source:
                    if _signature(os.fstat(source.fileno())) != node.signature:
                        raise TrainingControlError("训练产物在归档期间发生变化")
                    archive.addfile(info, source)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        if _signature(tree.root.lstat()) != tree.root_signature:
            raise TrainingControlError("训练产物目录在归档期间发生变化")
        for node in tree.nodes:
            _verify_node(tree.root, node)
        archive_path.chmod(0o600)
        return archive_path, artifact_read(tree, job.id)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise


def _load_small_json(path: Path, label: str) -> dict[str, Any]:
    try:
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise TrainingControlError(f"{label}必须是非软链接普通文件")
        if not 1 <= file_stat.st_size <= MAX_JSON_METADATA_BYTES:
            raise TrainingControlError(f"{label}为空或超过 16 MiB")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            value = json.load(
                source,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except TrainingControlError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise TrainingControlError(f"{label}不是安全有效的 JSON") from exc
    if not isinstance(value, dict):
        raise TrainingControlError(f"{label}顶层必须是 JSON 对象")
    return value


def _validate_safetensors(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            file_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size < 10:
                raise TrainingControlError(f"safetensors 文件为空或过小：{path.name}")
            header_size = int.from_bytes(source.read(8), "little", signed=False)
            if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise TrainingControlError(f"safetensors header 大小无效：{path.name}")
            if 8 + header_size > file_stat.st_size:
                raise TrainingControlError(f"safetensors header 越界：{path.name}")
            header = json.loads(
                source.read(header_size),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except TrainingControlError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TrainingControlError(f"safetensors 文件头无效：{path.name}") from exc
    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise TrainingControlError(f"safetensors 不含张量索引：{path.name}")
    data_size = file_stat.st_size - 8 - header_size
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(tensor, dict):
            raise TrainingControlError(f"safetensors 张量描述无效：{path.name}")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise TrainingControlError(f"safetensors 张量偏移或形状无效：{path.name}")


def validate_deployable_model(tree: TrainingArtifactTree) -> None:
    unsafe = [
        node.relative_path.as_posix()
        for node in tree.nodes
        if not node.is_directory and node.relative_path.suffix.lower() in UNSAFE_MODEL_SUFFIXES
    ]
    if unsafe:
        raise TrainingControlError(f"可部署模型包含不安全的可执行/反序列化文件：{unsafe[:3]}")
    top_level_files = {
        node.relative_path.name: tree.root / node.relative_path
        for node in tree.nodes
        if not node.is_directory and len(node.relative_path.parts) == 1
    }
    config = top_level_files.get("config.json")
    tokenizer_config = top_level_files.get("tokenizer_config.json")
    if config is None or tokenizer_config is None:
        raise TrainingControlError("可部署模型必须包含 config.json 和 tokenizer_config.json")
    config_json = _load_small_json(config, "模型 config.json")
    if not isinstance(config_json.get("model_type"), str) or not config_json["model_type"].strip():
        raise TrainingControlError("模型 config.json 缺少有效 model_type")
    tokenizer_json = _load_small_json(tokenizer_config, "模型 tokenizer_config.json")
    for label, value in (("config.json", config_json), ("tokenizer_config.json", tokenizer_json)):
        if value.get("trust_remote_code") is True or value.get("auto_map"):
            raise TrainingControlError(f"模型 {label} 请求 remote custom code，禁止发布")
    if not TOKENIZER_PAYLOADS & top_level_files.keys():
        raise TrainingControlError("可部署模型缺少 tokenizer.json/tokenizer.model/vocab.json 等词表文件")
    single_weight = top_level_files.get("model.safetensors")
    index_path = top_level_files.get("model.safetensors.index.json")
    if single_weight is not None and index_path is not None:
        raise TrainingControlError("模型不能同时包含单文件权重与分片索引")
    if single_weight is not None:
        weight_files = [single_weight]
    elif index_path is not None:
        index = _load_small_json(index_path, "模型 model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise TrainingControlError("模型 safetensors 分片索引缺少 weight_map")
        shard_names = set(weight_map.values())
        if not all(
            isinstance(name, str) and re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name)
            for name in shard_names
        ):
            raise TrainingControlError("模型 safetensors 分片索引包含不安全文件名")
        actual_shards = {
            name for name in top_level_files if re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name)
        }
        if shard_names != actual_shards:
            raise TrainingControlError("模型 safetensors 分片索引与实际文件集合不一致")
        weight_files = [top_level_files[name] for name in sorted(shard_names)]
    else:
        raise TrainingControlError("可部署模型缺少 model.safetensors 或分片索引")
    unexpected_weights = [
        name
        for name in top_level_files
        if name.endswith(".safetensors")
        and name != "model.safetensors"
        and not re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name)
    ]
    if unexpected_weights:
        raise TrainingControlError(f"可部署模型包含非完整模型 safetensors：{unexpected_weights[:3]}")
    for weight in weight_files:
        _validate_safetensors(weight)


def _tree_content_fingerprint(tree: TrainingArtifactTree) -> str:
    digest = hashlib.sha256()
    for node in tree.nodes:
        if node.is_directory:
            continue
        digest.update(node.relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        descriptor = os.open(
            tree.root / node.relative_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as source:
            if _signature(os.fstat(source.fileno())) != node.signature:
                raise TrainingControlError("训练模型产物在计算指纹期间发生变化")
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_tree(tree: TrainingArtifactTree, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for node in tree.nodes:
        target = destination / node.relative_path
        _verify_node(tree.root, node)
        if node.is_directory:
            target.mkdir(mode=0o755)
            continue
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        source_descriptor = os.open(
            tree.root / node.relative_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        target_descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(source_descriptor, "rb") as source, os.fdopen(target_descriptor, "wb") as target_file:
            if _signature(os.fstat(source.fileno())) != node.signature:
                raise TrainingControlError("训练模型产物在复制期间发生变化")
            shutil.copyfileobj(source, target_file, length=8 * 1024 * 1024)
            target_file.flush()
            os.fsync(target_file.fileno())
    if _signature(tree.root.lstat()) != tree.root_signature:
        raise TrainingControlError("训练模型产物目录在复制期间发生变化")


def _publication_source(job: TrainingJob, settings: Settings) -> tuple[str, TrainingArtifactTree]:
    candidates: list[TrainingArtifactKind]
    if job.algorithm in {TrainingAlgorithm.LORA, TrainingAlgorithm.QLORA}:
        candidates = ["merged", "full"]
    else:
        candidates = ["merged", "full"]
    errors: list[str] = []
    for kind in candidates:
        try:
            tree = inspect_training_artifact(job, kind, settings)
            if tree is None:
                continue
            validate_deployable_model(tree)
            return kind, tree
        except TrainingControlError as exc:
            errors.append(f"{kind}: {exc}")
    detail = "；".join(errors) if errors else "没有已登记的完整模型目录"
    raise TrainingControlError(f"训练任务没有可发布的完整模型：{detail}")


@contextmanager
def _publication_lock(model_root: Path, job_id: uuid.UUID) -> Iterator[None]:
    lock_path = model_root / f".publish-{job_id}.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise TrainingControlError("训练模型发布锁不是受控普通文件")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except TrainingControlError:
        raise
    except OSError as exc:
        raise TrainingControlError("无法获取训练模型发布锁") from exc
    finally:
        if "descriptor" in locals():
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def publish_training_model_files(job: TrainingJob, settings: Settings) -> PublishedModelFiles:
    if job.actual_state != JobState.SUCCEEDED:
        raise TrainingControlError("只有 succeeded 训练任务可以发布模型")
    artifact_kind, source = _publication_source(job, settings)
    source_checksum = _tree_content_fingerprint(source)
    raw_model_root, _ = _configured_root(settings.model_root, "模型资产根目录")
    final_path = raw_model_root / f"trained-{job.id}"
    _reject_symlink_components(raw_model_root, final_path, "训练模型发布目录")
    with _publication_lock(raw_model_root, job.id):
        try:
            final_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise TrainingControlError("无法检查训练模型发布目录") from exc
        else:
            existing_path = _strict_existing_path(
                final_path,
                raw_model_root,
                directory=True,
                label="已发布训练模型",
            )
            existing = _scan_tree("full", existing_path, settings)
            validate_deployable_model(existing)
            if _tree_content_fingerprint(existing) != source_checksum:
                raise TrainingControlError("训练模型目标目录已存在但内容与本任务产物不一致")
            return PublishedModelFiles(
                existing_path,
                existing.size_bytes,
                source_checksum,
                False,
                artifact_kind,
            )

        staging = Path(tempfile.mkdtemp(prefix=f".publish-{job.id}-", dir=raw_model_root))
        try:
            # mkdtemp 已创建目录；复制函数要求自行创建，以便目标存在即失败。
            staging.rmdir()
            _copy_tree(source, staging)
            staged_tree = _scan_tree("full", staging, settings)
            validate_deployable_model(staged_tree)
            if _tree_content_fingerprint(staged_tree) != source_checksum:
                raise TrainingControlError("训练模型 staging 内容校验不一致")
            # 持有跨进程 flock 后再次确认目标不存在，绝不调用可覆盖目标的 os.replace。
            if final_path.exists():
                raise TrainingControlError("训练模型发布目录被并发创建，请重试")
            try:
                os.rename(staging, final_path)
            except OSError as exc:
                raise TrainingControlError("训练模型发布目录原子落盘失败且未覆盖现有目录") from exc
            directory_descriptor = os.open(raw_model_root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            return PublishedModelFiles(
                final_path,
                staged_tree.size_bytes,
                source_checksum,
                True,
                artifact_kind,
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)
