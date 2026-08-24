from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from collections.abc import Iterable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest
from openllmops_training_config import Algorithm as TrainingAlgorithm
from openllmops_training_config import DatasetFormat
from openllmops_training_config import Stage as TrainingStage
from openllmops_training_runtime import (
    WORKSPACE_CACHE,
    WORKSPACE_CONFIG,
    WORKSPACE_DATA_FILE,
    WORKSPACE_DATASET,
    WORKSPACE_MODEL,
    WORKSPACE_OUTPUT,
    TrainingArtifactError,
    TrainingRuntimeError,
    TrainingSpec,
    validate_adapter_directory,
    validate_checkpoint_directory,
    validate_full_model_directory,
    validate_training_config,
)

from .config import Settings
from .evaluation_image_policy import (
    UnsafeEvaluationImage,
    validate_evaluation_image_labels,
)
from .evaluation_runtime import (
    EvaluationInputError,
    load_dataset_manifest_summary,
    load_pair_report_metadata,
    prepare_output_directory,
    strict_existing_path,
)
from .gpu import NVMLUnavailable, read_busy_gpu_ids
from .image_policy import UnsafeTrainingImage, validate_hardening_labels
from .schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
    TrainingLaunchRequest,
    WorkloadInfo,
)

MANAGED_LABEL = "com.openllmops.managed"
KIND_LABEL = "com.openllmops.kind"
ID_LABEL = "com.openllmops.id"
GPU_LABEL = "com.openllmops.gpus"
ENDPOINT_LABEL = "com.openllmops.endpoint"
SERVICE_TYPE_LABEL = "com.openllmops.service-type"
GENERATION_LABEL = "com.openllmops.generation"
OWNER_TYPE_LABEL = "com.openllmops.owner-type"
OUTPUT_PATH_LABEL = "com.openllmops.output-path"
DATASET_MANIFEST_PATH_LABEL = "com.openllmops.dataset-manifest-path"
BASE_TEMPLATE_LABEL = "com.openllmops.base-template"
CANDIDATE_TEMPLATE_LABEL = "com.openllmops.candidate-template"
PORT_LABEL = "com.openllmops.port"
TRAINING_ALGORITHM_LABEL = "com.openllmops.training-algorithm"

ACTIVE_STATES = frozenset({"created", "running", "restarting", "paused"})
TERMINAL_STATES = frozenset({"exited", "dead"})
INFERENCE_HEALTH_INTERVAL_SECONDS = 5
INFERENCE_MAX_RESTARTS = 3

# 详细配置仍只允许无代码执行、无任意文件读取能力的 vLLM 参数。
VLLM_ARGUMENT_ALLOWLIST = frozenset(
    {
        "block-size",
        "cpu-offload-gb",
        "disable-custom-all-reduce",
        "disable-log-stats",
        "distributed-executor-backend",
        "dtype",
        "enable-chunked-prefill",
        "enable-prefix-caching",
        "enforce-eager",
        "gpu-memory-utilization",
        "kv-cache-dtype",
        "max-logprobs",
        "max-model-len",
        "max-num-batched-tokens",
        "max-num-seqs",
        "pipeline-parallel-size",
        "quantization",
        "seed",
        "tokenizer-mode",
    }
)

VLLM_BOOLEAN_ARGUMENTS = frozenset(
    {
        "disable-custom-all-reduce",
        "disable-log-stats",
        "enable-chunked-prefill",
        "enable-prefix-caching",
        "enforce-eager",
    }
)
VLLM_POSITIVE_INTEGER_ARGUMENTS = frozenset(
    {
        "max-model-len",
        "max-num-batched-tokens",
        "max-num-seqs",
    }
)
VLLM_POSITIVE_INTEGER_LIMITS = {
    # 目标节点最多 4 张 24 GiB 4090D；限制异常配置先于 vLLM OOM 失败。
    "max-model-len": 131_072,
    "max-num-batched-tokens": 65_536,
    "max-num-seqs": 1_024,
}
VLLM_NONNEGATIVE_INTEGER_ARGUMENTS = frozenset({"max-logprobs", "seed"})
VLLM_NONNEGATIVE_NUMBER_ARGUMENTS = frozenset({"cpu-offload-gb"})
VLLM_ENUM_ARGUMENTS: dict[str, frozenset[str]] = {
    "distributed-executor-backend": frozenset({"mp"}),
    "dtype": frozenset({"auto", "half", "float16", "bfloat16", "float", "float32"}),
    "kv-cache-dtype": frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}),
    "tokenizer-mode": frozenset({"auto", "slow", "mistral"}),
}

INFERENCE_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "VLLM_LOGGING_LEVEL",
    }
)
TRAINING_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "TOKENIZERS_PARALLELISM",
        "WANDB_DISABLED",
    }
)
SAFE_NAME = re.compile(r"^openllmops-(inference|training|evaluation)-[0-9a-f-]{36}$")


class RunnerError(RuntimeError):
    """可安全回传给控制面的 node-agent 错误。"""


class WorkloadConflict(RunnerError):
    pass


class WorkloadNotFound(RunnerError):
    pass


class InvalidWorkload(RunnerError):
    pass


class DockerRunner:
    def __init__(self, settings: Settings, client: docker.DockerClient | None = None) -> None:
        self.settings = settings
        self.client = client or docker.DockerClient(base_url=settings.docker_host, timeout=30)
        # 一个 worker 内将“检查空闲 GPU + 创建/启动容器”序列化，避免并发请求双重分配。
        self._allocation_lock = threading.RLock()

    def initialize(self) -> None:
        self.client.ping()
        try:
            self.client.networks.get(self.settings.runtime_network)
        except NotFound as exc:
            raise RunnerError(f"运行网络不存在：{self.settings.runtime_network}") from exc

    def close(self) -> None:
        self.client.close()

    def list_workloads(self) -> list[WorkloadInfo]:
        containers = self.client.containers.list(all=True, filters={"label": f"{MANAGED_LABEL}=true"})
        return [self._to_info(container) for container in containers]

    def get_workload(self, name: str) -> WorkloadInfo:
        return self._to_info(self._managed_container(name))

    @staticmethod
    def contract_workload_name(owner_type: str, workload_id: UUID) -> str:
        kind = {
            "deployment": "inference",
            "training": "training",
            "evaluation": "evaluation",
        }.get(owner_type)
        if kind is None:
            raise InvalidWorkload(f"不支持的工作负载类型：{owner_type}")
        return DockerRunner._workload_name(kind, workload_id)

    def get_contract_workload(self, owner_type: str, workload_id: UUID) -> WorkloadInfo:
        return self.get_workload(self.contract_workload_name(owner_type, workload_id))

    def prepare_contract_start(
        self, owner_type: str, workload_id: UUID, generation: int
    ) -> WorkloadInfo | None:
        """复用同代容器；只清理已结束的旧代容器，绝不抢占运行实例。"""

        name = self.contract_workload_name(owner_type, workload_id)
        with self._allocation_lock:
            try:
                container = self._managed_container(name)
            except WorkloadNotFound:
                return None
            container.reload()
            existing_generation = self._generation_from_labels(container.labels)
            if existing_generation > generation:
                raise WorkloadConflict("命令 generation 早于节点当前运行代")
            if existing_generation == generation:
                return self._to_info(container)
            if container.status in ACTIVE_STATES:
                raise WorkloadConflict("旧 generation 工作负载仍在运行，禁止自动抢占")
            container.remove(force=False, v=True)
            return None

    def stop_contract_workload(
        self,
        owner_type: str,
        workload_id: UUID,
        generation: int,
        timeout_seconds: int = 30,
    ) -> WorkloadInfo | None:
        """取消活动任务；与 STOP 竞争而自然成功的训练/评测容器必须保留。"""

        name = self.contract_workload_name(owner_type, workload_id)
        with self._allocation_lock:
            container = self._managed_container(name)
            container.reload()
            if self._generation_from_labels(container.labels) != generation:
                raise WorkloadConflict("停止命令 generation 与节点容器不一致")
            info = self._to_info(container)
            if owner_type in {"training", "evaluation"} and info.status == "exited" and info.exit_code == 0:
                return info
            if container.status in ACTIVE_STATES:
                container.stop(timeout=timeout_seconds)
                info = self._to_info(container)
                if (
                    owner_type in {"training", "evaluation"}
                    and info.status == "exited"
                    and info.exit_code == 0
                ):
                    return info
            container.remove(force=False, v=True)
            return None

    def cleanup_contract_workload(
        self,
        owner_type: str,
        workload_id: UUID,
        generation: int,
    ) -> None:
        """幂等删除一个已经确认处于终态的合同容器。

        cleanup 与普通 STOP 刻意分离：它绝不停止 active 容器，也不会在 Docker 状态
        不确定时报告成功。这里只删除容器本身（``v=False``），保留宿主 bind mount。
        """

        name = self.contract_workload_name(owner_type, workload_id)
        expected_kind = {
            "deployment": "inference",
            "training": "training",
            "evaluation": "evaluation",
        }[owner_type]
        with self._allocation_lock:
            try:
                # 此处不能使用 _managed_container：cleanup 必须区分真正 absent 与
                # 同名但标签不可信的容器，后者不能被当作安全清理成功。
                container = self.client.containers.get(name)
            except NotFound:
                return
            try:
                container.reload()
            except NotFound:
                return

            labels = container.labels
            if (
                labels.get(MANAGED_LABEL) != "true"
                or labels.get(ID_LABEL) != str(workload_id)
                or labels.get(KIND_LABEL) != expected_kind
                or labels.get(OWNER_TYPE_LABEL) != owner_type
            ):
                raise WorkloadConflict("cleanup 目标容器缺少可信 owner 标签")
            raw_generation = labels.get(GENERATION_LABEL)
            if not isinstance(raw_generation, str) or re.fullmatch(r"[1-9][0-9]*", raw_generation) is None:
                raise WorkloadConflict("cleanup 目标容器缺少可信 generation 标签")
            existing_generation = int(raw_generation)
            if existing_generation != generation:
                raise WorkloadConflict("cleanup generation 与节点容器不一致")
            if container.status in ACTIVE_STATES:
                raise WorkloadConflict("cleanup 仅允许删除已进入终态的工作负载")
            if container.status == "removing":
                raise WorkloadConflict("cleanup 目标容器正在删除，结果尚未确认")
            if container.status not in TERMINAL_STATES:
                raise WorkloadConflict(f"cleanup 无法确认容器终态：{container.status}")
            try:
                container.remove(force=False, v=False)
            except NotFound:
                # 另一个并发 cleanup 在终态检查后先完成，最终结果仍是可信 absent。
                return

    def quiesce_failed_inference(
        self,
        workload_id: UUID,
        generation: int,
        *,
        timeout_seconds: int,
    ) -> bool:
        """先停止并确认推理容器不再活动，确认前绝不允许控制面释放租约。"""

        name = self._workload_name("inference", workload_id)
        with self._allocation_lock:
            try:
                container = self._managed_container(name)
                container.reload()
            except WorkloadNotFound:
                return True
            except NotFound:
                # get 成功后 reload 才发现容器消失，同样已确认不再占用 GPU。
                return True
            except DockerException:
                return False
            if self._generation_from_labels(container.labels) != generation:
                return False
            if container.status == "removing":
                return False
            if container.status == "created":
                try:
                    container.remove(force=False, v=True)
                except DockerException:
                    return False
                return True
            if container.status in {"running", "restarting", "paused"}:
                # stop 响应丢失不代表停止未发生；仍需 inspect 二次确认。
                with suppress(DockerException):
                    container.stop(timeout=timeout_seconds)
            try:
                container.reload()
            except NotFound:
                return True
            except DockerException:
                return False
            return container.status not in ACTIVE_STATES and container.status != "removing"

    def training_metadata(self, workload_id: UUID, *, completed: bool = False) -> dict[str, Any]:
        container = self._managed_container(self._workload_name("training", workload_id))
        raw_output = container.labels.get(OUTPUT_PATH_LABEL)
        algorithm = container.labels.get(TRAINING_ALGORITHM_LABEL)
        if not raw_output or algorithm not in {"freeze", "lora", "qlora"}:
            raise InvalidWorkload("训练容器缺少受控输出或算法标签")
        output_path = self._training_output_path(
            Path(raw_output), workload_id, create=False, require_empty=False
        )
        metadata: dict[str, Any] = {}
        checkpoint_directories: list[tuple[int, Path]] = []
        for candidate in output_path.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-([1-9][0-9]*)", candidate.name)
            if match is None:
                continue
            try:
                mode = candidate.lstat().st_mode
            except OSError:
                continue
            if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                checkpoint_directories.append((int(match.group(1)), candidate))

        # 根状态通常只在结束时出现；运行中从最大 numeric checkpoint 回退，并按
        # global_step 选择更新的一份，避免根文件存在但落后时进度倒退。
        state_sources = [output_path / "trainer_state.json"]
        state_sources.extend(
            candidate / "trainer_state.json" for _, candidate in sorted(checkpoint_directories, reverse=True)
        )
        states: list[dict[str, Any]] = []
        state_error: InvalidWorkload | None = None
        for index, state_path in enumerate(state_sources):
            try:
                value = self._safe_training_json(state_path, maximum_bytes=2 * 1024 * 1024)
            except InvalidWorkload as exc:
                # Trainer 非原子重写时可能短暂暴露空/部分 JSON。运行态忽略本轮；
                # 成功态的根状态损坏，或没有任何可用状态时，才严格失败。
                if completed and index == 0:
                    raise
                state_error = exc
                continue
            if value is not None:
                states.append(value)
        if completed and not states and state_error is not None:
            raise state_error
        state = max(
            states,
            key=lambda item: (
                item.get("global_step")
                if isinstance(item.get("global_step"), int) and not isinstance(item.get("global_step"), bool)
                else -1
            ),
            default=None,
        )
        if state is not None:
            current_step = state.get("global_step")
            total_steps = state.get("max_steps")
            if isinstance(current_step, int) and not isinstance(current_step, bool) and current_step >= 0:
                metadata["current_step"] = current_step
            if isinstance(total_steps, int) and not isinstance(total_steps, bool) and total_steps > 0:
                metadata["total_steps"] = total_steps
                if isinstance(current_step, int) and not isinstance(current_step, bool):
                    metadata["progress"] = min(100.0, max(0.0, current_step * 100.0 / total_steps))
            history = state.get("log_history")
            if isinstance(history, list):
                last_metrics = next(
                    (item for item in reversed(history) if isinstance(item, dict)),
                    None,
                )
                if last_metrics is not None:
                    metadata["metrics"] = {
                        str(key): value
                        for key, value in last_metrics.items()
                        if isinstance(value, str | int | bool)
                        or (isinstance(value, float) and math.isfinite(value))
                    }

        # 运行中/失败任务只返回有界监控值。路径只有在 exit 0 后完成整套产物
        # 校验才会上报，避免控制面把尚在写入或仅部分生成的目录当作可部署资产。
        if not completed:
            return metadata

        valid_checkpoints: list[tuple[int, Path]] = []
        for step, candidate in checkpoint_directories:
            try:
                valid = validate_checkpoint_directory(candidate)
            except (OSError, RuntimeError, TrainingArtifactError):
                continue
            valid_checkpoints.append((step, valid))
        if valid_checkpoints:
            metadata["checkpoint_path"] = str(max(valid_checkpoints, key=lambda item: item[0])[1])

        try:
            if algorithm in {"lora", "qlora"}:
                adapter = validate_adapter_directory(output_path)
                metadata["adapter_path"] = str(adapter)
                merged = validate_full_model_directory(output_path / "merged")
                metadata["merged_model_path"] = str(merged)
            else:
                deployable = validate_full_model_directory(output_path)
                # 保留控制面既有字段；Freeze 没有合并步骤，该路径即完整可部署模型。
                metadata["merged_model_path"] = str(deployable)
        except (OSError, RuntimeError, TrainingArtifactError) as exc:
            raise InvalidWorkload(f"训练成功退出但产物校验失败：{exc}") from exc
        return metadata

    def evaluation_metadata(self, workload_id: UUID) -> dict[str, Any]:
        container = self._managed_container(self._workload_name("evaluation", workload_id))
        raw_output = container.labels.get(OUTPUT_PATH_LABEL)
        raw_manifest = container.labels.get(DATASET_MANIFEST_PATH_LABEL)
        if not raw_output or not raw_manifest:
            raise InvalidWorkload("评测容器缺少受控产物标签")
        base_template = container.labels.get(BASE_TEMPLATE_LABEL)
        candidate_template = container.labels.get(CANDIDATE_TEMPLATE_LABEL)
        if base_template not in {"base", "instruct"} or candidate_template not in {
            "base",
            "instruct",
        }:
            raise InvalidWorkload("评测容器缺少可信模板标签")
        try:
            output_path, _ = strict_existing_path(
                Path(raw_output),
                (self.settings.evaluation_output_root,),
                directory=True,
            )
            expected_output = self.settings.evaluation_output_root.resolve(strict=True) / str(workload_id)
            if output_path != expected_output:
                raise EvaluationInputError("评测产物目录与 run UUID 不一致")
            manifest_path, _ = strict_existing_path(
                Path(raw_manifest),
                (self.settings.runtime_root,),
                directory=False,
            )
            dataset_sha256, record_count = load_dataset_manifest_summary(manifest_path)
            metadata = load_pair_report_metadata(
                output_path,
                expected_dataset_sha256=dataset_sha256,
                expected_total=record_count,
                expected_base_template=base_template,
                expected_candidate_template=candidate_template,
            )
        except EvaluationInputError as exc:
            raise InvalidWorkload(f"评测产物无效：{exc}") from exc
        metadata["dataset_manifest_path"] = str(manifest_path)
        return metadata

    def launch_inference(self, request: InferenceLaunchRequest) -> WorkloadInfo:
        if request.image not in self.settings.vllm_images:
            raise InvalidWorkload("推理镜像未命中 VLLM_ALLOWED_IMAGES 白名单")
        verified_image_id = self._verified_vllm_image_id(request.image)
        model_path = self._existing_path(request.model_path, self.settings.model_root, directory=True)
        gpu_ids = self._validate_gpu_ids(request.gpu_ids)
        name = self._workload_name("inference", request.deployment_id)
        cache_path = self._writable_task_path("inference", request.deployment_id)
        command = self._vllm_command(request, gpu_ids)
        environment = self._safe_environment(request.environment, INFERENCE_ENV_ALLOWLIST)
        environment.update(self._offline_environment())
        environment.update(
            {
                "HF_HOME": "/workspace/cache/huggingface",
                "TRITON_CACHE_DIR": "/workspace/cache/triton",
                "TORCHINDUCTOR_CACHE_DIR": "/workspace/cache/torchinductor",
                "VLLM_CACHE_ROOT": "/workspace/cache/vllm",
                "XDG_CACHE_HOME": "/workspace/cache/xdg",
                "VLLM_NO_USAGE_STATS": "1",
            }
        )
        endpoint = f"http://{name}:{request.port}"
        # Docker healthcheck 同时承担 readiness 与持续 liveness 探测。仅进程处于
        # running 不能说明 vLLM 已完成模型加载，也不能说明已加载服务仍可响应。
        healthcheck = {
            "test": [
                "CMD",
                "python",
                "-c",
                (
                    "import urllib.request; "
                    f"urllib.request.urlopen('http://127.0.0.1:{request.port}/health', timeout=3).close()"
                ),
            ],
            "interval": INFERENCE_HEALTH_INTERVAL_SECONDS * 1_000_000_000,
            "timeout": 4 * 1_000_000_000,
            "retries": 3,
            # 首次加载窗口与控制面判断使用同一配置；成功检查仍可提前变为 healthy。
            "start_period": self.settings.inference_startup_timeout_seconds * 1_000_000_000,
        }

        with self._allocation_lock:
            self._assert_name_available(name)
            self._assert_gpus_available(gpu_ids)
            container = self._run_container(
                name=name,
                # 白名单校验后使用不可变 image ID，避免 tag 在窗口期被替换。
                image=verified_image_id,
                command=command,
                gpu_ids=gpu_ids,
                kind="inference",
                workload_id=request.deployment_id,
                environment=environment,
                volumes={
                    str(model_path): {"bind": "/workspace/model", "mode": "ro"},
                    str(cache_path): {"bind": "/workspace/cache", "mode": "rw"},
                },
                endpoint=endpoint,
                service_type=request.service_type,
                generation=request.generation,
                owner_type="deployment",
                port=request.port,
                healthcheck=healthcheck,
                restart_policy={
                    "Name": "on-failure",
                    "MaximumRetryCount": INFERENCE_MAX_RESTARTS,
                },
            )
        return self._to_info(container)

    def launch_training(self, request: TrainingLaunchRequest) -> WorkloadInfo:
        if request.image not in self.settings.llamafactory_images:
            raise InvalidWorkload("训练镜像未命中 LLAMAFACTORY_ALLOWED_IMAGES 白名单")
        verified_image_id = self._verified_training_image_id(request.image)
        try:
            model_path, _ = strict_existing_path(
                request.model_path, (self.settings.model_root,), directory=True
            )
            dataset_path, _ = strict_existing_path(
                request.dataset_path, (self.settings.dataset_root,), directory=False
            )
            config_path, _ = strict_existing_path(
                request.config_path, (self.settings.training_config_root,), directory=False
            )
            dataset_dir, _ = strict_existing_path(
                request.dataset_dir, (self.settings.runtime_root,), directory=True
            )
        except EvaluationInputError as exc:
            raise InvalidWorkload(str(exc).replace("评测", "训练")) from exc
        if dataset_path.suffix.casefold() != ".jsonl":
            raise InvalidWorkload("训练数据必须是 .jsonl 普通文件")
        output_path = self._training_output_path(
            request.output_path, request.job_id, create=True, require_empty=True
        )
        try:
            validate_training_config(
                TrainingSpec(
                    config_path=config_path,
                    model_path=WORKSPACE_MODEL,
                    dataset_dir=WORKSPACE_DATASET,
                    output_path=WORKSPACE_OUTPUT,
                    stage=TrainingStage(request.stage),
                    algorithm=TrainingAlgorithm(request.algorithm),
                    dataset_format=DatasetFormat(request.dataset_format),
                )
            )
        except (TrainingRuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"节点派生训练配置无效：{exc}") from exc
        gpu_ids = self._validate_gpu_ids(request.gpu_ids)
        name = self._workload_name("training", request.job_id)
        cache_path = self._writable_task_path("training", request.job_id)
        environment = self._safe_environment(request.environment, TRAINING_ENV_ALLOWLIST)
        environment.update(self._offline_environment())
        environment.update(
            {
                "HF_HOME": str(WORKSPACE_CACHE / "huggingface"),
                "TRITON_CACHE_DIR": str(WORKSPACE_CACHE / "triton"),
                "XDG_CACHE_HOME": str(WORKSPACE_CACHE / "xdg"),
                "WANDB_DISABLED": "true",
                "DISABLE_VERSION_CHECK": "1",
                # 单机多卡显式固定 world size；容器只看得到整卡租约中的 GPU。
                "FORCE_TORCHRUN": "1" if len(gpu_ids) > 1 else "0",
                "NNODES": "1",
                "NODE_RANK": "0",
                "NPROC_PER_NODE": str(len(gpu_ids)),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                # 目标 RTX 4090D 无 NVLink/IB，默认使用本机 socket/共享内存路径。
                "NCCL_IB_DISABLE": "1",
                "NCCL_P2P_DISABLE": "1",
            }
        )

        with self._allocation_lock:
            self._assert_name_available(name)
            self._assert_gpus_available(gpu_ids)
            volumes = {
                str(model_path): {"bind": str(WORKSPACE_MODEL), "mode": "ro"},
                str(dataset_path): {"bind": str(WORKSPACE_DATA_FILE), "mode": "ro"},
                str(dataset_dir): {"bind": str(WORKSPACE_DATASET), "mode": "ro"},
                str(config_path): {"bind": str(WORKSPACE_CONFIG), "mode": "ro"},
                str(output_path): {"bind": str(WORKSPACE_OUTPUT), "mode": "rw"},
                str(cache_path): {"bind": str(WORKSPACE_CACHE), "mode": "rw"},
            }
            container = self._run_container(
                name=name,
                # 使用刚完成标签校验的不可变 image ID，避免校验后 tag 被替换。
                image=verified_image_id,
                command=[
                    "run",
                    "--config",
                    str(WORKSPACE_CONFIG),
                    "--model-path",
                    str(WORKSPACE_MODEL),
                    "--dataset-dir",
                    str(WORKSPACE_DATASET),
                    "--output-dir",
                    str(WORKSPACE_OUTPUT),
                    "--stage",
                    request.stage,
                    "--algorithm",
                    request.algorithm,
                    "--dataset-format",
                    request.dataset_format,
                ],
                gpu_ids=gpu_ids,
                kind="training",
                workload_id=request.job_id,
                environment=environment,
                volumes=volumes,
                generation=request.generation,
                owner_type="training",
                output_path=output_path,
                training_algorithm=request.algorithm,
                restart_policy={"Name": "no"},
                network_mode="none",
            )
        return self._to_info(container)

    def launch_evaluation(self, request: EvaluationLaunchRequest) -> WorkloadInfo:
        if request.image not in self.settings.evaluation_images:
            raise InvalidWorkload("评测镜像未命中 EVALUATION_ALLOWED_IMAGES 白名单")
        verified_image_id = self._verified_evaluation_image_id(request.image)
        try:
            baseline_path, _ = strict_existing_path(
                request.baseline_model_path,
                (self.settings.model_root,),
                directory=True,
            )
            candidate_path, _ = strict_existing_path(
                request.candidate_model_path,
                (self.settings.model_root,),
                directory=True,
            )
            dataset_path, _ = strict_existing_path(
                request.dataset_path,
                (self.settings.runtime_root,),
                directory=False,
            )
            manifest_path, _ = strict_existing_path(
                request.dataset_manifest_path,
                (self.settings.runtime_root,),
                directory=False,
            )
            if (
                dataset_path.name != "evaluation.jsonl"
                or manifest_path.name != "dataset-manifest.json"
                or dataset_path.parent != manifest_path.parent
            ):
                raise EvaluationInputError("评测合并数据与 manifest 必须位于同一系统派生目录")
            output_path = prepare_output_directory(
                request.output_path,
                self.settings.evaluation_output_root,
                request.run_id,
            )
        except EvaluationInputError as exc:
            raise InvalidWorkload(str(exc)) from exc
        gpu_ids = self._validate_gpu_ids(request.gpu_ids)
        if request.tensor_parallel_size != len(gpu_ids):
            raise InvalidWorkload("评测 tensor_parallel_size 必须等于 gpu_ids 数量")
        name = self._workload_name("evaluation", request.run_id)
        baseline_container_path = "/workspace/models/baseline"
        candidate_container_path = "/workspace/models/candidate"
        volumes = {
            str(baseline_path): {"bind": baseline_container_path, "mode": "ro"},
            str(dataset_path.parent): {"bind": "/workspace/dataset", "mode": "ro"},
            str(output_path): {"bind": "/workspace/output", "mode": "rw"},
        }
        if baseline_path == candidate_path:
            candidate_container_path = baseline_container_path
        else:
            volumes[str(candidate_path)] = {"bind": candidate_container_path, "mode": "ro"}
        command = [
            "run-pair",
            "--dataset",
            "/workspace/dataset/evaluation.jsonl",
            "--output-dir",
            "/workspace/output",
            "--baseline-path",
            baseline_container_path,
            "--baseline-name",
            "baseline",
            "--baseline-template",
            request.base_template,
            "--candidate-path",
            candidate_container_path,
            "--candidate-name",
            "candidate",
            "--candidate-template",
            request.candidate_template,
            "--tensor-parallel-size",
            str(request.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(request.gpu_memory_utilization),
            "--concurrency",
            str(request.concurrency),
            "--max-tokens",
            str(request.max_tokens),
        ]
        environment = self._offline_environment()
        environment.update(
            {
                "HF_HOME": "/tmp/huggingface",
                "TRITON_CACHE_DIR": "/tmp/triton",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
                "VLLM_CACHE_ROOT": "/tmp/vllm",
                "XDG_CACHE_HOME": "/tmp/xdg",
                "VLLM_NO_USAGE_STATS": "1",
            }
        )

        with self._allocation_lock:
            self._assert_name_available(name)
            self._assert_gpus_available(gpu_ids)
            container = self._run_container(
                name=name,
                image=verified_image_id,
                command=command,
                gpu_ids=gpu_ids,
                kind="evaluation",
                workload_id=request.run_id,
                environment=environment,
                volumes=volumes,
                generation=request.generation,
                owner_type="evaluation",
                output_path=output_path,
                dataset_manifest_path=manifest_path,
                base_template=request.base_template,
                candidate_template=request.candidate_template,
                restart_policy={"Name": "no"},
                network_mode="none",
            )
        return self._to_info(container)

    def _verified_vllm_image_id(self, reference: str) -> str:
        try:
            image = self.client.images.get(reference)
        except ImageNotFound as exc:
            raise InvalidWorkload("vLLM 镜像尚未预拉取；节点禁止任务请求隐式拉取动态镜像") from exc
        if not isinstance(image.id, str) or not image.id.startswith("sha256:"):
            raise InvalidWorkload("Docker 未返回可验证的 vLLM 镜像 ID")
        return image.id

    def _verified_training_image_id(self, reference: str) -> str:
        try:
            image = self.client.images.get(reference)
        except ImageNotFound as exc:
            raise InvalidWorkload(
                "训练镜像尚未预构建/预拉取；节点禁止在任务请求中隐式拉取高权限训练镜像"
            ) from exc
        try:
            validate_hardening_labels(image.labels)
        except UnsafeTrainingImage as exc:
            raise InvalidWorkload(str(exc)) from exc
        if not isinstance(image.id, str) or not image.id.startswith("sha256:"):
            raise InvalidWorkload("Docker 未返回可验证的训练镜像 ID")
        return image.id

    def _verified_evaluation_image_id(self, reference: str) -> str:
        try:
            image = self.client.images.get(reference)
        except ImageNotFound as exc:
            raise InvalidWorkload("评测镜像尚未预构建/预拉取；节点禁止在任务请求中隐式拉取评测镜像") from exc
        try:
            validate_evaluation_image_labels(image.labels)
        except UnsafeEvaluationImage as exc:
            raise InvalidWorkload(str(exc)) from exc
        if not isinstance(image.id, str) or not image.id.startswith("sha256:"):
            raise InvalidWorkload("Docker 未返回可验证的评测镜像 ID")
        return image.id

    def start(self, name: str) -> WorkloadInfo:
        with self._allocation_lock:
            container = self._managed_container(name)
            info = self._to_info(container)
            if info.status in {"running", "restarting"}:
                return info
            if info.status == "paused":
                raise WorkloadConflict("工作负载处于 paused 状态，禁止用 start 隐式改变状态")
            self._assert_gpus_available(info.gpu_ids, exclude_name=name)
            container.start()
            return self._to_info(container)

    def stop(self, name: str, timeout_seconds: int) -> WorkloadInfo:
        with self._allocation_lock:
            container = self._managed_container(name)
            info = self._to_info(container)
            if info.status not in ACTIVE_STATES:
                return info
            container.stop(timeout=timeout_seconds)
            return self._to_info(container)

    def delete(self, name: str, force: bool = False) -> None:
        with self._allocation_lock:
            container = self._managed_container(name)
            container.reload()
            if container.status in ACTIVE_STATES and not force:
                raise WorkloadConflict("工作负载仍在运行，请先停止或显式使用 force")
            container.remove(force=force, v=True)

    def logs(self, name: str, tail: int) -> str:
        output = self._managed_container(name).logs(stdout=True, stderr=True, tail=tail)
        return output.decode("utf-8", errors="replace")

    def gpu_allocations(self) -> dict[int, str]:
        allocations: dict[int, str] = {}
        for container in self.client.containers.list(all=True, filters={"label": f"{MANAGED_LABEL}=true"}):
            container.reload()
            if container.status not in ACTIVE_STATES:
                continue
            for gpu_id in self._gpu_ids_from_labels(container.labels):
                allocations[gpu_id] = container.name
        return allocations

    def _run_container(
        self,
        *,
        name: str,
        image: str,
        command: list[str],
        gpu_ids: list[int],
        kind: str,
        workload_id: UUID,
        environment: dict[str, str],
        volumes: dict[str, dict[str, str]],
        restart_policy: dict[str, str],
        endpoint: str | None = None,
        service_type: str | None = None,
        generation: int = 1,
        owner_type: str | None = None,
        output_path: Path | None = None,
        dataset_manifest_path: Path | None = None,
        base_template: str | None = None,
        candidate_template: str | None = None,
        training_algorithm: str | None = None,
        port: int | None = None,
        healthcheck: dict[str, Any] | None = None,
        entrypoint: list[str] | None = None,
        network_mode: str | None = None,
    ) -> Container:
        labels = {
            MANAGED_LABEL: "true",
            KIND_LABEL: kind,
            ID_LABEL: str(workload_id),
            GPU_LABEL: ",".join(str(item) for item in gpu_ids),
            GENERATION_LABEL: str(generation),
        }
        if owner_type:
            labels[OWNER_TYPE_LABEL] = owner_type
        if endpoint:
            labels[ENDPOINT_LABEL] = endpoint
        if service_type:
            labels[SERVICE_TYPE_LABEL] = service_type
        if output_path:
            labels[OUTPUT_PATH_LABEL] = str(output_path)
        if dataset_manifest_path:
            labels[DATASET_MANIFEST_PATH_LABEL] = str(dataset_manifest_path)
        if base_template:
            labels[BASE_TEMPLATE_LABEL] = base_template
        if candidate_template:
            labels[CANDIDATE_TEMPLATE_LABEL] = candidate_template
        if training_algorithm:
            labels[TRAINING_ALGORITHM_LABEL] = training_algorithm
        if port is not None:
            labels[PORT_LABEL] = str(port)
        try:
            network_arguments = (
                {"network_mode": network_mode}
                if network_mode is not None
                else {"network": self.settings.runtime_network}
            )
            health_arguments = {"healthcheck": healthcheck} if healthcheck is not None else {}
            return self.client.containers.run(
                image=image,
                name=name,
                command=command,
                entrypoint=entrypoint,
                detach=True,
                **network_arguments,
                **health_arguments,
                device_requests=[
                    DeviceRequest(
                        device_ids=[str(item) for item in gpu_ids],
                        capabilities=[["gpu"]],
                    )
                ],
                environment=environment,
                volumes=volumes,
                labels=labels,
                user=f"{self.settings.workload_uid}:{self.settings.workload_gid}",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                init=True,
                shm_size=self.settings.workload_shm_size,
                tmpfs={"/tmp": "rw,nosuid,nodev,size=4g,mode=1777"},
                restart_policy=restart_policy,
                log_config={
                    "type": "local",
                    "config": {"max-size": "20m", "max-file": "5"},
                },
            )
        except APIError as exc:
            raise RunnerError(f"Docker 创建工作负载失败：{exc.explanation}") from exc

    def _vllm_command(self, request: InferenceLaunchRequest, gpu_ids: list[int]) -> list[str]:
        normalized: dict[str, Any] = {}
        for raw_key, value in request.vllm_args.items():
            key = raw_key.strip().lower().replace("_", "-")
            if key not in VLLM_ARGUMENT_ALLOWLIST:
                raise InvalidWorkload(f"不允许的 vLLM 参数：{raw_key}")
            if isinstance(value, str) and (len(value) > 2048 or any(ord(char) < 32 for char in value)):
                raise InvalidWorkload(f"vLLM 参数 {raw_key} 包含非法字符或过长")
            self._validate_vllm_argument(key, value)
            normalized[key] = value

        # GPU 数量和模型路径由调度器/受控目录决定，不能被详细参数覆盖。
        task_arguments = (
            ["--runner", "pooling", "--convert", "embed"]
            if request.service_type == "embedding"
            else ["--runner", "generate"]
        )
        command = [
            "--host",
            "0.0.0.0",
            "--port",
            str(request.port),
            "--model",
            "/workspace/model",
            "--served-model-name",
            request.served_model_name,
            *task_arguments,
            "--tensor-parallel-size",
            str(len(gpu_ids)),
            # 资产层只接收 Safetensors；这里再次固定加载器，避免回退到 pickle 权重。
            "--load-format",
            "safetensors",
        ]
        for key in sorted(normalized):
            value = normalized[key]
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
                continue
            command.extend((f"--{key}", str(value)))
        return command

    @staticmethod
    def _validate_vllm_argument(key: str, value: Any) -> None:
        if key in VLLM_BOOLEAN_ARGUMENTS:
            if not isinstance(value, bool):
                raise InvalidWorkload(f"vLLM 参数 {key} 必须是布尔值")
            return
        if key in VLLM_POSITIVE_INTEGER_ARGUMENTS:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise InvalidWorkload(f"vLLM 参数 {key} 必须是正整数")
            if value > VLLM_POSITIVE_INTEGER_LIMITS[key]:
                raise InvalidWorkload(f"vLLM 参数 {key} 超出节点安全上限 {VLLM_POSITIVE_INTEGER_LIMITS[key]}")
            return
        if key in VLLM_NONNEGATIVE_INTEGER_ARGUMENTS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidWorkload(f"vLLM 参数 {key} 必须是非负整数")
            upper_bound = 100 if key == "max-logprobs" else 2**32 - 1
            if value > upper_bound:
                raise InvalidWorkload(f"vLLM 参数 {key} 超出节点安全上限 {upper_bound}")
            return
        if key in VLLM_NONNEGATIVE_NUMBER_ARGUMENTS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise InvalidWorkload(f"vLLM 参数 {key} 必须是非负数")
            if value > 16:
                raise InvalidWorkload(f"vLLM 参数 {key} 超出每 GPU 安全上限 16 GiB")
            return
        if key == "block-size":
            if isinstance(value, bool) or not isinstance(value, int) or value not in {8, 16, 32}:
                raise InvalidWorkload("block-size 在 CUDA 节点仅允许 8、16 或 32")
            return
        if key == "gpu-memory-utilization":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= value <= 0.98:
                raise InvalidWorkload("gpu-memory-utilization 必须位于 0.1..0.98")
            return
        if key == "pipeline-parallel-size":
            if isinstance(value, bool) or not isinstance(value, int) or value != 1:
                raise InvalidWorkload("首版单机调度固定 pipeline-parallel-size=1")
            return
        allowed_values = VLLM_ENUM_ARGUMENTS.get(key)
        if allowed_values is not None:
            if not isinstance(value, str) or value.lower() not in allowed_values:
                raise InvalidWorkload(f"vLLM 参数 {key} 仅允许：{', '.join(sorted(allowed_values))}")
            return
        if key == "quantization":
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
                raise InvalidWorkload(f"vLLM 参数 {key} 的值格式不安全")
            return
        raise InvalidWorkload(f"vLLM 参数 {key} 缺少节点侧类型约束")

    def _existing_path(self, candidate: Path, root: Path, *, directory: bool | None) -> Path:
        try:
            root_real = root.resolve(strict=True)
            path_real = candidate.resolve(strict=True)
            path_real.relative_to(root_real)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"路径不存在或越出受控目录：{candidate}") from exc
        if directory is True and not path_real.is_dir():
            raise InvalidWorkload(f"路径必须是目录：{candidate}")
        if directory is False and not path_real.is_file():
            raise InvalidWorkload(f"路径必须是文件：{candidate}")
        return path_real

    @staticmethod
    def _controlled_path(candidate: Path, root: Path, *, must_exist: bool) -> Path:
        try:
            root_real = root.resolve(strict=True)
            candidate_real = candidate.resolve(strict=must_exist)
            candidate_real.relative_to(root_real)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"路径越出受控目录：{candidate}") from exc
        return candidate_real

    def _training_output_path(
        self,
        candidate: Path,
        workload_id: UUID,
        *,
        create: bool,
        require_empty: bool,
    ) -> Path:
        raw_root = Path(os.path.abspath(self.settings.checkpoint_root))
        raw_candidate = Path(os.path.abspath(candidate))
        expected = raw_root / str(workload_id)
        if not candidate.is_absolute() or raw_candidate != expected:
            raise InvalidWorkload(f"训练输出目录必须由系统派生为：{expected}")
        try:
            root_info = raw_root.lstat()
            if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
                raise InvalidWorkload("checkpoint 根目录必须是非软链接目录")
            resolved_root = raw_root.resolve(strict=True)
            if raw_candidate.exists() or raw_candidate.is_symlink():
                candidate_info = raw_candidate.lstat()
                if stat.S_ISLNK(candidate_info.st_mode) or not stat.S_ISDIR(candidate_info.st_mode):
                    raise InvalidWorkload("训练输出必须是非软链接目录")
                if require_empty and any(raw_candidate.iterdir()):
                    raise InvalidWorkload("新的训练任务不能复用非空输出目录")
            elif create:
                raw_candidate.mkdir(mode=0o700)
            else:
                raise InvalidWorkload("训练输出目录不存在")
            raw_candidate.chmod(0o700)
            resolved = raw_candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if resolved != resolved_root / str(workload_id):
                raise InvalidWorkload("训练输出目录解析后与任务 UUID 不一致")
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"无法准备受控训练输出目录：{candidate}") from exc
        return resolved

    @staticmethod
    def _safe_training_json(path: Path, *, maximum_bytes: int) -> dict[str, Any] | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise InvalidWorkload("无法读取训练状态") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InvalidWorkload("训练状态必须是非软链接普通文件")
        if not 1 <= info.st_size <= maximum_bytes:
            raise InvalidWorkload("训练状态文件大小无效")

        def reject_constant(value: str) -> None:
            raise ValueError(f"非有限数值：{value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"重复字段：{key}")
                result[key] = value
            return result

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as source:
                raw = source.read(maximum_bytes + 1)
            value = json.loads(
                raw,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidWorkload("训练状态不是有效的有限 UTF-8 JSON") from exc
        if len(raw) > maximum_bytes or not isinstance(value, dict):
            raise InvalidWorkload("训练状态必须是有界 JSON 对象")
        return value

    def _writable_task_path(self, kind: str, workload_id: UUID) -> Path:
        path = (self.settings.runtime_root / kind / str(workload_id)).resolve(strict=False)
        root = self.settings.runtime_root.resolve(strict=True)
        try:
            path.relative_to(root)
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            raise InvalidWorkload("无法创建任务缓存目录") from exc
        return path.resolve(strict=True)

    def _validate_gpu_ids(self, gpu_ids: Iterable[int]) -> list[int]:
        submitted = list(gpu_ids)
        normalized = sorted(set(submitted))
        if not normalized or len(normalized) != len(submitted):
            raise InvalidWorkload("GPU 编号不能为空或重复")
        invalid = [item for item in normalized if item < 0 or item >= self.settings.gpu_count]
        if invalid:
            raise InvalidWorkload(f"GPU 编号 {invalid} 超出配置范围 0..{self.settings.gpu_count - 1}")
        return normalized

    def _assert_gpus_available(self, gpu_ids: list[int], exclude_name: str | None = None) -> None:
        allocations = self.gpu_allocations()
        conflicts = {
            gpu_id: allocations[gpu_id]
            for gpu_id in gpu_ids
            if gpu_id in allocations and allocations[gpu_id] != exclude_name
        }
        if conflicts:
            detail = ", ".join(f"GPU {gpu}: {name}" for gpu, name in conflicts.items())
            raise WorkloadConflict(f"整卡独占冲突：{detail}")
        if self.settings.enforce_nvml_process_check:
            try:
                externally_busy = read_busy_gpu_ids(gpu_ids)
            except NVMLUnavailable as exc:
                raise RunnerError("NVML 进程检查不可用，为避免 GPU 重复分配已拒绝启动") from exc
            if externally_busy:
                formatted = ", ".join(str(item) for item in sorted(externally_busy))
                raise WorkloadConflict(f"GPU {formatted} 存在非受管进程，不能执行整卡独占分配")

    def _assert_name_available(self, name: str) -> None:
        try:
            self.client.containers.get(name)
        except NotFound:
            return
        raise WorkloadConflict(f"工作负载已存在：{name}")

    def _managed_container(self, name: str) -> Container:
        if not SAFE_NAME.fullmatch(name):
            raise WorkloadNotFound("工作负载不存在")
        try:
            container = self.client.containers.get(name)
        except NotFound as exc:
            raise WorkloadNotFound("工作负载不存在") from exc
        if container.labels.get(MANAGED_LABEL) != "true":
            raise WorkloadNotFound("工作负载不存在")
        return container

    @staticmethod
    def _workload_name(kind: str, workload_id: UUID) -> str:
        return f"openllmops-{kind}-{workload_id}"

    @staticmethod
    def _safe_environment(values: dict[str, str], allowlist: frozenset[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in values.items():
            if key not in allowlist:
                raise InvalidWorkload(f"不允许的环境变量：{key}")
            if len(value) > 1024 or "\x00" in value or "\n" in value or "\r" in value:
                raise InvalidWorkload(f"环境变量 {key} 的值不安全")
            result[key] = value
        return result

    @staticmethod
    def _offline_environment() -> dict[str, str]:
        # 模型先由资产模块导入，本地工作负载不应在运行时访问任意远端代码或模型。
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }

    @staticmethod
    def _gpu_ids_from_labels(labels: dict[str, str]) -> list[int]:
        try:
            return [int(item) for item in labels.get(GPU_LABEL, "").split(",") if item]
        except ValueError:
            return []

    @staticmethod
    def _generation_from_labels(labels: dict[str, str]) -> int:
        try:
            generation = int(labels.get(GENERATION_LABEL, "1"))
        except ValueError:
            return 1
        return max(1, generation)

    def _to_info(self, container: Container) -> WorkloadInfo:
        container.reload()
        labels = container.labels
        created_at = self._docker_datetime(container.attrs.get("Created"))
        state = container.attrs.get("State", {})
        health = state.get("Health") if isinstance(state, dict) else None
        raw_health_status = health.get("Status") if isinstance(health, dict) else None
        health_status = (
            raw_health_status if raw_health_status in {"starting", "healthy", "unhealthy"} else None
        )
        raw_failing_streak = health.get("FailingStreak", 0) if isinstance(health, dict) else 0
        health_failing_streak = (
            raw_failing_streak
            if isinstance(raw_failing_streak, int)
            and not isinstance(raw_failing_streak, bool)
            and raw_failing_streak >= 0
            else 0
        )
        raw_restart_count = container.attrs.get("RestartCount", 0)
        restart_count = (
            raw_restart_count
            if isinstance(raw_restart_count, int)
            and not isinstance(raw_restart_count, bool)
            and raw_restart_count >= 0
            else 0
        )
        started_at = self._docker_datetime(state.get("StartedAt") if isinstance(state, dict) else None)
        finished_at = self._docker_datetime(state.get("FinishedAt") if isinstance(state, dict) else None)
        return WorkloadInfo(
            name=container.name,
            workload_id=UUID(labels[ID_LABEL]),
            kind=labels[KIND_LABEL],
            image=container.image.tags[0] if container.image.tags else container.image.id,
            status=container.status,
            gpu_ids=self._gpu_ids_from_labels(labels),
            service_type=labels.get(SERVICE_TYPE_LABEL),
            endpoint=labels.get(ENDPOINT_LABEL),
            port=int(labels[PORT_LABEL]) if labels.get(PORT_LABEL, "").isdigit() else None,
            generation=self._generation_from_labels(labels),
            exit_code=state.get("ExitCode") if isinstance(state, dict) else None,
            health_status=health_status,
            health_failing_streak=health_failing_streak,
            restart_count=restart_count,
            started_at=started_at,
            finished_at=finished_at,
            created_at=created_at,
        )

    @staticmethod
    def _docker_datetime(raw: Any) -> datetime | None:
        if not isinstance(raw, str):
            return None
        try:
            # 项目最低 Python 3.11，fromisoformat 已原生接受 Docker 的尾部 Z。
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        # Docker 对尚未开始的容器使用 year=1 的零值时间，不能用于超时计算。
        return parsed if parsed.year > 1 else None


def docker_error_message(exc: DockerException) -> str:
    """去除可能含内部请求细节的异常串，只保留稳定错误类别。"""

    return f"Docker 服务不可用（{type(exc).__name__}）"
