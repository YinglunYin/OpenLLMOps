from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import docker
import yaml
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest

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
PORT_LABEL = "com.openllmops.port"

ACTIVE_STATES = frozenset({"created", "running", "restarting", "paused"})

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
    ) -> None:
        name = self.contract_workload_name(owner_type, workload_id)
        with self._allocation_lock:
            container = self._managed_container(name)
            container.reload()
            if self._generation_from_labels(container.labels) != generation:
                raise WorkloadConflict("停止命令 generation 与节点容器不一致")
            if container.status in ACTIVE_STATES:
                container.stop(timeout=timeout_seconds)
                container.reload()
            container.remove(force=False, v=True)

    def training_metadata(self, workload_id: UUID) -> dict[str, Any]:
        container = self._managed_container(self._workload_name("training", workload_id))
        raw_output = container.labels.get(OUTPUT_PATH_LABEL)
        if not raw_output:
            return {}
        output_path = self._controlled_path(Path(raw_output), self.settings.checkpoint_root, must_exist=False)
        metadata: dict[str, Any] = {}
        state_path = output_path / "trainer_state.json"
        if state_path.is_file() and state_path.stat().st_size <= 2 * 1024 * 1024:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                state = {}
            if isinstance(state, dict):
                current_step = state.get("global_step")
                total_steps = state.get("max_steps")
                if isinstance(current_step, int) and current_step >= 0:
                    metadata["current_step"] = current_step
                if isinstance(total_steps, int) and total_steps > 0:
                    metadata["total_steps"] = total_steps
                    if isinstance(current_step, int):
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
                            if isinstance(value, (str, int, bool))
                            or (isinstance(value, float) and math.isfinite(value))
                        }

        checkpoints = sorted(
            (
                item
                for item in output_path.glob("checkpoint-*")
                if item.is_dir() and item.resolve().is_relative_to(output_path)
            ),
            key=lambda item: item.stat().st_mtime,
        )
        if checkpoints:
            metadata["checkpoint_path"] = str(checkpoints[-1])
        adapter_candidates = [output_path, *(reversed(checkpoints))]
        adapter_path = next(
            (item for item in adapter_candidates if (item / "adapter_config.json").is_file()),
            None,
        )
        if adapter_path is not None:
            metadata["adapter_path"] = str(adapter_path)
        merged_path = output_path / "merged"
        if merged_path.is_dir():
            metadata["merged_model_path"] = str(merged_path)
        return metadata

    def evaluation_metadata(self, workload_id: UUID) -> dict[str, Any]:
        container = self._managed_container(self._workload_name("evaluation", workload_id))
        raw_output = container.labels.get(OUTPUT_PATH_LABEL)
        raw_manifest = container.labels.get(DATASET_MANIFEST_PATH_LABEL)
        if not raw_output or not raw_manifest:
            raise InvalidWorkload("评测容器缺少受控产物标签")
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
                restart_policy={"Name": "unless-stopped"},
            )
        return self._to_info(container)

    def launch_training(self, request: TrainingLaunchRequest) -> WorkloadInfo:
        if request.image not in self.settings.llamafactory_images:
            raise InvalidWorkload("训练镜像未命中 LLAMAFACTORY_ALLOWED_IMAGES 白名单")
        verified_image_id = self._verified_training_image_id(request.image)
        model_path = self._existing_path(request.model_path, self.settings.model_root, directory=True)
        dataset_path = self._existing_path(request.dataset_path, self.settings.dataset_root, directory=None)
        config_path = self._existing_path(
            request.config_path, self.settings.training_config_root, directory=False
        )
        output_path = self._output_path(request.output_path)
        if request.dataset_dir is None:
            if not dataset_path.is_dir():
                raise InvalidWorkload("JSONL 数据集任务必须提供受控 dataset_dir")
            dataset_dir = dataset_path
        else:
            dataset_dir = self._existing_path(request.dataset_dir, self.settings.runtime_root, directory=True)
        self._validate_training_config(config_path, model_path, dataset_dir, output_path)
        gpu_ids = self._validate_gpu_ids(request.gpu_ids)
        name = self._workload_name("training", request.job_id)
        cache_path = self._writable_task_path("training", request.job_id)
        environment = self._safe_environment(request.environment, TRAINING_ENV_ALLOWLIST)
        environment.update(self._offline_environment())
        environment.update(
            {
                "HF_HOME": str(cache_path / "huggingface"),
                "TRITON_CACHE_DIR": str(cache_path / "triton"),
            }
        )

        with self._allocation_lock:
            self._assert_name_available(name)
            self._assert_gpus_available(gpu_ids)
            volumes = {
                str(model_path): {"bind": str(model_path), "mode": "ro"},
                str(dataset_path): {"bind": str(dataset_path), "mode": "ro"},
                str(config_path): {"bind": str(config_path), "mode": "ro"},
                str(output_path): {"bind": str(output_path), "mode": "rw"},
                str(cache_path): {"bind": str(cache_path), "mode": "rw"},
            }
            volumes[str(dataset_dir)] = {"bind": str(dataset_dir), "mode": "ro"}
            container = self._run_container(
                name=name,
                # 使用刚完成标签校验的不可变 image ID，避免校验后 tag 被替换。
                image=verified_image_id,
                command=["llamafactory-cli", "train", str(config_path)],
                gpu_ids=gpu_ids,
                kind="training",
                workload_id=request.job_id,
                environment=environment,
                volumes=volumes,
                generation=request.generation,
                owner_type="training",
                output_path=output_path,
                restart_policy={"Name": "no"},
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
        port: int | None = None,
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
        if port is not None:
            labels[PORT_LABEL] = str(port)
        try:
            network_arguments = (
                {"network_mode": network_mode}
                if network_mode is not None
                else {"network": self.settings.runtime_network}
            )
            return self.client.containers.run(
                image=image,
                name=name,
                command=command,
                entrypoint=entrypoint,
                detach=True,
                **network_arguments,
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

    def _validate_training_config(
        self, config_path: Path, model_path: Path, dataset_path: Path, output_path: Path
    ) -> None:
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise InvalidWorkload(f"无法读取训练 YAML：{exc}") from exc
        if not isinstance(raw, dict):
            raise InvalidWorkload("训练 YAML 顶层必须是对象")
        if raw.get("trust_remote_code") not in (None, False):
            raise InvalidWorkload("训练任务永久禁止 trust_remote_code")
        if raw.get("stage") not in {"pt", "sft"}:
            raise InvalidWorkload("首版训练只允许 stage=pt 或 stage=sft")
        if raw.get("finetuning_type") not in {"lora", "freeze"}:
            raise InvalidWorkload("首版训练只允许 LoRA/QLoRA 或 Freeze")
        if raw.get("stage") == "pt" and raw.get("finetuning_type") != "lora":
            raise InvalidWorkload("继续预训练固定使用 LoRA，不允许 Freeze")
        if raw.get("quantization_bit") not in {None, 4}:
            raise InvalidWorkload("首版量化训练仅支持 4-bit QLoRA")

        self._assert_yaml_path(raw, "model_name_or_path", model_path)
        self._assert_yaml_path(raw, "dataset_dir", dataset_path)
        self._assert_yaml_path(raw, "output_dir", output_path)

    @staticmethod
    def _assert_yaml_path(config: dict[str, Any], key: str, expected: Path) -> None:
        value = config.get(key)
        if not isinstance(value, str) or Path(value).resolve(strict=False) != expected:
            raise InvalidWorkload(f"训练 YAML 的 {key} 必须等于任务受控路径 {expected}")

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

    def _output_path(self, candidate: Path) -> Path:
        try:
            root = self.settings.checkpoint_root.resolve(strict=True)
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root)
            resolved.mkdir(parents=True, exist_ok=True)
            final_path = resolved.resolve(strict=True)
            final_path.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"输出路径越出 checkpoint 受控目录：{candidate}") from exc
        return final_path

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
        created_at: datetime | None = None
        raw_created = container.attrs.get("Created")
        if isinstance(raw_created, str):
            try:
                # 项目最低 Python 3.11，fromisoformat 已原生接受 Docker 的尾部 Z。
                created_at = datetime.fromisoformat(raw_created)
            except ValueError:
                created_at = None
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
            exit_code=container.attrs.get("State", {}).get("ExitCode"),
            created_at=created_at,
        )


def docker_error_message(exc: DockerException) -> str:
    """去除可能含内部请求细节的异常串，只保留稳定错误类别。"""

    return f"Docker 服务不可用（{type(exc).__name__}）"
