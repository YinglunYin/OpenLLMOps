from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from openllmops_training_config import (
    Algorithm as TrainingAlgorithm,
)
from openllmops_training_config import (
    DatasetFormat,
    TrainingConfigError,
    TrainingHyperparameters,
    build_training_config,
)
from openllmops_training_config import (
    Stage as TrainingStage,
)
from openllmops_training_config import (
    TrainingRequest as SafeTrainingRequest,
)
from openllmops_training_runtime import (
    WORKSPACE_DATA_FILE,
    WORKSPACE_DATASET,
    WORKSPACE_MODEL,
    WORKSPACE_OUTPUT,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentWorkloadState,
    canonical_json,
)
from .config import Settings
from .docker_runner import (
    INFERENCE_HEALTH_INTERVAL_SECONDS,
    INFERENCE_MAX_RESTARTS,
    DockerRunner,
    InvalidWorkload,
    WorkloadConflict,
    WorkloadNotFound,
)
from .evaluation_runtime import (
    DatasetSource,
    EvaluationInputError,
    prepare_evaluation_workspace,
    strict_existing_path,
)
from .schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
    TrainingLaunchRequest,
    WorkloadInfo,
)

MAX_CACHED_REQUESTS = 2048
MAX_TRAINING_JSONL_LINE_BYTES = 16 * 1024 * 1024
STATE_VERSION = 1


class ExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TerminalCleanupExecution(ExecutionModel):
    cleanup_terminal: Literal[True]

    @field_validator("cleanup_terminal", mode="before")
    @classmethod
    def require_json_boolean_true(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError("cleanup_terminal 必须是 JSON 布尔值 true")
        return value


class VLLMExecution(ExecutionModel):
    runner: Literal["vllm"]
    service_type: Literal["generate", "embedding"]
    model_path: Path
    served_model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
    port: int | None = Field(default=None, ge=1024, le=65535)
    tensor_parallel_size: int = Field(ge=1, le=16)
    simplified_config: dict[str, Any] = Field(default_factory=dict)
    vllm_args: dict[str, Any] = Field(default_factory=dict)


class LLaMAFactoryExecution(ExecutionModel):
    runner: Literal["llamafactory"]
    model_path: Path
    dataset_path: Path
    stage: Literal["cpt", "sft"]
    algorithm: Literal["freeze", "lora", "qlora"]
    training_config: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    output_dir: Path


class EvaluationDatasetExecution(ExecutionModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    path: Path


class EvaluationExecution(ExecutionModel):
    runner: Literal["evaluation"]
    base_model_path: Path
    candidate_model_path: Path
    base_template: Literal["base", "instruct"]
    candidate_template: Literal["base", "instruct"]
    datasets: list[EvaluationDatasetExecution] = Field(min_length=1, max_length=16)
    output_dir: Path
    tensor_parallel_size: int = Field(ge=1, le=16)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=0.95)
    concurrency: int = Field(default=4, ge=1, le=32)
    max_tokens: int = Field(default=32, ge=1, le=512)

    @model_validator(mode="after")
    def unique_dataset_names(self) -> EvaluationExecution:
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("evaluation datasets.name 不能重复")
        return self


@dataclass(frozen=True)
class CommandResult:
    status_code: int
    response: AgentCommandResponse


@dataclass(frozen=True)
class CachedCommand:
    fingerprint: str
    result: CommandResult


class CommandStateStore:
    """持久化 generation 与有限 request_id 结果，agent 重启后仍能拒绝迟到命令。"""

    def __init__(self, runtime_root: Path) -> None:
        self._directory = runtime_root / "node-agent"
        self._path = self._directory / "command-state.json"
        self._generations: dict[str, int] = {}
        self._starts: dict[str, dict[str, Any]] = {}
        self._requests: dict[str, CachedCommand] = {}
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._load()

    @staticmethod
    def owner_key(owner_type: str, owner_id: UUID) -> str:
        return f"{owner_type}:{owner_id}"

    def generation(self, owner_type: str, owner_id: UUID) -> int:
        return self._generations.get(self.owner_key(owner_type, owner_id), 0)

    def advance_generation(self, owner_type: str, owner_id: UUID, generation: int) -> None:
        key = self.owner_key(owner_type, owner_id)
        if generation > self._generations.get(key, 0):
            self._generations[key] = generation
            self._persist()

    def bind_start(
        self,
        owner_type: str,
        owner_id: UUID,
        generation: int,
        fingerprint: str,
    ) -> None:
        key = self.owner_key(owner_type, owner_id)
        current = self._starts.get(key)
        if current is not None and current["generation"] == generation:
            if current["fingerprint"] != fingerprint:
                raise WorkloadConflict("同一 generation 已绑定到不同启动参数")
            # 兼容极端情况下由旧实现留下的“start 已绑定但水位未推进”状态。
            if generation > self._generations.get(key, 0):
                self._generations[key] = generation
                self._persist()
            return
        self._starts[key] = {"generation": generation, "fingerprint": fingerprint}
        # start 参数绑定与 generation 水位必须在同一次原子替换中落盘。若在两次
        # persist 之间宕机，重启后旧 start 可能越过尚未推进的水位并覆盖新代绑定。
        if generation > self._generations.get(key, 0):
            self._generations[key] = generation
        self._persist()

    def get_request(self, request_id: UUID) -> CachedCommand | None:
        return self._requests.get(str(request_id))

    def put_request(self, request_id: UUID, fingerprint: str, result: CommandResult) -> None:
        key = str(request_id)
        self._requests[key] = CachedCommand(fingerprint=fingerprint, result=result)
        while len(self._requests) > MAX_CACHED_REQUESTS:
            self._requests.pop(next(iter(self._requests)))
        self._persist()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if raw.get("version") != STATE_VERSION:
                raise ValueError("未知状态版本")
            generations = raw.get("generations")
            starts = raw.get("starts", {})
            requests = raw.get("requests")
            if (
                not isinstance(generations, dict)
                or not isinstance(starts, dict)
                or not isinstance(requests, dict)
            ):
                raise TypeError("状态结构无效")
            self._generations = {
                str(key): int(value)
                for key, value in generations.items()
                if isinstance(value, int) and value >= 1
            }
            self._starts = {
                str(key): {
                    "generation": int(value["generation"]),
                    "fingerprint": str(value["fingerprint"]),
                }
                for key, value in starts.items()
                if isinstance(value, dict)
                and isinstance(value.get("generation"), int)
                and value["generation"] >= 1
                and isinstance(value.get("fingerprint"), str)
            }
            # 旧文件若在两阶段持久化之间宕机，start 绑定本身仍是已观察代际的证据；
            # 以内存水位取两者最大值，避免重启窗口接受更旧命令。
            for key, start in self._starts.items():
                self._generations[key] = max(self._generations.get(key, 0), int(start["generation"]))
            for request_id, item in requests.items():
                if not isinstance(item, dict):
                    raise TypeError("request 缓存无效")
                result = CommandResult(
                    status_code=int(item["status_code"]),
                    response=AgentCommandResponse.model_validate(item["response"]),
                )
                self._requests[str(UUID(request_id))] = CachedCommand(
                    fingerprint=str(item["fingerprint"]), result=result
                )
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("node-agent 命令状态损坏，已拒绝降级启动") from exc

    def _persist(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "generations": self._generations,
            "starts": self._starts,
            "requests": {
                request_id: {
                    "fingerprint": cached.fingerprint,
                    "status_code": cached.result.status_code,
                    "response": cached.result.response.model_dump(mode="json"),
                }
                for request_id, cached in self._requests.items()
            },
        }
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)


class CommandProcessor:
    def __init__(
        self,
        settings: Settings,
        runner: DockerRunner,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.state = CommandStateStore(settings.runtime_root)
        self._lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(self, command: AgentCommand) -> CommandResult:
        fingerprint = hashlib.sha256(canonical_json(command)).hexdigest()
        with self._lock:
            current_generation = self.state.generation(command.owner.type, command.owner.id)
            if command.owner.generation < current_generation:
                return self._rejected(
                    command.request_id,
                    409,
                    "命令 generation 早于节点已观察到的 generation",
                    "stale_generation",
                )

            cached = self.state.get_request(command.request_id)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    return self._rejected(
                        command.request_id,
                        409,
                        "request_id 已绑定到不同命令",
                        "request_id_reused",
                    )
                return cached.result

            # 在执行副作用前持久化代际水位；崩溃重试可通过容器标签恢复幂等状态。
            if command.action == AgentAction.START:
                start_body = command.model_dump(mode="json", exclude={"request_id"})
                start_fingerprint = hashlib.sha256(canonical_json(start_body)).hexdigest()
                try:
                    self.state.bind_start(
                        command.owner.type,
                        command.owner.id,
                        command.owner.generation,
                        start_fingerprint,
                    )
                except WorkloadConflict as exc:
                    result = self._rejected(command.request_id, 409, str(exc), "generation_reused")
                    self.state.put_request(command.request_id, fingerprint, result)
                    return result
            if command.action != AgentAction.START:
                self.state.advance_generation(command.owner.type, command.owner.id, command.owner.generation)
            try:
                result = self._dispatch(command)
            except ValidationError as exc:
                result = self._rejected(
                    command.request_id,
                    422,
                    f"execution 参数无效：{exc.errors(include_url=False)}",
                    "invalid_execution",
                )
            except InvalidWorkload as exc:
                result = self._rejected(command.request_id, 422, str(exc), "invalid_workload")
            except WorkloadConflict as exc:
                result = self._rejected(command.request_id, 409, str(exc), "workload_conflict")
            self.state.put_request(command.request_id, fingerprint, result)
            return result

    def _dispatch(self, command: AgentCommand) -> CommandResult:
        if command.action == AgentAction.START:
            return self._start(command)
        if command.action == AgentAction.STOP:
            if command.execution:
                TerminalCleanupExecution.model_validate(command.execution)
                return self._cleanup(command)
            return self._stop(command)
        if command.execution:
            raise InvalidWorkload("status 命令的 execution 必须为空对象")
        return self._observe(command, accepted=True)

    def _start(self, command: AgentCommand) -> CommandResult:
        runner_name = command.execution.get("runner")
        expected_runner = {
            "deployment": "vllm",
            "training": "llamafactory",
            "evaluation": "evaluation",
        }[command.owner.type]
        if runner_name != expected_runner:
            raise InvalidWorkload(f"{command.owner.type} 只能使用 execution.runner={expected_runner}")
        existing = self.runner.prepare_contract_start(
            command.owner.type, command.owner.id, command.owner.generation
        )
        if existing is not None:
            return self._response_from_info(command, existing, accepted=True)

        if command.owner.type == "deployment":
            execution = VLLMExecution.model_validate(command.execution)
            if execution.tensor_parallel_size != len(command.resources.gpu_ids):
                raise InvalidWorkload("tensor_parallel_size 必须等于 gpu_ids 数量")
            arguments = self._merge_vllm_arguments(execution.simplified_config, execution.vllm_args)
            info = self.runner.launch_inference(
                InferenceLaunchRequest(
                    deployment_id=command.owner.id,
                    generation=command.owner.generation,
                    image=self.settings.vllm_runtime_image,
                    gpu_ids=command.resources.gpu_ids,
                    model_path=execution.model_path,
                    served_model_name=execution.served_model_name,
                    service_type=execution.service_type,
                    port=execution.port or 8000,
                    vllm_args=arguments,
                )
            )
        elif command.owner.type == "training":
            execution = LLaMAFactoryExecution.model_validate(command.execution)
            config_path, dataset_dir, dataset_format = self._materialize_training_files(command, execution)
            info = self.runner.launch_training(
                TrainingLaunchRequest(
                    job_id=command.owner.id,
                    generation=command.owner.generation,
                    image=self.settings.llamafactory_runtime_image,
                    gpu_ids=command.resources.gpu_ids,
                    model_path=execution.model_path,
                    dataset_path=execution.dataset_path,
                    dataset_dir=dataset_dir,
                    config_path=config_path,
                    output_path=execution.output_dir,
                    stage=execution.stage,
                    algorithm=execution.algorithm,
                    dataset_format=dataset_format.value,
                )
            )
        else:
            execution = EvaluationExecution.model_validate(command.execution)
            if execution.tensor_parallel_size != len(command.resources.gpu_ids):
                raise InvalidWorkload("tensor_parallel_size 必须等于 gpu_ids 数量")
            try:
                workspace = prepare_evaluation_workspace(
                    run_id=command.owner.id,
                    generation=command.owner.generation,
                    sources=[DatasetSource(dataset.name, dataset.path) for dataset in execution.datasets],
                    dataset_root=self.settings.dataset_root,
                    evaluation_dataset_root=self.settings.evaluation_dataset_root,
                    evaluation_output_root=self.settings.evaluation_output_root,
                    requested_output_path=execution.output_dir,
                    runtime_root=self.settings.runtime_root,
                )
            except EvaluationInputError as exc:
                raise InvalidWorkload(str(exc)) from exc
            info = self.runner.launch_evaluation(
                EvaluationLaunchRequest(
                    run_id=command.owner.id,
                    generation=command.owner.generation,
                    image=self.settings.evaluation_runtime_image,
                    gpu_ids=command.resources.gpu_ids,
                    baseline_model_path=execution.base_model_path,
                    candidate_model_path=execution.candidate_model_path,
                    dataset_path=workspace.dataset_path,
                    dataset_manifest_path=workspace.dataset_manifest_path,
                    output_path=workspace.output_path,
                    base_template=execution.base_template,
                    candidate_template=execution.candidate_template,
                    tensor_parallel_size=execution.tensor_parallel_size,
                    gpu_memory_utilization=execution.gpu_memory_utilization,
                    concurrency=execution.concurrency,
                    max_tokens=execution.max_tokens,
                )
            )
        return self._response_from_info(command, info, accepted=True)

    def _stop(self, command: AgentCommand) -> CommandResult:
        try:
            info = self.runner.stop_contract_workload(
                command.owner.type,
                command.owner.id,
                command.owner.generation,
            )
        except WorkloadNotFound:
            return self._accepted_absent(command.request_id)
        if info is not None:
            return self._response_from_info(command, info, accepted=True)
        return self._accepted_absent(command.request_id)

    def _cleanup(self, command: AgentCommand) -> CommandResult:
        """仅删除精确代次的终态容器；目标不存在时视为已完成清理。"""

        self.runner.cleanup_contract_workload(
            command.owner.type,
            command.owner.id,
            command.owner.generation,
        )
        return self._accepted_absent(command.request_id)

    def _observe(self, command: AgentCommand, *, accepted: bool) -> CommandResult:
        try:
            info = self.runner.get_contract_workload(command.owner.type, command.owner.id)
        except WorkloadNotFound:
            return self._accepted_absent(command.request_id)
        if info.generation != command.owner.generation:
            raise WorkloadConflict("status generation 与节点容器不一致")
        return self._response_from_info(command, info, accepted=accepted)

    def _response_from_info(
        self, command: AgentCommand, info: WorkloadInfo, *, accepted: bool
    ) -> CommandResult:
        state = self._state_from_info(info)
        inference_failure: str | None = None
        stop_uncertain = False
        if info.kind == "inference" and state == AgentWorkloadState.STARTING:
            inference_failure = self._inference_failure_reason(info)
            if inference_failure is not None:
                quiesced = self.runner.quiesce_failed_inference(
                    info.workload_id,
                    info.generation,
                    timeout_seconds=self.settings.inference_failure_stop_timeout_seconds,
                )
                if quiesced:
                    state = AgentWorkloadState.FAILED
                else:
                    stop_uncertain = True
        metadata: dict[str, Any] = {}
        if command.owner.type == "deployment":
            # 未通过 readiness 前不发布上游地址；控制面会保留 GPU 租约并继续 STATUS。
            metadata = {"health_status": info.health_status}
            if state == AgentWorkloadState.RUNNING:
                metadata.update(
                    {
                        "endpoint": info.endpoint,
                        "port": info.port,
                        "service_type": info.service_type,
                    }
                )
            metadata = {key: value for key, value in metadata.items() if value is not None}
        elif command.owner.type == "training":
            metadata = self.runner.training_metadata(
                command.owner.id,
                completed=state == AgentWorkloadState.SUCCEEDED,
            )
        elif command.owner.type == "evaluation" and state == AgentWorkloadState.SUCCEEDED:
            metadata = self.runner.evaluation_metadata(command.owner.id)
        message = None
        if state == AgentWorkloadState.FAILED:
            message = inference_failure or (
                f"工作负载异常退出，exit_code={info.exit_code}"
                if info.exit_code is not None
                else "工作负载处于失败状态"
            )
        elif info.kind == "inference" and state == AgentWorkloadState.STARTING:
            if stop_uncertain:
                message = f"{inference_failure}；停止结果尚未确认，继续保留 GPU 并探测"
            elif info.status == "restarting":
                message = "推理容器正在重启，继续保留 GPU 并探测"
            elif info.health_status == "unhealthy":
                message = "推理服务健康检查未通过，继续保留 GPU 并探测"
            elif info.health_status == "starting":
                message = "推理服务正在加载模型，等待 readiness"
            else:
                message = "推理容器尚未提供可信健康状态，等待重建或后续探测"
        response = AgentCommandResponse(
            request_id=command.request_id,
            accepted=accepted,
            observed_state=state,
            observed_at=datetime.now(UTC),
            message=message,
            metadata=metadata,
        )
        return CommandResult(status_code=200, response=response)

    def _inference_failure_reason(self, info: WorkloadInfo) -> str | None:
        if info.health_status == "healthy" and info.status == "running":
            return None
        if info.restart_count >= INFERENCE_MAX_RESTARTS:
            return f"推理容器重启次数已达到上限 {INFERENCE_MAX_RESTARTS}"
        if info.status == "restarting":
            return self._startup_timeout_reason(info.finished_at or info.created_at)
        if info.health_status == "unhealthy":
            if info.health_failing_streak <= 0:
                return "推理服务 unhealthy 状态缺少可信连续失败计数"
            unhealthy_seconds = info.health_failing_streak * INFERENCE_HEALTH_INTERVAL_SECONDS
            if unhealthy_seconds >= self.settings.inference_unhealthy_timeout_seconds:
                return f"推理服务连续健康检查失败达到 {self.settings.inference_unhealthy_timeout_seconds} 秒"
            return None
        return self._startup_timeout_reason(info.started_at or info.created_at)

    def _startup_timeout_reason(self, started_at: datetime | None) -> str | None:
        if started_at is None:
            # Docker 正常 inspect 必有 Created；缺失意味着无法证明启动窗口仍然有效。
            # 继续等待会让异常容器无限占用整卡，因此进入同一“先停止、后失败”流程。
            return "推理容器缺少可信启动时间，无法执行有界 readiness 探测"
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        elapsed = max(0.0, (now - started_at).total_seconds())
        if elapsed >= self.settings.inference_startup_timeout_seconds:
            return f"推理服务启动超过 {self.settings.inference_startup_timeout_seconds} 秒仍未就绪"
        return None

    @staticmethod
    def _state_from_info(info: WorkloadInfo) -> AgentWorkloadState:
        if info.kind == "inference":
            if info.status in {"created", "restarting", "paused"}:
                return AgentWorkloadState.STARTING
            if info.status == "running":
                # fail closed：只有 Docker healthcheck 的 /health 成功才可对外服务。
                return (
                    AgentWorkloadState.RUNNING
                    if info.health_status == "healthy"
                    else AgentWorkloadState.STARTING
                )
            if info.status == "removing":
                return AgentWorkloadState.STOPPING
            return AgentWorkloadState.FAILED
        if info.status in {"created", "restarting"}:
            return AgentWorkloadState.STARTING
        if info.status in {"running", "paused"}:
            return AgentWorkloadState.RUNNING
        if info.status == "removing":
            return AgentWorkloadState.STOPPING
        if info.status == "exited" and info.kind in {"training", "evaluation"} and info.exit_code == 0:
            return AgentWorkloadState.SUCCEEDED
        return AgentWorkloadState.FAILED

    @staticmethod
    def _merge_vllm_arguments(simplified: dict[str, Any], detailed: dict[str, Any]) -> dict[str, Any]:
        if len(simplified) + len(detailed) > 128:
            raise InvalidWorkload("vLLM 参数数量超过节点安全上限")
        merged = dict(simplified)
        merged.update(detailed)
        return merged

    def _materialize_training_files(
        self, command: AgentCommand, execution: LLaMAFactoryExecution
    ) -> tuple[Path, Path, DatasetFormat]:
        expected_output = self.settings.checkpoint_root.resolve(strict=True) / str(command.owner.id)
        if (
            not execution.output_dir.is_absolute()
            or Path(os.path.abspath(execution.output_dir)) != expected_output
        ):
            raise InvalidWorkload(f"训练输出目录必须由系统派生为：{expected_output}")
        try:
            dataset_path, _ = strict_existing_path(
                execution.dataset_path,
                (self.settings.dataset_root,),
                directory=False,
            )
        except EvaluationInputError as exc:
            raise InvalidWorkload(str(exc).replace("评测", "训练")) from exc
        dataset_name = "openllmops_dataset"
        dataset_dir = (
            self.settings.runtime_root
            / "contract"
            / "training"
            / str(command.owner.id)
            / str(command.owner.generation)
            / "dataset"
        )
        dataset_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        dataset_info, dataset_format = self._dataset_info(dataset_path, execution.stage)
        self._write_atomic_json(dataset_dir / "dataset_info.json", {dataset_name: dataset_info})

        try:
            config = build_training_config(
                SafeTrainingRequest(
                    stage=TrainingStage(execution.stage),
                    algorithm=TrainingAlgorithm(execution.algorithm),
                    model_path=WORKSPACE_MODEL,
                    dataset_dir=WORKSPACE_DATASET,
                    output_dir=WORKSPACE_OUTPUT,
                    dataset_name=dataset_name,
                    dataset_format=dataset_format,
                    **execution.training_config.model_dump(),
                )
            )
        except (TrainingConfigError, ValueError) as exc:
            raise InvalidWorkload(str(exc)) from exc
        config_dir = self.settings.training_config_root / str(command.owner.id)
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        config_path = config_dir / f"generation-{command.owner.generation}.json"
        self._write_atomic_json(config_path, config)
        return config_path, dataset_dir, dataset_format

    @staticmethod
    def _dataset_info(dataset_path: Path, stage: str) -> tuple[dict[str, Any], DatasetFormat]:
        def reject_constant(value: str) -> None:
            raise ValueError(f"非有限数值：{value}")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"重复字段：{key}")
                result[key] = value
            return result

        try:
            with dataset_path.open("rb") as source:
                first_line: bytes | None = None
                while True:
                    raw = source.readline(MAX_TRAINING_JSONL_LINE_BYTES + 1)
                    if not raw:
                        break
                    if len(raw) > MAX_TRAINING_JSONL_LINE_BYTES:
                        raise InvalidWorkload("训练数据集单行超过 16 MiB 安全上限")
                    if raw.strip():
                        first_line = raw
                        break
            first_record = (
                json.loads(
                    first_line.decode("utf-8"),
                    parse_constant=reject_constant,
                    object_pairs_hook=unique_object,
                )
                if first_line is not None
                else None
            )
        except InvalidWorkload:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise InvalidWorkload(f"无法解析训练数据集首条记录：{exc}") from exc
        if not isinstance(first_record, dict):
            raise InvalidWorkload("训练数据集没有可识别的 JSON 对象记录")
        info: dict[str, Any] = {"file_name": str(WORKSPACE_DATA_FILE)}
        if stage == "cpt":
            prompt = "text" if isinstance(first_record.get("text"), str) else "content"
            if not isinstance(first_record.get(prompt), str):
                raise InvalidWorkload("CPT 数据集缺少 text/content 字段")
            info["columns"] = {"prompt": prompt}
            dataset_format = DatasetFormat.CPT_TEXT
        elif isinstance(first_record.get("messages"), list):
            info.update(
                {
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
            )
            dataset_format = DatasetFormat.MESSAGES
        elif isinstance(first_record.get("conversations"), list):
            info.update(
                {
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations"},
                }
            )
            dataset_format = DatasetFormat.MESSAGES
        elif isinstance(first_record.get("instruction"), str) and isinstance(first_record.get("output"), str):
            columns = {
                "prompt": "instruction",
                "response": "output",
            }
            # Alpaca 数据允许省略 input；只声明真实存在的列，避免 LLaMAFactory
            # 在无 input 字段的数据集上把整批样本判为格式错误。
            if isinstance(first_record.get("input"), str):
                columns["query"] = "input"
            info["columns"] = columns
            dataset_format = DatasetFormat.ALPACA
        else:
            raise InvalidWorkload("SFT 数据集字段无法映射到 LLaMAFactory")
        return info, dataset_format

    @staticmethod
    def _write_atomic_json(path: Path, value: dict[str, Any]) -> None:
        body = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        CommandProcessor._write_atomic_text(path, body)

    @staticmethod
    def _write_atomic_text(path: Path, value: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _accepted_absent(request_id: UUID) -> CommandResult:
        return CommandResult(
            status_code=200,
            response=AgentCommandResponse(
                request_id=request_id,
                accepted=True,
                observed_state=AgentWorkloadState.ABSENT,
                observed_at=datetime.now(UTC),
            ),
        )

    @staticmethod
    def _rejected(
        request_id: UUID,
        status_code: int,
        message: str,
        error_code: str,
    ) -> CommandResult:
        return CommandResult(
            status_code=status_code,
            response=AgentCommandResponse(
                request_id=request_id,
                accepted=False,
                observed_state=AgentWorkloadState.FAILED,
                observed_at=datetime.now(UTC),
                message=message,
                error_code=error_code,
            ),
        )
