from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .agent_contract import (
    AgentAction,
    AgentCommand,
    AgentCommandResponse,
    AgentWorkloadState,
    canonical_json,
)
from .config import Settings
from .docker_runner import (
    DockerRunner,
    InvalidWorkload,
    WorkloadConflict,
    WorkloadNotFound,
)
from .evaluation_runtime import (
    DatasetSource,
    EvaluationInputError,
    prepare_evaluation_workspace,
)
from .schemas import (
    EvaluationLaunchRequest,
    InferenceLaunchRequest,
    TrainingLaunchRequest,
    WorkloadInfo,
)

MAX_CACHED_REQUESTS = 2048
STATE_VERSION = 1
PROTECTED_TRAINING_KEYS = frozenset(
    {
        "dataset",
        "dataset_dir",
        "finetuning_type",
        "model_name_or_path",
        "output_dir",
        "quantization_bit",
        "stage",
        "trust_remote_code",
    }
)


class ExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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
    training_config: dict[str, Any] = Field(default_factory=dict)
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
    def __init__(self, settings: Settings, runner: DockerRunner) -> None:
        self.settings = settings
        self.runner = runner
        self.state = CommandStateStore(settings.runtime_root)
        self._lock = threading.RLock()

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
        if command.execution:
            raise InvalidWorkload("stop/status 命令的 execution 必须为空对象")
        if command.action == AgentAction.STOP:
            return self._stop(command)
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
            config_path, dataset_dir = self._materialize_training_files(command, execution)
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
        with suppress(WorkloadNotFound):
            self.runner.stop_contract_workload(
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
        metadata: dict[str, Any] = {}
        if command.owner.type == "deployment":
            metadata = {
                "endpoint": info.endpoint,
                "port": info.port,
                "service_type": info.service_type,
            }
            metadata = {key: value for key, value in metadata.items() if value is not None}
        elif command.owner.type == "training":
            metadata = self.runner.training_metadata(command.owner.id)
        elif command.owner.type == "evaluation" and state == AgentWorkloadState.SUCCEEDED:
            metadata = self.runner.evaluation_metadata(command.owner.id)
        message = None
        if state == AgentWorkloadState.FAILED:
            message = (
                f"工作负载异常退出，exit_code={info.exit_code}"
                if info.exit_code is not None
                else "工作负载处于失败状态"
            )
        response = AgentCommandResponse(
            request_id=command.request_id,
            accepted=accepted,
            observed_state=state,
            observed_at=datetime.now(UTC),
            message=message,
            metadata=metadata,
        )
        return CommandResult(status_code=200, response=response)

    @staticmethod
    def _state_from_info(info: WorkloadInfo) -> AgentWorkloadState:
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
    ) -> tuple[Path, Path]:
        overlap = PROTECTED_TRAINING_KEYS & execution.training_config.keys()
        if overlap:
            raise InvalidWorkload(f"训练参数不能覆盖系统字段：{', '.join(sorted(overlap))}")
        if len(execution.training_config) > 256:
            raise InvalidWorkload("训练参数数量超过节点安全上限")
        dataset_path = self._resolve_existing_file(execution.dataset_path, self.settings.dataset_root)
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
        dataset_info = self._dataset_info(dataset_path, execution.stage)
        self._write_atomic_json(dataset_dir / "dataset_info.json", {dataset_name: dataset_info})

        finetuning_type = "lora" if execution.algorithm == "qlora" else execution.algorithm
        config: dict[str, Any] = {
            **execution.training_config,
            "model_name_or_path": str(execution.model_path),
            "dataset": dataset_name,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(execution.output_dir),
            "stage": "pt" if execution.stage == "cpt" else "sft",
            "finetuning_type": finetuning_type,
            "trust_remote_code": False,
        }
        if execution.algorithm == "qlora":
            config["quantization_bit"] = 4
        config_dir = self.settings.training_config_root / str(command.owner.id)
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        config_path = config_dir / f"generation-{command.owner.generation}.yaml"
        self._write_atomic_text(config_path, yaml.safe_dump(config, sort_keys=True))
        return config_path, dataset_dir

    @staticmethod
    def _dataset_info(dataset_path: Path, stage: str) -> dict[str, Any]:
        try:
            with dataset_path.open("r", encoding="utf-8") as source:
                first_record = next((json.loads(line) for line in source if line.strip()), None)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidWorkload(f"无法解析训练数据集首条记录：{exc}") from exc
        if not isinstance(first_record, dict):
            raise InvalidWorkload("训练数据集没有可识别的 JSON 对象记录")
        info: dict[str, Any] = {"file_name": str(dataset_path)}
        if stage == "cpt":
            prompt = "text" if isinstance(first_record.get("text"), str) else "content"
            if not isinstance(first_record.get(prompt), str):
                raise InvalidWorkload("CPT 数据集缺少 text/content 字段")
            info["columns"] = {"prompt": prompt}
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
        elif isinstance(first_record.get("conversations"), list):
            info.update(
                {
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations"},
                }
            )
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
        else:
            raise InvalidWorkload("SFT 数据集字段无法映射到 LLaMAFactory")
        return info

    @staticmethod
    def _resolve_existing_file(candidate: Path, root: Path) -> Path:
        try:
            root_real = root.resolve(strict=True)
            candidate_real = candidate.resolve(strict=True)
            candidate_real.relative_to(root_real)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise InvalidWorkload(f"数据集路径不存在或越出受控目录：{candidate}") from exc
        if not candidate_real.is_file():
            raise InvalidWorkload("合同训练任务的数据集必须是 JSONL 文件")
        return candidate_real

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
