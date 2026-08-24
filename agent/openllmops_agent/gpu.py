from __future__ import annotations

from collections.abc import Mapping

from .schemas import GPUInfo, GPUInventory


class NVMLUnavailable(RuntimeError):
    pass


def read_gpu_inventory(configured_count: int, allocations: Mapping[int, str]) -> GPUInventory:
    """读取 NVML；驱动暂不可用时仍返回配置槽位，便于控制面展示降级原因。"""

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            detected_count = pynvml.nvmlDeviceGetCount()
            items: list[GPUInfo] = []
            for index in range(detected_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power_watts: float | None
                try:
                    power_watts = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
                except pynvml.NVMLError:
                    power_watts = None
                items.append(
                    GPUInfo(
                        index=index,
                        uuid=_decode(pynvml.nvmlDeviceGetUUID(handle)),
                        name=_decode(pynvml.nvmlDeviceGetName(handle)),
                        memory_total_mib=memory.total // 1024 // 1024,
                        memory_used_mib=memory.used // 1024 // 1024,
                        memory_free_mib=memory.free // 1024 // 1024,
                        utilization_percent=utilization.gpu,
                        temperature_celsius=pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        power_watts=power_watts,
                        allocated_to=allocations.get(index),
                    )
                )
            return GPUInventory(driver_available=True, gpus=items)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        placeholders = [
            GPUInfo(index=index, allocated_to=allocations.get(index)) for index in range(configured_count)
        ]
        return GPUInventory(driver_available=False, error=str(exc), gpus=placeholders)


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def read_busy_gpu_ids(indices: list[int]) -> set[int]:
    """返回存在计算/图形进程的卡；NVML 不可用时由调用方执行安全失败。"""

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            busy: set[int] = set()
            for index in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = []
                for getter_name in (
                    "nvmlDeviceGetComputeRunningProcesses",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ):
                    getter = getattr(pynvml, getter_name, None)
                    if getter is None:
                        continue
                    try:
                        processes.extend(getter(handle))
                    except pynvml.NVMLError_NotSupported:
                        continue
                if processes:
                    busy.add(index)
            return busy
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        raise NVMLUnavailable(str(exc)) from exc
