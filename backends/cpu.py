import platform
from typing import Any

from core.backend import Backend, BackendInfo
from core.backend_registry import register_backend
from core.device_types import DeviceType
from core.task_types import TaskType
from core.policy import RoutingPolicy
from core.runtime_types import RuntimeType

class CPUBackend(Backend):

    def detect(self) -> BackendInfo:
        return BackendInfo(
            name=platform.processor() or "CPU",
            device_type=DeviceType.CPU,
            runtime=RuntimeType.NATIVE,
            accelerator_api=None,
            available=True,
            details={
                "architecture": platform.machine(),
                "system": platform.system(),
            },
        )

    def capabilities(self) -> list[str]:
        return [
            TaskType.GENERAL.value,
            TaskType.CLASSIFICATION.value,
        ]

    def score(self, task_type: str, policy: RoutingPolicy) -> int:
        if task_type == TaskType.GENERAL:
            if policy == RoutingPolicy.PERFORMANCE:
                return 40

            if policy == RoutingPolicy.BALANCED:
                return 50

            if policy == RoutingPolicy.LOW_POWER:
                return 70

        if task_type == TaskType.CLASSIFICATION:
            if policy == RoutingPolicy.PERFORMANCE:
                return 10

            if policy == RoutingPolicy.BALANCED:
                return 20

            if policy == RoutingPolicy.LOW_POWER:
                return 50

        return 0

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type == TaskType.GENERAL:
            return {
                "backend": "cpu",
                "input": payload,
                "output": str(payload).upper(),
            }

        if task_type == TaskType.CLASSIFICATION:
            return {
                "backend": "cpu",
                "input": payload,
                "output": "cpu-test-category",
            }

        raise NotImplementedError(
            f"CPU backend does not support task: {task_type}"
        )

register_backend(CPUBackend)
