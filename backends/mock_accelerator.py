from typing import Any

from core.backend import Backend, BackendInfo
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.task_types import TaskType


class MockAcceleratorBackend(Backend):

    def detect(self) -> BackendInfo:
        return BackendInfo(
            name="Mock Accelerator",
            device_type=DeviceType.ACCELERATOR,
            available=True,
            details={
                "description": "Test backend for AI Router",
            },
        )

    def capabilities(self) -> list[str]:
        return [
            TaskType.CLASSIFICATION.value,
            TaskType.IMAGE_CLASSIFICATION.value,
        ]

    def score(self, task_type: str, policy: RoutingPolicy) -> int:
        if task_type == TaskType.CLASSIFICATION:
            if policy == RoutingPolicy.PERFORMANCE:
                return 100

            if policy == RoutingPolicy.BALANCED:
                return 80

            if policy == RoutingPolicy.LOW_POWER:
                return 30

        if task_type == TaskType.IMAGE_CLASSIFICATION:
            if policy == RoutingPolicy.PERFORMANCE:
                return 70

            if policy == RoutingPolicy.BALANCED:
                return 50

            if policy == RoutingPolicy.LOW_POWER:
                return 80

        return 0

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type == TaskType.CLASSIFICATION:
            return {
                "backend": "mock_accelerator",
                "input": payload,
                "output": "test-category",
            }

        if task_type == TaskType.IMAGE_CLASSIFICATION:
            return {
                "backend": "mock_accelerator",
                "input": payload,
                "predictions": [
                    {
                        "category": "mock-image-category",
                        "confidence_percent": 99.0,
                    }
                ],
            }

        raise NotImplementedError(
            f"Mock accelerator does not support task: {task_type}"
        )
