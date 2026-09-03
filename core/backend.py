from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.device_types import DeviceType


@dataclass
class BackendInfo:
    name: str
    device_type: DeviceType
    available: bool
    details: dict[str, Any]


class Backend(ABC):

    @abstractmethod
    def detect(self) -> BackendInfo:
        """
        Detect whether this backend is available on the current machine.
        """
        pass

    @abstractmethod
    def capabilities(self) -> list[str]:
        """
        Return the types of AI tasks this backend can perform.
        """
        pass

    def score(self, task_type: str, policy) -> int:
        """
        Return how suitable this backend is for the given task.

        Higher score = preferred backend.
        """
        return 0

    @abstractmethod
    def run(self, task_type: str, payload: Any) -> Any:
        """
        Execute a task using this backend.
        """
        pass
