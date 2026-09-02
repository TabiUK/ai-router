from dataclasses import dataclass
from typing import Any

from core.task_types import TaskType


@dataclass
class Task:
    task_type: TaskType
    payload: Any