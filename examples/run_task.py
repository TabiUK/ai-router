from backends.mock_accelerator import MockAcceleratorBackend
from core.router import AIRouter
from core.registry import discover_backends
from core.task import Task
from core.task_types import TaskType
from core.policy import RoutingPolicy


task = Task(
    task_type=TaskType.CLASSIFICATION,
    payload="picture of a cat",
)

for policy in RoutingPolicy:
    print()
    print(f"Policy: {policy.value}")

    router = AIRouter(
        policy=policy,
        backends=[
            *discover_backends(),
            MockAcceleratorBackend(),
        ],
    )

    result = router.route(task)

    print(result)
