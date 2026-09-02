# Regression test for explicit opt-in of the synthetic Mock backend.
from backends.cpu import CPUBackend
from backends.mock_accelerator import MockAcceleratorBackend
from core.policy import RoutingPolicy
from core.registry import discover_backends
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


MOCK_BACKEND_NAME = "Mock Accelerator"


def backend_names(backends):
    return [
        backend.detect().name
        for backend in backends
    ]


def main() -> None:
    production_backends = discover_backends()
    production_names = backend_names(production_backends)
    production_types = {
        type(backend).__name__
        for backend in production_backends
    }

    assert MOCK_BACKEND_NAME not in production_names
    assert "CPUBackend" in production_types
    assert "OpenVINOBackend" in production_types
    assert "OpenVINOIntelGPUBackend" in production_types
    assert "TorchvisionClassifierBackend" in production_types

    production_router = AIRouter()

    assert MOCK_BACKEND_NAME not in backend_names(
        production_router.backends
    )

    supplied_backends = [
        *production_backends,
        MockAcceleratorBackend(),
    ]
    injected_router = AIRouter(backends=supplied_backends)
    injected_names = backend_names(injected_router.backends)

    assert MOCK_BACKEND_NAME in injected_names

    supplied_backends.clear()

    assert backend_names(injected_router.backends) == injected_names

    task = Task(
        task_type=TaskType.CLASSIFICATION,
        payload="picture of a cat",
    )
    cpu_backend = CPUBackend()
    expected_backends = {
        RoutingPolicy.PERFORMANCE: MOCK_BACKEND_NAME,
        RoutingPolicy.BALANCED: MOCK_BACKEND_NAME,
        RoutingPolicy.LOW_POWER: cpu_backend.detect().name,
    }

    for policy, expected_backend in expected_backends.items():
        router = AIRouter(
            policy=policy,
            backends=[
                cpu_backend,
                MockAcceleratorBackend(),
            ],
        )
        result = router.route(task)

        assert result["routing"]["backend"] == expected_backend

    print("Mock backend explicit opt-in regression passed.")


if __name__ == "__main__":
    main()
