# Production registration and detection smoke test for PyTorch MPS.
import torch

from backends.mps import (
    PyTorchMPSBackend,
    RegisteredPyTorchMPSBackend,
)
from core.backend_registry import get_registered_backends
from core.policy import RoutingPolicy
from core.task_types import TaskType


MPS_BACKEND_NAME = "PyTorch MPS"
MPS_RESULT_IDENTITY = "pytorch_mps_resnet18"


def expected_mps_state() -> tuple[bool, bool]:
    if getattr(torch.backends, "mps", None) is None:
        return False, False

    return (
        torch.backends.mps.is_built(),
        torch.backends.mps.is_available(),
    )


def main() -> None:
    backend = PyTorchMPSBackend()
    info = backend.detect()
    mps_built, mps_available = expected_mps_state()

    assert info.name == MPS_BACKEND_NAME
    assert info.device_type == "mps"
    assert info.available == (mps_built and mps_available)
    assert info.details["mps_built"] == mps_built
    assert info.details["mps_available"] == mps_available
    assert info.details["pytorch_version"] == torch.__version__
    assert backend.capabilities() == [
        TaskType.IMAGE_CLASSIFICATION.value,
    ]
    assert backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.PERFORMANCE,
    ) == 37
    assert backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.BALANCED,
    ) == 57
    assert backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.LOW_POWER,
    ) == 0
    assert backend.score(
        TaskType.GENERAL,
        RoutingPolicy.PERFORMANCE,
    ) == 0

    registered_mps_backends = [
        registered_backend
        for registered_backend in get_registered_backends()
        if isinstance(
            registered_backend,
            RegisteredPyTorchMPSBackend,
        )
    ]

    assert len(registered_mps_backends) == 1
    registered_backend = registered_mps_backends[0]
    registered_info = registered_backend.detect()

    assert registered_info.name == MPS_BACKEND_NAME
    assert registered_backend.warmup_runs == 2
    assert registered_backend.model is None

    if not registered_info.available:
        print("SKIPPED: PyTorch MPS is not available.")
        return

    result = registered_backend.run(
        TaskType.IMAGE_CLASSIFICATION,
        "test.png",
    )

    assert result["backend"] == MPS_RESULT_IDENTITY
    assert result["inference_time_ms"] > 0
    assert result["warmup_runs"] == 2
    assert result["warmup_time_ms"] >= 0
    assert len(result["predictions"]) == 5
    assert (
        registered_backend._last_input_device.type
        == registered_backend.device.type
    )
    assert (
        registered_backend._last_output_device.type
        == registered_backend.device.type
    )
    assert all(
        parameter.device.type == registered_backend.device.type
        for parameter in registered_backend.model.parameters()
    )

    second_result = registered_backend.run(
        TaskType.IMAGE_CLASSIFICATION,
        "test.png",
    )

    assert second_result["backend"] == MPS_RESULT_IDENTITY
    assert second_result["inference_time_ms"] > 0
    assert second_result["warmup_runs"] == 0
    assert second_result["warmup_time_ms"] == 0.0
    assert len(second_result["predictions"]) == 5

    print("PyTorch MPS backend registration and inference test passed.")


if __name__ == "__main__":
    main()
