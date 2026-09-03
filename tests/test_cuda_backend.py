# Production regression for the registered PyTorch CUDA backend.
import multiprocessing
from unittest.mock import patch

import torch

from backends.cuda import (
    PyTorchCUDABackend,
    RegisteredPyTorchCUDABackend,
)
from backends.torchvision_classifier import TorchvisionClassifierBackend
from core.benchmark import BenchmarkStats
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


CUDA_BACKEND_NAME = "PyTorch CUDA"
CUDA_RESULT_IDENTITY = "pytorch_cuda_resnet18"
TORCHVISION_BACKEND_NAME = "Torchvision ResNet18 CPU"
OPENVINO_CPU_BACKEND_NAME = "OpenVINO"
INTEL_GPU_BACKEND_NAME = "OpenVINO Intel GPU"


def assert_prediction_parity(cpu_result, cuda_result) -> None:
    cpu_predictions = cpu_result["predictions"]
    cuda_predictions = cuda_result["predictions"]

    assert len(cpu_predictions) == 5
    assert len(cuda_predictions) == 5

    cpu_categories = [
        item["category"]
        for item in cpu_predictions
    ]
    cuda_categories = [
        item["category"]
        for item in cuda_predictions
    ]

    assert cpu_categories[0] == cuda_categories[0], (
        "Torchvision CPU and CUDA top-1 categories differ: "
        f"{cpu_categories[0]} != {cuda_categories[0]}"
    )
    assert set(cpu_categories) == set(cuda_categories), (
        "Torchvision CPU and CUDA top-five category sets differ: "
        f"{set(cpu_categories)} != {set(cuda_categories)}"
    )

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    for cuda_prediction in cuda_predictions:
        category = cuda_prediction["category"]
        confidence_difference = abs(
            cuda_prediction["confidence_percent"]
            - cpu_confidences[category]
        )

        print(
            f"{category} confidence difference: "
            f"{confidence_difference:.2f} percentage points"
        )

        assert confidence_difference <= 0.1, (
            "Confidence difference exceeded tolerance for "
            f"{category}: {confidence_difference:.2f}"
        )


def expected_router_backend(
    router,
    task,
    policy,
) -> str:
    expected_backend = None
    expected_score = None

    for backend in router.backends:
        info = backend.detect()

        if not info.available:
            continue

        if task.task_type.value not in backend.capabilities():
            continue

        base_score = backend.score(
            task.task_type,
            policy,
        )
        historical_bonus = router.benchmarks.performance_score(
            backend=info.name,
            task_type=task.task_type.value,
        )
        combined_score = base_score + (
            historical_bonus
            if historical_bonus is not None
            else 0
        )

        print(
            f"{policy.value} candidate {info.name}: "
            f"base={base_score}, "
            f"history={historical_bonus}, "
            f"combined={combined_score:.4f}"
        )

        if (
            expected_score is None
            or combined_score > expected_score
        ):
            expected_backend = info.name
            expected_score = combined_score

    assert expected_backend is not None

    return expected_backend


def assert_unavailable_fallback() -> None:
    backend = PyTorchCUDABackend()

    with (
        patch(
            "backends.cuda.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "backends.cuda.torch.cuda.device_count",
            return_value=0,
        ),
    ):
        info = backend.detect()

        assert info.name == CUDA_BACKEND_NAME
        assert info.device_type == DeviceType.GPU
        assert not info.available
        assert info.details["device_index"] == 0
        assert info.details["device_count"] == 0
        assert info.details["device_name"] is None
        assert info.details["compute_capability"] is None
        assert info.details["vram_bytes"] is None
        assert backend.model is None

        try:
            backend.run(
                TaskType.IMAGE_CLASSIFICATION,
                "test.png",
            )
        except RuntimeError as error:
            assert "not available" in str(error)
        else:
            raise AssertionError(
                "Unavailable CUDA execution unexpectedly succeeded."
            )

        assert backend.model is None


def main() -> None:
    assert_unavailable_fallback()

    generic_backend = PyTorchCUDABackend()

    assert generic_backend.device_index == 0
    assert generic_backend.warmup_runs == 0
    assert generic_backend.model is None

    router = AIRouter(policy=RoutingPolicy.BALANCED)
    registered_cuda_backends = [
        backend
        for backend in router.backends
        if isinstance(
            backend,
            RegisteredPyTorchCUDABackend,
        )
    ]

    assert len(registered_cuda_backends) == 1
    assert not any(
        type(backend) is PyTorchCUDABackend
        for backend in router.backends
    )

    cuda_backend = registered_cuda_backends[0]
    cuda_info = cuda_backend.detect()

    assert cuda_info.name == CUDA_BACKEND_NAME
    assert cuda_info.device_type == DeviceType.GPU
    assert cuda_backend.device_index == 0
    assert cuda_backend.warmup_runs == 2
    assert cuda_backend.model is None
    assert cuda_backend.capabilities() == [
        TaskType.IMAGE_CLASSIFICATION.value,
    ]

    assert cuda_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.PERFORMANCE,
    ) == 37
    assert cuda_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.BALANCED,
    ) == 57
    assert cuda_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.LOW_POWER,
    ) == 0
    assert cuda_backend.score(
        TaskType.GENERAL,
        RoutingPolicy.PERFORMANCE,
    ) == 0

    task = Task(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        payload="test.png",
    )

    if not cuda_info.available:
        assert cuda_backend.model is None
        routed = router.route(task)

        assert routed["routing"]["backend"] != CUDA_BACKEND_NAME
        print(
            "SKIPPED: CUDA is unavailable; safe CPU/OpenVINO routing "
            "fallback passed."
        )
        return

    assert cuda_info.details["device_count"] >= 1
    assert cuda_info.details["device_index"] == 0
    assert cuda_info.details["device_name"] == (
        torch.cuda.get_device_name(0)
    )
    assert cuda_info.details["compute_capability"] == (
        torch.cuda.get_device_capability(0)
    )
    assert cuda_info.details["vram_bytes"] == (
        torch.cuda.get_device_properties(0).total_memory
    )
    assert cuda_info.details["pytorch_version"] == torch.__version__
    assert cuda_info.details["cuda_build_version"] == torch.version.cuda

    print(f"CUDA device name: {cuda_info.details['device_name']}")
    print(
        "CUDA compute capability: "
        f"{cuda_info.details['compute_capability']}"
    )
    print(f"CUDA VRAM bytes: {cuda_info.details['vram_bytes']}")

    cpu_backend = TorchvisionClassifierBackend()
    cpu_result = cpu_backend.run(
        TaskType.IMAGE_CLASSIFICATION,
        "test.png",
    )

    with patch.object(
        torch.cuda,
        "synchronize",
        wraps=torch.cuda.synchronize,
    ) as synchronize:
        first_routed = router.route(
            task,
            benchmark_backend=CUDA_BACKEND_NAME,
        )

    first_result = first_routed["result"]

    assert synchronize.call_count == 4
    assert all(
        call.args == (cuda_backend.device,)
        for call in synchronize.call_args_list
    )
    assert first_routed["routing"]["backend"] == CUDA_BACKEND_NAME
    assert first_result["backend"] == CUDA_RESULT_IDENTITY
    assert first_result["inference_time_ms"] >= 0
    assert first_result["warmup_runs"] == 2
    assert first_result["warmup_time_ms"] >= 0
    assert (
        first_routed["routing"]["execution_time_ms"]
        >= first_result["warmup_time_ms"]
    )
    assert cuda_backend._last_input_device == cuda_backend.device
    assert cuda_backend._last_output_device == cuda_backend.device
    assert all(
        parameter.device == cuda_backend.device
        for parameter in cuda_backend.model.parameters()
    )

    assert_prediction_parity(cpu_result, first_result)

    cuda_results = [first_result]

    for _ in range(4):
        routed = router.route(
            task,
            benchmark_backend=CUDA_BACKEND_NAME,
        )
        result = routed["result"]

        assert routed["routing"]["backend"] == CUDA_BACKEND_NAME
        assert result["backend"] == CUDA_RESULT_IDENTITY
        assert result["inference_time_ms"] >= 0
        assert result["warmup_runs"] == 0
        assert result["warmup_time_ms"] == 0.0
        assert_prediction_parity(cpu_result, result)
        cuda_results.append(result)

    assert len(cuda_results) == 5

    cuda_records = router.benchmarks.filter_records(
        backend=CUDA_BACKEND_NAME,
        task_type=TaskType.IMAGE_CLASSIFICATION.value,
    )

    assert len(cuda_records) == 5
    assert all(
        record.backend == CUDA_BACKEND_NAME
        for record in cuda_records
    )
    assert all(
        record.task_type == TaskType.IMAGE_CLASSIFICATION.value
        for record in cuda_records
    )
    assert all(
        record.inference_time_ms is not None
        for record in cuda_records
    )

    for other_backend_name in (
        TORCHVISION_BACKEND_NAME,
        OPENVINO_CPU_BACKEND_NAME,
        INTEL_GPU_BACKEND_NAME,
    ):
        isolated_records = router.benchmarks.filter_records(
            backend=other_backend_name,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )

        assert isolated_records == []

    router.policy = RoutingPolicy.PERFORMANCE

    while len(router.benchmarks.filter_records(
        backend=TORCHVISION_BACKEND_NAME,
        task_type=TaskType.IMAGE_CLASSIFICATION.value,
    )) < 5:
        eligible_record_counts = {}

        for backend in router.backends:
            info = backend.detect()

            if (
                info.available
                and task.task_type.value in backend.capabilities()
                and backend.score(task.task_type, router.policy) > 0
            ):
                eligible_record_counts[info.name] = len(
                    router.benchmarks.filter_records(
                        backend=info.name,
                        task_type=task.task_type.value,
                    )
                )

        least_sampled_count = min(eligible_record_counts.values())
        routed = router.route(task)

        assert eligible_record_counts[
            routed["routing"]["backend"]
        ] == least_sampled_count

    eligible_record_counts = {
        backend.detect().name: len(
            router.benchmarks.filter_records(
                backend=backend.detect().name,
                task_type=task.task_type.value,
            )
        )
        for backend in router.backends
        if backend.detect().available
        and task.task_type.value in backend.capabilities()
        and backend.score(task.task_type, router.policy) > 0
    }

    assert eligible_record_counts[TORCHVISION_BACKEND_NAME] == 5
    assert all(
        record_count >= 5
        for record_count in eligible_record_counts.values()
    )

    seeded_records = list(router.benchmarks.records)

    for policy in (
        RoutingPolicy.PERFORMANCE,
        RoutingPolicy.BALANCED,
        RoutingPolicy.LOW_POWER,
    ):
        router.benchmarks = BenchmarkStats(
            records=list(seeded_records),
        )
        router.policy = policy

        expected_backend = expected_router_backend(
            router,
            task,
            policy,
        )
        routed = router.route(task)

        assert routed["routing"]["backend"] == expected_backend

        if policy == RoutingPolicy.LOW_POWER:
            assert routed["routing"]["backend"] == TORCHVISION_BACKEND_NAME

    print("PyTorch CUDA production backend test passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
