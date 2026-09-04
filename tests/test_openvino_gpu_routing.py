# Reusable routing regression for the Intel OpenVINO GPU backend.
import multiprocessing

from backends.openvino import (
    OpenVINOIntelGPUBackend,
    find_intel_gpu_device,
)
from core.benchmark import BenchmarkStats
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.runtime_types import RuntimeType
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


CPU_BACKEND_NAME = "OpenVINO"
GPU_BACKEND_NAME = "OpenVINO Intel GPU"


class FakeCore:
    available_devices = ["CPU", "GPU.0", "GPU.1"]

    def get_property(self, device, property_name):
        assert property_name == "FULL_DEVICE_NAME"

        names = {
            "GPU.0": "Example Discrete Graphics",
            "GPU.1": "Intel(R) Test Graphics",
        }
        return names[device]


class FailingPropertyCore:
    available_devices = ["GPU.0"]

    def get_property(self, device, property_name):
        raise RuntimeError("advertised GPU property query failed")


def assert_prediction_parity(cpu_result, gpu_result):
    cpu_predictions = cpu_result["predictions"]
    gpu_predictions = gpu_result["predictions"]

    assert len(cpu_predictions) == 5
    assert len(gpu_predictions) == 5

    cpu_categories = [
        item["category"]
        for item in cpu_predictions
    ]
    gpu_categories = [
        item["category"]
        for item in gpu_predictions
    ]

    assert cpu_categories[0] == gpu_categories[0]
    assert set(cpu_categories) == set(gpu_categories)

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    for gpu_prediction in gpu_predictions:
        category = gpu_prediction["category"]
        confidence_difference = abs(
            gpu_prediction["confidence_percent"]
            - cpu_confidences[category]
        )

        assert confidence_difference <= 0.1, (
            "Confidence difference exceeded tolerance for "
            f"{category}: {confidence_difference:.2f}"
        )


def expected_router_backend(
    router,
    task,
    policy,
):
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


def main():
    selected = find_intel_gpu_device(FakeCore())

    assert selected == (
        "GPU.1",
        "Intel(R) Test Graphics",
    )

    try:
        find_intel_gpu_device(FailingPropertyCore())
    except RuntimeError as error:
        assert str(error) == "advertised GPU property query failed"
    else:
        raise AssertionError(
            "Unexpected GPU property failure was silently hidden"
        )

    router = AIRouter(policy=RoutingPolicy.BALANCED)

    cpu_backend = next(
        backend
        for backend in router.backends
        if backend.detect().name == CPU_BACKEND_NAME
    )
    gpu_backend = next(
        backend
        for backend in router.backends
        if isinstance(backend, OpenVINOIntelGPUBackend)
    )

    gpu_info = gpu_backend.detect()

    assert gpu_info.name == GPU_BACKEND_NAME
    assert gpu_info.device_type == DeviceType.GPU
    assert gpu_info.runtime == RuntimeType.OPENVINO
    assert gpu_info.accelerator_api is None
    assert gpu_backend.warmup_runs == 2

    assert gpu_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.PERFORMANCE,
    ) == 37
    assert gpu_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.BALANCED,
    ) == 57
    assert gpu_backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        RoutingPolicy.LOW_POWER,
    ) == 0

    if not gpu_info.available:
        print("SKIPPED: No Intel OpenVINO GPU is available.")
        return

    assert gpu_info.details["full_device_name"] is not None
    assert "intel" in gpu_info.details["full_device_name"].casefold()

    task = Task(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        payload="test.png",
    )
    cpu_results = []
    gpu_results = []

    for run_index in range(5):
        execution_order = (
            [CPU_BACKEND_NAME, GPU_BACKEND_NAME]
            if run_index % 2 == 0
            else [GPU_BACKEND_NAME, CPU_BACKEND_NAME]
        )
        paired_results = {}

        for backend_name in execution_order:
            routed = router.route(
                task,
                benchmark_backend=backend_name,
            )
            paired_results[backend_name] = routed["result"]

        cpu_result = paired_results[CPU_BACKEND_NAME]
        gpu_result = paired_results[GPU_BACKEND_NAME]

        cpu_results.append(cpu_result)
        gpu_results.append(gpu_result)
        assert_prediction_parity(cpu_result, gpu_result)

    assert gpu_results[0]["warmup_runs"] == 2
    assert gpu_results[0]["warmup_time_ms"] >= 0

    for result in gpu_results[1:]:
        assert result["warmup_runs"] == 0
        assert result["warmup_time_ms"] == 0.0

    assert all(
        result["backend"] == "openvino_resnet18_cpu"
        for result in cpu_results
    )

    expected_gpu_identity = (
        f"openvino_resnet18_{gpu_backend.target_device.lower()}"
    )
    assert all(
        result["backend"] == expected_gpu_identity
        for result in gpu_results
    )

    assert list(
        cpu_backend.compiled_model.get_property("EXECUTION_DEVICES")
    ) == ["CPU"]
    assert list(
        gpu_backend.compiled_model.get_property("EXECUTION_DEVICES")
    ) == [gpu_backend.target_device]

    seeded_records = list(router.benchmarks.records)

    for policy in (
        RoutingPolicy.PERFORMANCE,
        RoutingPolicy.BALANCED,
        RoutingPolicy.LOW_POWER,
    ):
        policy_router = AIRouter(
            policy=policy,
            backends=[cpu_backend, gpu_backend],
        )
        policy_router.benchmarks = BenchmarkStats(
            records=list(seeded_records),
        )

        expected_backend = expected_router_backend(
            policy_router,
            task,
            policy,
        )
        routed = policy_router.route(task)

        assert routed["routing"]["backend"] == expected_backend

        if policy == RoutingPolicy.LOW_POWER:
            assert (
                routed["routing"]["backend"]
                != GPU_BACKEND_NAME
            )

    print("Intel OpenVINO GPU routing test passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
