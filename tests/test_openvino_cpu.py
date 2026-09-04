# Reusable regression test for the OpenVINO CPU ResNet18 backend.
import builtins
import multiprocessing
from statistics import median

import openvino
from openvino import Core

from core.benchmark import BenchmarkStats
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.runtime_types import RuntimeType
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


def main() -> None:
    print("1. OpenVINO runtime detection")
    print("-----------------------------")

    runtime_version = openvino.__version__
    runtime_devices = Core().available_devices

    print(f"Version: {runtime_version}")
    print(f"Core().available_devices: {runtime_devices}")

    assert runtime_version.startswith("2025.4.1")
    assert "CPU" in runtime_devices

    print()
    print("2. Missing-package safety")
    print("-------------------------")

    router = AIRouter(policy=RoutingPolicy.BALANCED)

    openvino_backend = next(
        backend
        for backend in router.backends
        if type(backend).__name__ == "OpenVINOBackend"
    )

    real_info = openvino_backend.detect()

    assert openvino_backend.target_device == "CPU"
    assert real_info.name == "OpenVINO"
    assert real_info.device_type == DeviceType.CPU
    assert real_info.runtime == RuntimeType.OPENVINO
    assert real_info.accelerator_api is None
    assert real_info.available is True
    assert real_info.details["available_devices"] == runtime_devices
    assert real_info.details["target_device"] == "CPU"

    real_import = builtins.__import__

    def import_without_openvino(name, *args, **kwargs):
        if name == "openvino" or name.startswith("openvino."):
            raise ImportError("Simulated missing OpenVINO")

        return real_import(name, *args, **kwargs)

    builtins.__import__ = import_without_openvino

    try:
        missing_info = openvino_backend.detect()
    finally:
        builtins.__import__ = real_import

    assert missing_info.available is False
    assert missing_info.details["version"] is None
    assert missing_info.details["available_devices"] == []
    assert missing_info.details["target_device"] == "CPU"

    print("Missing-package simulation passed.")

    print()
    print("3-8. Fair interleaved backend comparison")
    print("----------------------------------------")

    task = Task(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        payload="test.png",
    )

    assert openvino_backend.score(
        task.task_type,
        RoutingPolicy.PERFORMANCE,
    ) == 38
    assert openvino_backend.score(
        task.task_type,
        RoutingPolicy.BALANCED,
    ) == 58
    assert openvino_backend.score(
        task.task_type,
        RoutingPolicy.LOW_POWER,
    ) == 63

    backend_names = [
        "OpenVINO",
        "Torchvision ResNet18 CPU",
    ]
    results = {
        backend_name: []
        for backend_name in backend_names
    }

    for run_number in range(1, 6):
        for backend_name in backend_names:
            result = router.route(
                task,
                benchmark_backend=backend_name,
            )
            results[backend_name].append(result)

            predictions = result["result"]["predictions"]
            inference_time_ms = result["result"].get("inference_time_ms")

            assert result["routing"]["backend"] == backend_name
            assert len(predictions) == 5
            assert inference_time_ms is not None
            assert inference_time_ms >= 0

            print(
                f"{backend_name} run {run_number}: "
                f"total={result['routing']['execution_time_ms']:.2f} ms, "
                f"inference={inference_time_ms:.2f} ms"
            )

    backend_statistics = {}

    for backend_name in backend_names:
        records = router.benchmarks.filter_records(
            backend=backend_name,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )

        assert len(records) == 5

        cold_total = records[0].total_time_ms
        warm_totals = [
            record.total_time_ms
            for record in records[1:]
        ]
        inference_times = [
            record.inference_time_ms
            for record in records
        ]

        assert len(warm_totals) == 4
        assert all(value is not None for value in inference_times)

        performance_bonus = router.benchmarks.performance_score(
            backend=backend_name,
            task_type=TaskType.IMAGE_CLASSIFICATION.value,
        )

        assert performance_bonus is not None

        backend = next(
            candidate
            for candidate in router.backends
            if candidate.detect().name == backend_name
        )
        base_score = backend.score(
            task.task_type,
            router.policy,
        )
        combined_score = base_score + performance_bonus

        backend_statistics[backend_name] = {
            "cold_total": cold_total,
            "warm_totals": warm_totals,
            "median_warm_total": median(warm_totals),
            "inference_times": inference_times,
            "performance_bonus": performance_bonus,
            "combined_score": combined_score,
        }

        print()
        print(backend_name)
        print("-" * len(backend_name))
        print(f"Cold total: {cold_total:.2f} ms")
        print(
            "Warm totals:",
            [round(value, 2) for value in warm_totals],
        )
        print(
            f"Median warm total: "
            f"{backend_statistics[backend_name]['median_warm_total']:.2f} ms"
        )
        print(
            "Inference times:",
            [round(value, 2) for value in inference_times],
        )
        print(f"Historical performance bonus: {performance_bonus:.2f}")
        print(f"Combined score: {combined_score:.2f}")

    real_cpu_backends = {
        backend_name: next(
            backend
            for backend in router.backends
            if backend.detect().name == backend_name
        )
        for backend_name in backend_names
    }

    def selected_real_cpu_backend(policy: RoutingPolicy) -> str:
        scores = {}

        for backend_name, backend in real_cpu_backends.items():
            base_score = backend.score(task.task_type, policy)
            performance_bonus = router.benchmarks.performance_score(
                backend=backend_name,
                task_type=TaskType.IMAGE_CLASSIFICATION.value,
            )

            assert performance_bonus is not None

            scores[backend_name] = base_score + performance_bonus

        return max(scores, key=scores.get)

    openvino_predictions = results["OpenVINO"][0]["result"][
        "predictions"
    ]
    torchvision_predictions = results["Torchvision ResNet18 CPU"][0][
        "result"
    ]["predictions"]

    openvino_categories = [
        prediction["category"]
        for prediction in openvino_predictions
    ]
    torchvision_categories = [
        prediction["category"]
        for prediction in torchvision_predictions
    ]

    print()
    print("OpenVINO predictions:")
    for prediction in openvino_predictions:
        print(
            f"  {prediction['category']}: "
            f"{prediction['confidence_percent']:.2f}%"
        )

    print("Torchvision predictions:")
    for prediction in torchvision_predictions:
        print(
            f"  {prediction['category']}: "
            f"{prediction['confidence_percent']:.2f}%"
        )

    assert openvino_categories == torchvision_categories

    for openvino_prediction, torchvision_prediction in zip(
        openvino_predictions,
        torchvision_predictions,
    ):
        confidence_difference = abs(
            openvino_prediction["confidence_percent"]
            - torchvision_prediction["confidence_percent"]
        )

        assert confidence_difference <= 0.1, (
            "Confidence difference exceeded tolerance for "
            f"{openvino_prediction['category']}: "
            f"{confidence_difference:.2f} percentage points"
        )

    print(
        "Categories match and confidence differences are within 0.1 points."
    )

    print()
    print("9. Confirm CPU compilation")
    print("--------------------------")

    execution_devices = list(
        openvino_backend.compiled_model.get_property(
            "EXECUTION_DEVICES"
        )
    )

    print(f"Compiled execution devices: {execution_devices}")

    assert execution_devices == ["CPU"]

    print()
    print("10. Unsupported-task behavior")
    print("-----------------------------")

    try:
        openvino_backend.run(
            TaskType.GENERAL,
            "unsupported",
        )
    except NotImplementedError:
        print("Unsupported task correctly raised NotImplementedError.")
    else:
        raise AssertionError(
            "Unsupported OpenVINO task did not raise NotImplementedError"
        )

    print()
    print("11. Normal routing behavior")
    print("---------------------------")

    seeded_records = list(router.benchmarks.records)

    for policy in (
        RoutingPolicy.PERFORMANCE,
        RoutingPolicy.BALANCED,
        RoutingPolicy.LOW_POWER,
    ):
        policy_router = AIRouter(
            policy=policy,
            backends=list(real_cpu_backends.values()),
        )
        policy_router.benchmarks = BenchmarkStats(
            records=list(seeded_records),
        )
        expected_backend = selected_real_cpu_backend(policy)
        normal_result = policy_router.route(task)

        print(
            f"Normally selected {policy.value} backend:",
            normal_result["routing"]["backend"],
        )

        assert normal_result["routing"]["backend"] == expected_backend

        if policy == RoutingPolicy.LOW_POWER:
            assert expected_backend == "Torchvision ResNet18 CPU"

    print()
    print("All real OpenVINO CPU ResNet18 tests passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
