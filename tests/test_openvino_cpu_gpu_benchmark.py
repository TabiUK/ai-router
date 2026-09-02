# Standalone OpenVINO CPU vs Intel GPU ResNet18 benchmark.
import multiprocessing
import math
import time
from statistics import median

from openvino import Core

from backends.openvino import OpenVINOBackend
from core.task_types import TaskType


INITIALIZATION_RUNS = 1
WARMUP_RUNS = 2
MEASURED_RUNS = 7
MAX_CONFIDENCE_DIFFERENCE = 0.1
STABILITY_TOLERANCE = 0.2
MINIMUM_STABLE_RUNS = 5


def assert_prediction_parity(
    cpu_predictions: list[dict],
    gpu_predictions: list[dict],
    run_label: str,
) -> None:
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

    assert cpu_categories[0] == gpu_categories[0], (
        f"{run_label} CPU and GPU top-1 categories differ: "
        f"{cpu_categories[0]} != {gpu_categories[0]}"
    )
    assert set(cpu_categories) == set(gpu_categories), (
        f"{run_label} CPU and GPU top-5 category sets differ: "
        f"{set(cpu_categories)} != {set(gpu_categories)}"
    )

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    print(f"{run_label} confidence differences by category:")

    for gpu_prediction in gpu_predictions:
        category = gpu_prediction["category"]
        confidence_difference = abs(
            gpu_prediction["confidence_percent"]
            - cpu_confidences[category]
        )

        print(
            f"  {category}: "
            f"{confidence_difference:.2f} percentage points"
        )

        assert confidence_difference <= MAX_CONFIDENCE_DIFFERENCE, (
            f"{run_label} confidence difference exceeded "
            f"tolerance for {category}: "
            f"{confidence_difference:.2f} percentage points"
        )


def assert_stable(
    values: list[float],
    median_value: float,
    backend_name: str,
    metric_name: str,
) -> int:
    assert all(math.isfinite(value) and value >= 0 for value in values)

    lower_bound = median_value * (1 - STABILITY_TOLERANCE)
    upper_bound = median_value * (1 + STABILITY_TOLERANCE)
    stable_count = sum(
        lower_bound <= value <= upper_bound
        for value in values
    )

    print(
        f"Measured {metric_name} stability: {stable_count}/7 within "
        f"±20% of median ({lower_bound:.2f}-{upper_bound:.2f} ms)"
    )

    assert stable_count >= MINIMUM_STABLE_RUNS, (
        f"{backend_name} has only {stable_count}/7 measured "
        f"{metric_name} values within ±20% of its median"
    )

    return stable_count


def main() -> None:
    core = Core()
    available_devices = core.available_devices

    print(f"Core().available_devices: {available_devices}")

    if "CPU" not in available_devices:
        print("SKIPPED: OpenVINO CPU is not available on this machine.")
        return

    intel_gpu_devices = []

    for device in available_devices:
        if not device.startswith("GPU"):
            continue

        full_device_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )

        if "intel" in full_device_name.casefold():
            intel_gpu_devices.append((device, full_device_name))

    if not intel_gpu_devices:
        print("SKIPPED: No Intel GPU is available through OpenVINO.")
        return

    gpu_device, gpu_full_device_name = intel_gpu_devices[0]

    print(f"Selected Intel GPU device: {gpu_device}")
    print(f"Selected Intel GPU name: {gpu_full_device_name}")

    cpu_backend = OpenVINOBackend(target_device="CPU")
    gpu_backend = OpenVINOBackend(target_device=gpu_device)

    backends = {
        "CPU": cpu_backend,
        "GPU": gpu_backend,
    }
    phases = [
        ("initialization", INITIALIZATION_RUNS),
        ("warm-up", WARMUP_RUNS),
        ("measured", MEASURED_RUNS),
    ]
    timings = {
        name: {
            phase_name: []
            for phase_name, _ in phases
        }
        for name in backends
    }
    results = {
        name: {
            phase_name: []
            for phase_name, _ in phases
        }
        for name in backends
    }

    for phase_name, run_count in phases:
        print()
        print(f"{phase_name.title()} phase")
        print("-" * (len(phase_name) + len(" phase")))

        for run_index in range(run_count):
            run_number = run_index + 1
            run_label = f"{phase_name.title()} run {run_number}"
            execution_order = (
                ["CPU", "GPU"]
                if run_index % 2 == 0
                else ["GPU", "CPU"]
            )

            print()
            print(
                f"{run_label} execution order: "
                f"{' -> '.join(execution_order)}"
            )

            paired_results = {}

            for name in execution_order:
                backend = backends[name]
                start = time.perf_counter()
                result = backend.run(
                    TaskType.IMAGE_CLASSIFICATION,
                    "test.png",
                )
                total_time_ms = (time.perf_counter() - start) * 1000
                inference_time_ms = result["inference_time_ms"]

                assert len(result["predictions"]) == 5
                assert math.isfinite(total_time_ms)
                assert total_time_ms >= 0
                assert math.isfinite(inference_time_ms)
                assert inference_time_ms >= 0

                timings[name][phase_name].append(
                    (total_time_ms, inference_time_ms)
                )
                results[name][phase_name].append(result)
                paired_results[name] = result

                print(
                    f"{name}: total={total_time_ms:.2f} ms, "
                    f"inference={inference_time_ms:.2f} ms"
                )

            assert_prediction_parity(
                paired_results["CPU"]["predictions"],
                paired_results["GPU"]["predictions"],
                run_label,
            )

    expected_gpu_identity = f"openvino_resnet18_{gpu_device.lower()}"

    assert all(
        result["backend"] == "openvino_resnet18_cpu"
        for phase_name, _ in phases
        for result in results["CPU"][phase_name]
    )
    assert all(
        result["backend"] == expected_gpu_identity
        for phase_name, _ in phases
        for result in results["GPU"][phase_name]
    )

    execution_devices = {
        "CPU": list(
            cpu_backend.compiled_model.get_property("EXECUTION_DEVICES")
        ),
        "GPU": list(
            gpu_backend.compiled_model.get_property("EXECUTION_DEVICES")
        ),
    }

    assert execution_devices["CPU"] == ["CPU"]
    assert execution_devices["GPU"] == [gpu_device]

    identities = {
        "CPU": "openvino_resnet18_cpu",
        "GPU": expected_gpu_identity,
    }

    print()
    print("Benchmark summary")
    print("-----------------")

    for name in ["CPU", "GPU"]:
        initialization_timings = timings[name]["initialization"]
        warmup_timings = timings[name]["warm-up"]
        measured_timings = timings[name]["measured"]
        measured_totals = [
            total_time_ms
            for total_time_ms, _ in measured_timings
        ]
        measured_inference_times = [
            inference_time_ms
            for _, inference_time_ms in measured_timings
        ]
        median_measured_total = median(measured_totals)
        median_measured_inference = median(measured_inference_times)

        assert len(initialization_timings) == INITIALIZATION_RUNS
        assert len(warmup_timings) == WARMUP_RUNS
        assert len(measured_timings) == MEASURED_RUNS

        print()
        print(name)
        print(f"Result identity: {identities[name]}")
        print(
            "Initialization timings (total, inference):",
            [
                (round(total, 2), round(inference, 2))
                for total, inference in initialization_timings
            ],
        )
        print(
            "Warm-up timings (total, inference):",
            [
                (round(total, 2), round(inference, 2))
                for total, inference in warmup_timings
            ],
        )
        print(
            "Measured totals:",
            [round(value, 2) for value in measured_totals],
        )
        print(
            "Measured inference times:",
            [round(value, 2) for value in measured_inference_times],
        )
        print(f"Median measured total: {median_measured_total:.2f} ms")
        print(
            f"Median measured inference: "
            f"{median_measured_inference:.2f} ms"
        )
        print(
            f"Measured total min/max: {min(measured_totals):.2f} / "
            f"{max(measured_totals):.2f} ms"
        )
        print(f"EXECUTION_DEVICES: {execution_devices[name]}")

        assert_stable(
            measured_totals,
            median_measured_total,
            name,
            "total",
        )
        assert_stable(
            measured_inference_times,
            median_measured_inference,
            name,
            "inference",
        )

    print()
    print("OpenVINO CPU vs Intel GPU benchmark passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
