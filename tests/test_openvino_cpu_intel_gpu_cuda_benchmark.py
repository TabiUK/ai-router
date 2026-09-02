# Standalone four-way ResNet18 hardware benchmark.
import math
import multiprocessing
import time
from statistics import median

import torch
from openvino import Core
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from backends.openvino import OpenVINOBackend, find_intel_gpu_device
from backends.torchvision_classifier import TorchvisionClassifierBackend
from core.task_types import TaskType


INITIALIZATION_RUNS = 1
WARMUP_RUNS = 2
MEASURED_RUNS = 7
MAX_CONFIDENCE_DIFFERENCE = 0.1
STABILITY_TOLERANCE = 0.2
ABSOLUTE_STABILITY_TOLERANCE_MS = 2.0
MINIMUM_STABLE_RUNS = 5

TORCHVISION_CPU = "Torchvision ResNet18 CPU"
OPENVINO_CPU = "OpenVINO CPU"
OPENVINO_INTEL_GPU = "OpenVINO Intel GPU"
PYTORCH_CUDA = "PyTorch CUDA"


class CUDABenchmarkPath:

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.device = torch.device("cuda", device_index)
        self.device_name = torch.cuda.get_device_name(self.device)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = None

    def _load_model(self) -> None:
        if self.model is not None:
            return

        self.model = resnet18(weights=self.weights)
        self.model.eval()
        self.model.to(self.device)

        parameter_devices = {
            parameter.device
            for parameter in self.model.parameters()
        }

        assert parameter_devices == {self.device}, (
            "CUDA model parameters are not all on the selected device: "
            f"{parameter_devices}"
        )

    def run(self, task_type: str, payload: str) -> dict:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            raise NotImplementedError(
                f"CUDA benchmark path does not support: {task_type}"
            )

        self._load_model()

        with Image.open(payload) as image:
            preprocess = self.weights.transforms()
            batch = preprocess(image.convert("RGB")).unsqueeze(0)

        cuda_batch = batch.to(self.device)

        assert cuda_batch.device == self.device

        torch.cuda.synchronize(self.device)
        inference_start = time.perf_counter()

        with torch.inference_mode():
            output = self.model(cuda_batch)

        torch.cuda.synchronize(self.device)
        inference_time_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        assert output.device == self.device

        prediction = output.squeeze(0).softmax(0).cpu()
        top_scores, top_ids = prediction.topk(5)
        predictions = []

        for score, class_id in zip(top_scores, top_ids):
            predictions.append(
                {
                    "category": self.weights.meta["categories"][
                        class_id.item()
                    ],
                    "confidence_percent": round(score.item() * 100, 2),
                }
            )

        return {
            "backend": f"pytorch_cuda_resnet18_cuda_{self.device_index}",
            "inference_time_ms": round(inference_time_ms, 2),
            "predictions": predictions,
        }


def assert_prediction_parity(
    reference_predictions: list[dict],
    candidate_predictions: list[dict],
    candidate_name: str,
    run_label: str,
) -> None:
    assert len(reference_predictions) == 5
    assert len(candidate_predictions) == 5

    reference_categories = [
        item["category"]
        for item in reference_predictions
    ]
    candidate_categories = [
        item["category"]
        for item in candidate_predictions
    ]

    assert reference_categories[0] == candidate_categories[0], (
        f"{run_label} Torchvision CPU and {candidate_name} top-1 "
        f"categories differ: {reference_categories[0]} != "
        f"{candidate_categories[0]}"
    )
    assert set(reference_categories) == set(candidate_categories), (
        f"{run_label} Torchvision CPU and {candidate_name} top-five "
        f"category sets differ: {set(reference_categories)} != "
        f"{set(candidate_categories)}"
    )

    reference_confidences = {
        item["category"]: item["confidence_percent"]
        for item in reference_predictions
    }

    print(f"{run_label} {candidate_name} confidence differences:")

    for candidate_prediction in candidate_predictions:
        category = candidate_prediction["category"]
        confidence_difference = abs(
            candidate_prediction["confidence_percent"]
            - reference_confidences[category]
        )

        print(
            f"  {category}: "
            f"{confidence_difference:.2f} percentage points"
        )

        assert confidence_difference <= MAX_CONFIDENCE_DIFFERENCE, (
            f"{run_label} {candidate_name} confidence difference "
            f"exceeded tolerance for {category}: "
            f"{confidence_difference:.2f} percentage points"
        )


def assert_stable(
    values: list[float],
    median_value: float,
    path_name: str,
    metric_name: str,
) -> int:
    assert all(math.isfinite(value) and value >= 0 for value in values)

    allowed_deviation = max(
        median_value * STABILITY_TOLERANCE,
        ABSOLUTE_STABILITY_TOLERANCE_MS,
    )
    lower_bound = median_value - allowed_deviation
    upper_bound = median_value + allowed_deviation
    stable_count = sum(
        lower_bound <= value <= upper_bound
        for value in values
    )

    print(
        f"Measured {metric_name} stability: "
        f"{stable_count}/{len(values)} within allowed deviation "
        f"of median (±{allowed_deviation:.2f} ms; "
        f"{lower_bound:.2f}-{upper_bound:.2f} ms)"
    )

    assert stable_count >= MINIMUM_STABLE_RUNS, (
        f"{path_name} has only {stable_count}/{len(values)} measured "
        f"{metric_name} values within its allowed median deviation"
    )

    return stable_count


def main() -> None:
    core = Core()
    available_devices = core.available_devices

    print(f"Core().available_devices: {available_devices}")

    if "CPU" not in available_devices:
        print("SKIPPED: OpenVINO CPU is not available on this machine.")
        return

    intel_gpu = find_intel_gpu_device(core)

    if intel_gpu is None:
        print("SKIPPED: No Intel GPU is available through OpenVINO.")
        return

    if not torch.cuda.is_available():
        print(
            "SKIPPED: CUDA is not available through the installed "
            "PyTorch build."
        )
        return

    cuda_device_count = torch.cuda.device_count()

    if cuda_device_count < 1:
        print("SKIPPED: PyTorch reports no CUDA devices.")
        return

    intel_gpu_device, intel_gpu_name = intel_gpu
    cuda_device_index = 0
    cuda_device = torch.device("cuda", cuda_device_index)
    torch.cuda.set_device(cuda_device)
    cuda_device_name = torch.cuda.get_device_name(cuda_device)

    print(f"Selected Intel GPU device: {intel_gpu_device}")
    print(f"Selected Intel GPU name: {intel_gpu_name}")
    print(f"CUDA device count: {cuda_device_count}")
    print(f"Selected CUDA device index: {cuda_device_index}")
    print(f"Selected CUDA device name: {cuda_device_name}")

    torchvision_cpu_path = TorchvisionClassifierBackend()
    cpu_path = OpenVINOBackend(
        target_device="CPU",
        warmup_runs=0,
    )
    intel_gpu_path = OpenVINOBackend(
        target_device=intel_gpu_device,
        warmup_runs=0,
    )
    cuda_path = CUDABenchmarkPath(
        device_index=cuda_device_index,
    )

    paths = {
        TORCHVISION_CPU: torchvision_cpu_path,
        OPENVINO_CPU: cpu_path,
        OPENVINO_INTEL_GPU: intel_gpu_path,
        PYTORCH_CUDA: cuda_path,
    }
    path_names = list(paths)
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
        for name in paths
    }
    results = {
        name: {
            phase_name: []
            for phase_name, _ in phases
        }
        for name in paths
    }

    for phase_name, run_count in phases:
        print()
        print(f"{phase_name.title()} phase")
        print("-" * (len(phase_name) + len(" phase")))

        for run_index in range(run_count):
            run_number = run_index + 1
            run_label = f"{phase_name.title()} run {run_number}"
            order_offset = run_index % len(path_names)
            execution_order = (
                path_names[order_offset:]
                + path_names[:order_offset]
            )

            print()
            print(
                f"{run_label} execution order: "
                f"{' -> '.join(execution_order)}"
            )

            paired_results = {}

            for name in execution_order:
                path = paths[name]
                total_start = time.perf_counter()
                result = path.run(
                    TaskType.IMAGE_CLASSIFICATION,
                    "test.png",
                )
                total_time_ms = (
                    time.perf_counter() - total_start
                ) * 1000
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

            reference_predictions = paired_results[
                TORCHVISION_CPU
            ]["predictions"]

            for candidate_name in (
                OPENVINO_CPU,
                OPENVINO_INTEL_GPU,
                PYTORCH_CUDA,
            ):
                assert_prediction_parity(
                    reference_predictions,
                    paired_results[candidate_name]["predictions"],
                    candidate_name,
                    run_label,
                )

    expected_intel_gpu_identity = (
        f"openvino_resnet18_{intel_gpu_device.lower()}"
    )
    expected_cuda_identity = (
        f"pytorch_cuda_resnet18_cuda_{cuda_device_index}"
    )

    assert all(
        result["backend"] == "torchvision_resnet18"
        for phase_name, _ in phases
        for result in results[TORCHVISION_CPU][phase_name]
    )
    assert all(
        result["backend"] == "openvino_resnet18_cpu"
        for phase_name, _ in phases
        for result in results[OPENVINO_CPU][phase_name]
    )
    assert all(
        result["backend"] == expected_intel_gpu_identity
        for phase_name, _ in phases
        for result in results[OPENVINO_INTEL_GPU][phase_name]
    )
    assert all(
        result["backend"] == expected_cuda_identity
        for phase_name, _ in phases
        for result in results[PYTORCH_CUDA][phase_name]
    )

    execution_devices = {
        OPENVINO_CPU: list(
            cpu_path.compiled_model.get_property("EXECUTION_DEVICES")
        ),
        OPENVINO_INTEL_GPU: list(
            intel_gpu_path.compiled_model.get_property(
                "EXECUTION_DEVICES"
            )
        ),
    }

    assert execution_devices[OPENVINO_CPU] == ["CPU"]
    assert execution_devices[OPENVINO_INTEL_GPU] == [intel_gpu_device]

    identities = {
        TORCHVISION_CPU: "torchvision_resnet18",
        OPENVINO_CPU: "openvino_resnet18_cpu",
        OPENVINO_INTEL_GPU: expected_intel_gpu_identity,
        PYTORCH_CUDA: expected_cuda_identity,
    }

    print()
    print("Benchmark summary")
    print("-----------------")

    for name in path_names:
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
        median_measured_inference = median(
            measured_inference_times
        )

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
            "Median measured inference: "
            f"{median_measured_inference:.2f} ms"
        )
        print(
            f"Measured total min/max: {min(measured_totals):.2f} / "
            f"{max(measured_totals):.2f} ms"
        )
        print(
            "Measured inference min/max: "
            f"{min(measured_inference_times):.2f} / "
            f"{max(measured_inference_times):.2f} ms"
        )

        if name in execution_devices:
            print(f"EXECUTION_DEVICES: {execution_devices[name]}")
        elif name == PYTORCH_CUDA:
            print(f"CUDA device: {cuda_device}")
            print(f"CUDA device name: {cuda_device_name}")

        print("Predictions:")

        for prediction in results[name]["measured"][-1]["predictions"]:
            print(
                f"  {prediction['category']}: "
                f"{prediction['confidence_percent']:.2f}%"
            )

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
    print("Four-way ResNet18 benchmark passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
