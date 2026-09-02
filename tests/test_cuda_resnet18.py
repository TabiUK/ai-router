# Standalone CUDA ResNet18 diagnostic.
import time
from statistics import median

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from backends.torchvision_classifier import TorchvisionClassifierBackend
from core.task_types import TaskType


WARMUP_RUNS = 2
MEASURED_RUNS = 5
CONFIDENCE_TOLERANCE_PERCENTAGE_POINTS = 0.1


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print(
            "SKIPPED: CUDA is not available through the installed "
            "PyTorch build."
        )
        return

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    device_name = torch.cuda.get_device_name(device)
    print(f"Selected CUDA device index: {device_index}")
    print(f"Selected CUDA device name: {device_name}")

    initialization_start = time.perf_counter()

    weights = ResNet18_Weights.DEFAULT
    cuda_model = resnet18(weights=weights)
    cuda_model.eval()
    cuda_model.to(device)

    image = Image.open("test.png").convert("RGB")
    preprocess = weights.transforms()
    cpu_input = preprocess(image).unsqueeze(0)
    cuda_input = cpu_input.to(device)

    torch.cuda.synchronize(device)
    initialization_time_ms = (
        time.perf_counter() - initialization_start
    ) * 1000

    parameter_devices = {
        parameter.device
        for parameter in cuda_model.parameters()
    }

    assert parameter_devices == {device}, (
        "CUDA model parameters are not all on the selected device: "
        f"{parameter_devices}"
    )
    assert cuda_input.device == device

    print(f"CUDA initialization time: {initialization_time_ms:.2f} ms")
    print(f"CUDA model parameter device: {device}")
    print(f"CUDA input device: {cuda_input.device}")

    with torch.inference_mode():
        torch.cuda.synchronize(device)
        cold_start = time.perf_counter()
        cold_output = cuda_model(cuda_input)
        torch.cuda.synchronize(device)
        cold_inference_time_ms = (
            time.perf_counter() - cold_start
        ) * 1000

        assert cold_output.device == device

        print(
            "CUDA first inference before warm-up: "
            f"{cold_inference_time_ms:.2f} ms"
        )

        warmup_times_ms = []

        for _ in range(WARMUP_RUNS):
            torch.cuda.synchronize(device)
            warmup_start = time.perf_counter()
            warmup_output = cuda_model(cuda_input)
            torch.cuda.synchronize(device)
            warmup_times_ms.append(
                (time.perf_counter() - warmup_start) * 1000
            )

            assert warmup_output.device == device

        measured_times_ms = []
        cuda_output = None

        for _ in range(MEASURED_RUNS):
            torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            cuda_output = cuda_model(cuda_input)
            torch.cuda.synchronize(device)
            measured_times_ms.append(
                (time.perf_counter() - inference_start) * 1000
            )

            assert cuda_output.device == device

    assert cuda_output is not None

    prediction = cuda_output.squeeze(0).softmax(0).cpu()
    top_scores, top_ids = prediction.topk(5)
    cuda_predictions = []

    for score, class_id in zip(top_scores, top_ids):
        cuda_predictions.append(
            {
                "category": weights.meta["categories"][class_id.item()],
                "confidence_percent": round(score.item() * 100, 2),
            }
        )

    print(f"CUDA warm-up timings: {warmup_times_ms}")
    print(f"CUDA measured inference timings: {measured_times_ms}")
    print(
        "CUDA median measured inference: "
        f"{median(measured_times_ms):.2f} ms"
    )
    print("CUDA predictions:")

    for item in cuda_predictions:
        print(
            f"  {item['category']}: "
            f"{item['confidence_percent']:.2f}%"
        )

    cpu_backend = TorchvisionClassifierBackend()
    cpu_result = cpu_backend.run(
        TaskType.IMAGE_CLASSIFICATION,
        "test.png",
    )
    cpu_predictions = cpu_result["predictions"]

    assert len(cuda_predictions) == 5
    assert len(cpu_predictions) == 5

    cuda_categories = [
        item["category"]
        for item in cuda_predictions
    ]
    cpu_categories = [
        item["category"]
        for item in cpu_predictions
    ]

    assert cuda_categories[0] == cpu_categories[0], (
        "CUDA and CPU top-1 categories differ: "
        f"{cuda_categories[0]} != {cpu_categories[0]}"
    )
    assert set(cuda_categories) == set(cpu_categories), (
        "CUDA and CPU top-five category sets differ: "
        f"{cuda_categories} != {cpu_categories}"
    )

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    print("Confidence differences by category:")

    for cuda_prediction in cuda_predictions:
        category = cuda_prediction["category"]
        confidence_difference = abs(
            cuda_prediction["confidence_percent"]
            - cpu_confidences[category]
        )

        print(
            f"  {category}: "
            f"{confidence_difference:.2f} percentage points"
        )

        assert (
            confidence_difference
            <= CONFIDENCE_TOLERANCE_PERCENTAGE_POINTS
        ), (
            "Confidence difference exceeded tolerance for "
            f"{category}: {confidence_difference:.2f} percentage points"
        )

    print("CUDA ResNet18 diagnostic passed.")


if __name__ == "__main__":
    main()
