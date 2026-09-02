# Standalone OpenVINO NPU ResNet18 smoke test.
import multiprocessing
import time

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from backends.torchvision_classifier import TorchvisionClassifierBackend
from backends.openvino import OpenVINOBackend
from core.task_types import TaskType


def main() -> None:
    try:
        import openvino
        from openvino import Core
    except ImportError as error:
        print(
            "SKIPPED: OpenVINO is not installed; "
            f"the NPU diagnostic cannot run ({error})."
        )
        return

    core = Core()
    available_devices = core.available_devices
    backend = OpenVINOBackend(target_device="NPU")
    backend_info = backend.detect()

    print(f"Core().available_devices: {available_devices}")

    assert backend.target_device == "NPU"
    assert backend_info.name == "OpenVINO NPU Diagnostic"
    assert backend_info.details["target_device"] == "NPU"
    assert backend_info.available == ("NPU" in available_devices)
    assert backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        policy=None,
    ) == 0

    if "NPU" not in available_devices:
        print("SKIPPED: OpenVINO NPU is not available on this machine.")
        return

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    example_input = torch.zeros(1, 3, 224, 224)
    openvino_model = openvino.convert_model(
        model,
        example_input=example_input,
    )
    compiled_model = core.compile_model(
        openvino_model,
        backend.target_device,
    )
    output_layer = compiled_model.output(0)

    image = Image.open("test.png").convert("RGB")
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)

    start = time.perf_counter()
    result = compiled_model(batch.numpy())
    inference_time_ms = (time.perf_counter() - start) * 1000

    prediction = (
        torch.from_numpy(result[output_layer])
        .squeeze(0)
        .softmax(0)
    )
    top_scores, top_ids = prediction.topk(5)

    npu_predictions = []

    for score, class_id in zip(top_scores, top_ids):
        npu_predictions.append(
            {
                "category": weights.meta["categories"][class_id.item()],
                "confidence_percent": round(score.item() * 100, 2),
            }
        )

    assert len(npu_predictions) == 5

    print(f"OpenVINO NPU inference: {inference_time_ms:.2f} ms")
    print("OpenVINO NPU predictions:")

    for item in npu_predictions:
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

    assert len(cpu_predictions) == 5

    npu_categories = [
        item["category"]
        for item in npu_predictions
    ]
    cpu_categories = [
        item["category"]
        for item in cpu_predictions
    ]

    assert npu_categories[0] == cpu_categories[0], (
        "NPU and CPU top-1 categories differ: "
        f"{npu_categories[0]} != {cpu_categories[0]}"
    )

    overlapping_categories = set(npu_categories) & set(cpu_categories)

    assert len(overlapping_categories) >= 4, (
        "NPU and CPU top-5 predictions have insufficient overlap: "
        f"{len(overlapping_categories)} categories"
    )

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    print("Confidence differences for overlapping predictions:")

    for npu_prediction in npu_predictions:
        category = npu_prediction["category"]

        if category not in overlapping_categories:
            continue

        confidence_difference = abs(
            npu_prediction["confidence_percent"]
            - cpu_confidences[category]
        )

        print(
            f"  {category}: "
            f"{confidence_difference:.2f} percentage points"
        )

    execution_devices = list(
        compiled_model.get_property("EXECUTION_DEVICES")
    )

    print(f"Compiled execution devices: {execution_devices}")

    assert execution_devices == ["NPU"]

    print(
        "OpenVINO NPU and Torchvision CPU have matching top-1 "
        f"predictions and {len(overlapping_categories)}/5 "
        "top-5 category overlap."
    )
    print("OpenVINO NPU ResNet18 smoke test passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
