# Standalone OpenVINO GPU ResNet18 smoke test.
import multiprocessing

from openvino import Core

from backends.openvino import OpenVINOBackend
from backends.torchvision_classifier import TorchvisionClassifierBackend
from core.task_types import TaskType


def main() -> None:
    core = Core()
    available_devices = core.available_devices

    print(f"Core().available_devices: {available_devices}")

    gpu_devices = [
        device
        for device in available_devices
        if device.startswith("GPU")
    ]
    intel_gpu_devices = []

    for device in gpu_devices:
        full_device_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )

        if "intel" in full_device_name.casefold():
            intel_gpu_devices.append((device, full_device_name))

    if not intel_gpu_devices:
        print("SKIPPED: No Intel GPU is available through OpenVINO.")
        return

    target_device, full_device_name = intel_gpu_devices[0]

    print(f"Selected OpenVINO device: {target_device}")
    print(f"Selected device name: {full_device_name}")

    backend = OpenVINOBackend(target_device=target_device)
    backend_info = backend.detect()

    assert backend.target_device == target_device
    assert backend_info.name == (
        f"OpenVINO {target_device} Diagnostic"
    )
    assert backend_info.details["target_device"] == target_device
    assert backend_info.available == (
        target_device in available_devices
    )
    assert backend.score(
        TaskType.IMAGE_CLASSIFICATION,
        policy=None,
    ) == 0

    gpu_result = backend.run(
        TaskType.IMAGE_CLASSIFICATION,
        "test.png",
    )
    gpu_predictions = gpu_result["predictions"]

    expected_result_identity = (
        f"openvino_resnet18_{target_device.lower()}"
    )

    assert gpu_result["backend"] == expected_result_identity
    assert len(gpu_predictions) == 5

    print(
        f"OpenVINO GPU inference: "
        f"{gpu_result['inference_time_ms']:.2f} ms"
    )
    print("OpenVINO GPU predictions:")

    for item in gpu_predictions:
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

    gpu_categories = [
        item["category"]
        for item in gpu_predictions
    ]
    cpu_categories = [
        item["category"]
        for item in cpu_predictions
    ]

    assert gpu_categories[0] == cpu_categories[0], (
        "GPU and CPU top-1 categories differ: "
        f"{gpu_categories[0]} != {cpu_categories[0]}"
    )

    assert set(gpu_categories) == set(cpu_categories), (
        "GPU and CPU top-5 category sets differ: "
        f"{set(gpu_categories)} != {set(cpu_categories)}"
    )

    cpu_confidences = {
        item["category"]: item["confidence_percent"]
        for item in cpu_predictions
    }

    print("Confidence differences by category:")

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

        assert confidence_difference <= 0.1, (
            "Confidence difference exceeded tolerance for "
            f"{category}: "
            f"{confidence_difference:.2f} percentage points"
        )

    execution_devices = list(
        backend.compiled_model.get_property("EXECUTION_DEVICES")
    )

    print(f"Compiled execution devices: {execution_devices}")

    assert execution_devices == [target_device]

    print(
        "OpenVINO GPU predictions match Torchvision CPU within "
        "0.1 percentage points."
    )
    print("OpenVINO GPU ResNet18 smoke test passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
