import time
from typing import Any

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from core.backend import Backend, BackendInfo
from core.backend_registry import register_backend
from core.policy import RoutingPolicy
from core.task_types import TaskType


class OpenVINOBackend(Backend):

    def __init__(
        self,
        target_device: str = "CPU",
        warmup_runs: int = 0,
    ):
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")

        self.target_device = target_device.upper()
        self.warmup_runs = warmup_runs
        self._warmup_complete = False
        self.weights = ResNet18_Weights.DEFAULT
        self.compiled_model = None
        self.output_layer = None

    @property
    def backend_name(self) -> str:
        if self.target_device == "CPU":
            return "OpenVINO"

        return f"OpenVINO {self.target_device} Diagnostic"

    def detect(self) -> BackendInfo:
        try:
            import openvino
            from openvino import Core
        except ImportError:
            return BackendInfo(
                name=self.backend_name,
                device_type="openvino",
                available=False,
                details={
                    "version": None,
                    "available_devices": [],
                    "target_device": self.target_device,
                },
            )

        version = getattr(openvino, "__version__", None)

        try:
            available_devices = Core().available_devices
        except Exception as error:
            return BackendInfo(
                name=self.backend_name,
                device_type="openvino",
                available=False,
                details={
                    "version": version,
                    "available_devices": [],
                    "target_device": self.target_device,
                    "error": str(error),
                },
            )

        return BackendInfo(
            name=self.backend_name,
            device_type="openvino",
            available=self.target_device in available_devices,
            details={
                "version": version,
                "available_devices": available_devices,
                "target_device": self.target_device,
            },
        )

    def capabilities(self) -> list[str]:
        return [
            TaskType.IMAGE_CLASSIFICATION.value,
        ]

    def score(self, task_type: str, policy: RoutingPolicy) -> int:
        if self.target_device != "CPU":
            return 0

        if task_type != TaskType.IMAGE_CLASSIFICATION:
            return 0

        if policy == RoutingPolicy.PERFORMANCE:
            return 38

        if policy == RoutingPolicy.BALANCED:
            return 58

        if policy == RoutingPolicy.LOW_POWER:
            return 63

        return 0

    def _load_model(self) -> None:
        if self.compiled_model is not None:
            return

        import openvino
        from openvino import Core

        model = resnet18(weights=self.weights)
        model.eval()

        example_input = torch.zeros(1, 3, 224, 224)
        openvino_model = openvino.convert_model(
            model,
            example_input=example_input,
        )

        self.compiled_model = Core().compile_model(
            openvino_model,
            self.target_device,
        )
        self.output_layer = self.compiled_model.output(0)

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            raise NotImplementedError(
                f"OpenVINO backend does not support: {task_type}"
            )

        self._load_model()

        image = Image.open(payload).convert("RGB")
        preprocess = self.weights.transforms()
        batch = preprocess(image).unsqueeze(0)

        executed_warmup_runs = 0
        warmup_time_ms = 0.0

        if self.warmup_runs > 0 and not self._warmup_complete:
            warmup_start = time.perf_counter()

            for _ in range(self.warmup_runs):
                self.compiled_model(batch.numpy())

            warmup_time_ms = (
                time.perf_counter() - warmup_start
            ) * 1000
            executed_warmup_runs = self.warmup_runs
            self._warmup_complete = True

        start = time.perf_counter()

        result = self.compiled_model(batch.numpy())

        inference_time_ms = (time.perf_counter() - start) * 1000

        prediction = (
            torch.from_numpy(result[self.output_layer])
            .squeeze(0)
            .softmax(0)
        )

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

        backend_result = {
            "backend": f"openvino_resnet18_{self.target_device.lower()}",
            "inference_time_ms": round(inference_time_ms, 2),
            "predictions": predictions,
        }

        if self.warmup_runs > 0:
            backend_result["warmup_runs"] = executed_warmup_runs
            backend_result["warmup_time_ms"] = round(warmup_time_ms, 2)

        return backend_result


def find_intel_gpu_device(core) -> tuple[str, str] | None:
    for device in core.available_devices:
        if not device.startswith("GPU"):
            continue

        full_device_name = core.get_property(
            device,
            "FULL_DEVICE_NAME",
        )

        if "intel" in full_device_name.casefold():
            return device, full_device_name

    return None


class OpenVINOIntelGPUBackend(OpenVINOBackend):

    def __init__(self, warmup_runs: int = 2):
        try:
            from openvino import Core
        except ImportError:
            selected_device = None
        else:
            selected_device = find_intel_gpu_device(Core())

        if selected_device is None:
            target_device = "GPU.INTEL_UNAVAILABLE"
            self.full_device_name = None
        else:
            target_device, self.full_device_name = selected_device

        super().__init__(
            target_device=target_device,
            warmup_runs=warmup_runs,
        )

    @property
    def backend_name(self) -> str:
        return "OpenVINO Intel GPU"

    def detect(self) -> BackendInfo:
        info = super().detect()
        info.details["full_device_name"] = self.full_device_name
        return info

    def score(self, task_type: str, policy: RoutingPolicy) -> int:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            return 0

        if policy == RoutingPolicy.PERFORMANCE:
            return 37

        if policy == RoutingPolicy.BALANCED:
            return 57

        if policy == RoutingPolicy.LOW_POWER:
            return 0

        return 0


register_backend(OpenVINOBackend)
register_backend(OpenVINOIntelGPUBackend)
