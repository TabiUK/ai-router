import time
from typing import Any

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from core.backend import Backend, BackendInfo
from core.backend_registry import register_backend
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.task_types import TaskType
from core.runtime_types import AcceleratorAPI, RuntimeType

class PyTorchCUDABackend(Backend):

    def __init__(
        self,
        device_index: int = 0,
        warmup_runs: int = 0,
    ):
        if device_index < 0:
            raise ValueError("device_index must be non-negative")

        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")

        self.device_index = device_index
        self.warmup_runs = warmup_runs
        self.device = torch.device("cuda", device_index)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = None
        self._warmup_complete = False
        self._last_input_device = None
        self._last_output_device = None

    @property
    def backend_name(self) -> str:
        return "PyTorch CUDA"

    def detect(self) -> BackendInfo:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        details = {
            "device_index": self.device_index,
            "device_count": device_count,
            "device_name": None,
            "compute_capability": None,
            "vram_bytes": None,
            "pytorch_version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
        }

        if (
            not cuda_available
            or self.device_index >= device_count
        ):
            return BackendInfo(
                name=self.backend_name,
                device_type=DeviceType.GPU,
                runtime=RuntimeType.PYTORCH,
                accelerator_api=AcceleratorAPI.CUDA,
                available=False,
                details=details,
            )

        try:
            device_name = torch.cuda.get_device_name(self.device_index)
            compute_capability = torch.cuda.get_device_capability(
                self.device_index
            )
            device_properties = torch.cuda.get_device_properties(
                self.device_index
            )
        except Exception as error:
            details["error"] = str(error)

            return BackendInfo(
                name=self.backend_name,
                device_type=DeviceType.GPU,
                runtime=RuntimeType.PYTORCH,
                accelerator_api=AcceleratorAPI.CUDA,
                available=False,
                details=details,
            )

        details.update(
            {
                "device_name": device_name,
                "compute_capability": compute_capability,
                "vram_bytes": device_properties.total_memory,
            }
        )

        return BackendInfo(
            name=self.backend_name,
            device_type=DeviceType.GPU,
            runtime=RuntimeType.PYTORCH,
            accelerator_api=AcceleratorAPI.CUDA,
            available=True,
            details=details,
        )

    def capabilities(self) -> list[str]:
        return [
            TaskType.IMAGE_CLASSIFICATION.value,
        ]

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

    def _load_model(self) -> None:
        if self.model is not None:
            return

        info = self.detect()

        if not info.available:
            raise RuntimeError(
                "PyTorch CUDA backend is not available for "
                f"CUDA device index {self.device_index}."
            )

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

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            raise NotImplementedError(
                f"PyTorch CUDA backend does not support: {task_type}"
            )

        self._load_model()

        with Image.open(payload) as image:
            preprocess = self.weights.transforms()
            batch = preprocess(image.convert("RGB")).unsqueeze(0)

        cuda_batch = batch.to(self.device)
        self._last_input_device = cuda_batch.device

        assert self._last_input_device == self.device

        executed_warmup_runs = 0
        warmup_time_ms = 0.0

        if self.warmup_runs > 0 and not self._warmup_complete:
            torch.cuda.synchronize(self.device)
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                for _ in range(self.warmup_runs):
                    self.model(cuda_batch)

            torch.cuda.synchronize(self.device)
            warmup_time_ms = (
                time.perf_counter() - warmup_start
            ) * 1000
            executed_warmup_runs = self.warmup_runs
            self._warmup_complete = True

        torch.cuda.synchronize(self.device)
        inference_start = time.perf_counter()

        with torch.inference_mode():
            output = self.model(cuda_batch)

        torch.cuda.synchronize(self.device)
        inference_time_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        self._last_output_device = output.device

        assert self._last_output_device == self.device

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
            "backend": "pytorch_cuda_resnet18",
            "inference_time_ms": round(inference_time_ms, 2),
            "warmup_runs": executed_warmup_runs,
            "warmup_time_ms": round(warmup_time_ms, 2),
            "predictions": predictions,
        }


class RegisteredPyTorchCUDABackend(PyTorchCUDABackend):

    def __init__(self):
        super().__init__(
            device_index=0,
            warmup_runs=2,
        )


register_backend(RegisteredPyTorchCUDABackend)
