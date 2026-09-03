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


class PyTorchMPSBackend(Backend):

    def __init__(
        self,
        warmup_runs: int = 0,
    ):
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")

        self.warmup_runs = warmup_runs
        self.device = torch.device("mps")
        self.weights = ResNet18_Weights.DEFAULT
        self.model = None
        self._warmup_complete = False
        self._last_input_device = None
        self._last_output_device = None

    @property
    def backend_name(self) -> str:
        return "PyTorch MPS"

    def detect(self) -> BackendInfo:
        details = {
            "mps_built": False,
            "mps_available": False,
            "pytorch_version": torch.__version__,
        }

        if getattr(torch.backends, "mps", None) is None:
            return BackendInfo(
                name=self.backend_name,
                device_type=DeviceType.GPU,
                available=False,
                details=details,
            )

        try:
            mps_built = torch.backends.mps.is_built()
            mps_available = torch.backends.mps.is_available()
        except Exception as error:
            details["error"] = str(error)

            return BackendInfo(
                name=self.backend_name,
                device_type=DeviceType.GPU,
                available=False,
                details=details,
            )

        details.update(
            {
                "mps_built": mps_built,
                "mps_available": mps_available,
            }
        )

        return BackendInfo(
            name=self.backend_name,
            device_type=DeviceType.GPU,
            available=mps_built and mps_available,
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
                "PyTorch MPS backend is not available."
            )

        self.model = resnet18(weights=self.weights)
        self.model.eval()
        self.model.to(self.device)

        parameter_devices = {
            parameter.device.type
            for parameter in self.model.parameters()
        }

        assert parameter_devices == {self.device.type}, (
            "MPS model parameters are not all on the selected device: "
            f"{parameter_devices}"
        )

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            raise NotImplementedError(
                f"PyTorch MPS backend does not support: {task_type}"
            )

        self._load_model()

        with Image.open(payload) as image:
            preprocess = self.weights.transforms()
            batch = preprocess(image.convert("RGB")).unsqueeze(0)

        mps_batch = batch.to(self.device)
        self._last_input_device = mps_batch.device

        assert self._last_input_device.type == self.device.type

        executed_warmup_runs = 0
        warmup_time_ms = 0.0

        if self.warmup_runs > 0 and not self._warmup_complete:
            torch.mps.synchronize()
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                for _ in range(self.warmup_runs):
                    self.model(mps_batch)

            torch.mps.synchronize()
            warmup_time_ms = (
                time.perf_counter() - warmup_start
            ) * 1000
            executed_warmup_runs = self.warmup_runs
            self._warmup_complete = True

        torch.mps.synchronize()
        inference_start = time.perf_counter()

        with torch.inference_mode():
            output = self.model(mps_batch)

        torch.mps.synchronize()
        inference_time_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        self._last_output_device = output.device

        assert self._last_output_device.type == self.device.type

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
            "backend": "pytorch_mps_resnet18",
            "inference_time_ms": round(inference_time_ms, 2),
            "warmup_runs": executed_warmup_runs,
            "warmup_time_ms": round(warmup_time_ms, 2),
            "predictions": predictions,
        }


class RegisteredPyTorchMPSBackend(PyTorchMPSBackend):

    def __init__(self):
        super().__init__(
            warmup_runs=2,
        )


register_backend(RegisteredPyTorchMPSBackend)
