from typing import Any

import torch
import time

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from core.backend import Backend, BackendInfo
from core.backend_registry import register_backend
from core.device_types import DeviceType
from core.policy import RoutingPolicy
from core.task_types import TaskType
from core.runtime_types import RuntimeType

class TorchvisionClassifierBackend(Backend):

    def __init__(self):
        self.weights = ResNet18_Weights.DEFAULT
        self.model = None

    def detect(self) -> BackendInfo:
        return BackendInfo(
            name="Torchvision ResNet18 CPU",
            device_type=DeviceType.CPU,
            runtime=RuntimeType.PYTORCH,
            accelerator_api=None,
            available=True,
            details={
                "model": "ResNet18",
            },
        )

    def capabilities(self) -> list[str]:
        return [
            TaskType.IMAGE_CLASSIFICATION.value,
        ]

    def score(self, task_type: str, policy: RoutingPolicy) -> int:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            return 0

        if policy == RoutingPolicy.PERFORMANCE:
            return 40

        if policy == RoutingPolicy.BALANCED:
            return 60

        if policy == RoutingPolicy.LOW_POWER:
            return 70

        return 0

    def _load_model(self):
        if self.model is None:
            self.model = resnet18(weights=self.weights)
            self.model.eval()

    def run(self, task_type: str, payload: Any) -> Any:
        if task_type != TaskType.IMAGE_CLASSIFICATION:
            raise NotImplementedError(
                f"Torchvision classifier does not support: {task_type}"
            )

        self._load_model()

        image = Image.open(payload).convert("RGB")
        preprocess = self.weights.transforms()
        batch = preprocess(image).unsqueeze(0)

        start = time.perf_counter()

        with torch.no_grad():
            prediction = self.model(batch).squeeze(0).softmax(0)

        inference_time_ms = (time.perf_counter() - start) * 1000

        top_scores, top_ids = prediction.topk(5)

        predictions = []

        for score, class_id in zip(top_scores, top_ids):
            predictions.append(
                {
                    "category": self.weights.meta["categories"][class_id.item()],
                    "confidence_percent": round(score.item() * 100, 2),
                }
            )

        return {
            "backend": "torchvision_resnet18",
            "inference_time_ms": round(inference_time_ms, 2),
            "predictions": predictions,
        }

register_backend(TorchvisionClassifierBackend)
