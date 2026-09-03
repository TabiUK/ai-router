import sys
import tempfile
from pathlib import Path
from PIL import Image

AI_ROUTER_PATH = Path(__file__).resolve().parents[2]

if str(AI_ROUTER_PATH) not in sys.path:
    sys.path.insert(0, str(AI_ROUTER_PATH))

from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType


def comfyui_image_to_temp_png(image):
    image_tensor = image[0]
    image_array = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    pil_image = Image.fromarray(image_array)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    )
    temp_path = temp_file.name
    temp_file.close()

    pil_image.save(temp_path)

    return temp_path


class AIRouterDeviceInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("device_info",)
    FUNCTION = "get_device_info"
    CATEGORY = "AI Router"

    def get_device_info(self):
        router = AIRouter()

        lines = [
            "AI Router Devices",
            "-----------------",
        ]

        for backend in router.backends:
            info = backend.detect()

            lines.append("")
            lines.append(f"Device: {info.name}")
            lines.append(f"Type: {info.device_type.value}")
            lines.append(f"Runtime: {info.runtime.value}")
            lines.append(
                "Accelerator API: "
                f"{info.accelerator_api.value if info.accelerator_api is not None else None}"
            )
            lines.append(f"Available: {info.available}")
            lines.append(f"Capabilities: {backend.capabilities()}")

            if info.details:
                lines.append(f"Details: {info.details}")

        return ("\n".join(lines),)


class AIRouterShowDeviceInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device_info": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show_device_info"
    OUTPUT_NODE = True
    CATEGORY = "AI Router"

    def show_device_info(self, device_info):
        print(device_info)
        return {}


class AIRouterImageClassification:
    def __init__(self):
        self.router = AIRouter()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "benchmark_backend": ("STRING", {"default": ""}),
            }
        }


    @classmethod
    def IS_CHANGED(cls, image, benchmark_backend=""):
        return float("nan")

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "backend",
        "predictions",
        "inference_time_ms",
        "execution_time_ms",
    )
    FUNCTION = "classify_image"
    CATEGORY = "AI Router"


    def classify_image(self, image, benchmark_backend=""):
        temp_path = comfyui_image_to_temp_png(image)

        try:
            router = self.router

            task = Task(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                payload=temp_path,
            )

            routed = router.route(task, benchmark_backend=benchmark_backend or None)


            routing = routed["routing"]
            result = routed["result"]

            predictions = "\n".join(
                f'{item["category"]}: {item["confidence_percent"]}%'
                for item in result.get("predictions", [])
            )


            return (
                routing["backend"],
                predictions,
                float(result.get("inference_time_ms", 0.0)),
                float(routing.get("execution_time_ms", 0.0)),
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)


class AIRouterShowClassification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("STRING", {"forceInput": True}),
                "predictions": ("STRING", {"forceInput": True}),
                "inference_time_ms": ("FLOAT", {"forceInput": True}),
                "execution_time_ms": ("FLOAT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show_classification"
    OUTPUT_NODE = True
    CATEGORY = "AI Router"

    def show_classification(
        self,
        backend,
        predictions,
        inference_time_ms,
        execution_time_ms,
    ):
        result = "\n".join(
            [
                "AI Router Classification",
                "------------------------",
                "",
                f"Backend: {backend}",
                "",
                "Predictions:",
                predictions,
                "",
                f"Inference time: {inference_time_ms} ms",
                f"Execution time: {execution_time_ms} ms",
            ]
        )
        print(result)
        return {}


NODE_CLASS_MAPPINGS = {
    "AIRouterDeviceInfo": AIRouterDeviceInfo,
    "AIRouterShowDeviceInfo": AIRouterShowDeviceInfo,
    "AIRouterImageClassification": AIRouterImageClassification,
    "AIRouterShowClassification": AIRouterShowClassification,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIRouterDeviceInfo": "AI Router Device Info",
    "AIRouterShowDeviceInfo": "AI Router Show Device Info",
    "AIRouterImageClassification": "AI Router Image Classification",
    "AIRouterShowClassification": "AI Router Show Classification",
}
