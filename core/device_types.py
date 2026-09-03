from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    ACCELERATOR = "accelerator"
