from enum import Enum


class RuntimeType(str, Enum):
    NATIVE = "native"
    PYTORCH = "pytorch"
    OPENVINO = "openvino"


class AcceleratorAPI(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
