# AI Router Requirements

`README.md` is the source of truth for current project behavior. This document
collects only supported versions, runtime requirements, and important
compatibility constraints.

## Python

The packaged project supports:

```text
Python >=3.11,<3.12
```

Use 64-bit Python 3.11 for normal installations. The RunPod RTX 4090 validation
used an externally provisioned Python 3.12.3 environment; that validation does
not widen the package's supported Python range.

## Pinned Python packages

The base dependencies declared in `pyproject.toml` are:

```text
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
pillow
```

Pillow is required but is not pinned to an exact version.

PyTorch 2.2.2 and Torchvision 0.17.2 are a matched pair. NumPy remains pinned
to 1.26.4 because the validated PyTorch 2.2.2 environments produced ABI
compatibility warnings with NumPy 2.x.

## Windows NVIDIA CUDA package variants

The validated Windows CUDA variants are:

```text
torch==2.2.2+cu121
torchvision==0.17.2+cu121
compiled CUDA 12.1
```

These CUDA-enabled wheels must be installed before AI Router. Installing the
project first may select the CPU-only PyTorch wheel variant. The public version
pins remain `2.2.2` and `0.17.2`; the `+cu121` suffix identifies the Windows
CUDA wheel build.

The externally provisioned RunPod RTX 4090 validation used Python 3.12.3,
PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128, and CUDA build 12.8. Those
versions are evidence for that environment and do not replace the versions
pinned by this project or widen packaged Python support beyond `>=3.11,<3.12`.

## Optional OpenVINO

OpenVINO is an optional dependency:

```text
openvino==2025.4.1
```

The `openvino` extra enables the OpenVINO runtime without making it mandatory
for CUDA-only, MPS-only, or base CPU installations.

## Runtime and hardware requirements

### Torchvision CPU

The Torchvision ResNet18 CPU backend requires the base PyTorch, Torchvision,
Pillow, and NumPy dependencies. It is the always-available image-classification
candidate when those dependencies import successfully.

### NVIDIA CUDA

The PyTorch CUDA backend requires:

- a compatible NVIDIA GPU;
- a compatible NVIDIA driver;
- CUDA-enabled PyTorch and Torchvision wheels;
- `torch.cuda.is_available()` to be true; and
- at least one CUDA device reported by PyTorch.

The registered backend currently uses CUDA device 0. It supports ImageNet image
classification with pretrained Torchvision ResNet18. The wheel-based Windows
setup does not require a separate CUDA Toolkit installation.

When CUDA is unavailable, the CUDA backend remains unavailable without
preventing CPU, OpenVINO, or MPS backends from loading.

### Apple MPS

The PyTorch MPS backend requires Apple hardware and a PyTorch build for which:

```text
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
```

MPS uses the base PyTorch/Torchvision packages and requires no additional
Python dependency. Availability must be detected at runtime and must not be
inferred from the operating system name alone.

### OpenVINO

OpenVINO CPU requires the optional OpenVINO package. Intel GPU execution also
requires compatible Intel graphics hardware, operating-system drivers, and an
OpenVINO runtime/plugin that exposes the device.

Installing the OpenVINO Python package does not install Intel GPU or NPU system
drivers. Intel GPU discovery is dynamic and must not assume a fixed OpenVINO
device ID. OpenVINO NPU is diagnostic-only and is not a routable production
backend.

## Model and network requirements

Current real inference uses Torchvision's pretrained ResNet18 ImageNet weights
and preprocessing. Internet access is required for the first model download
unless the weights are already cached. After dependencies and weights are
cached, inference can run locally.

Official Python.org installations on macOS may require the bundled certificate
setup before HTTPS model downloads work. Do not disable TLS certificate
verification as a workaround.

## Validated hardware scope

Current real-hardware validation includes:

- Intel x86_64 macOS: Torchvision CPU and OpenVINO CPU;
- Apple M1 Pro arm64 macOS: Torchvision CPU and PyTorch MPS;
- 64-bit Windows: Intel Iris Xe through OpenVINO and NVIDIA RTX A1000 Laptop GPU
  through PyTorch CUDA;
- externally provisioned 64-bit RunPod Linux: NVIDIA GeForce RTX 4090 through
  PyTorch 2.10.0+cu128 with CUDA build 12.8. This is validation evidence
  only and does not widen the packaged Python support range.

Availability and performance on other systems depend on hardware, drivers,
framework wheels, runtime plugins, and system load. A validated device family
or timing result is not a universal compatibility or performance guarantee.
