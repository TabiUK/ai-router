# AI Router

AI Router is an experimental Python routing layer for selecting an available execution backend for an AI task. It combines policy-specific backend preferences with recent, backend-specific timing history and keeps device discovery, routing, execution, and integrations separated.

## Status

The project is under active development and is not yet a stable production API. Its real inference workload is currently limited to ImageNet image classification with pretrained Torchvision ResNet18. Benchmark history is held in memory for the lifetime of an `AIRouter` instance; persistent storage and broader model/task support are not implemented.

## Current backends

| Backend | Current role |
| --- | --- |
| Torchvision ResNet18 CPU | Always-available CPU image classification |
| OpenVINO | Optional OpenVINO CPU image classification |
| OpenVINO Intel GPU | Optional image classification when OpenVINO detects an Intel GPU |
| PyTorch CUDA | Image classification on CUDA device 0 when CUDA-enabled PyTorch detects a compatible NVIDIA GPU |
| PyTorch MPS | Image classification when PyTorch reports Apple MPS as built and available |
| Generic CPU | Simple `general` and `classification` fallback used by the core routing examples |

The mock accelerator is test-only and does not self-register. OpenVINO NPU, Core ML, image generation, language models, and distributed or remote execution are not currently supported routing targets.

The `PyTorch MPS` backend supports `TaskType.IMAGE_CLASSIFICATION` using
Torchvision pretrained ResNet18 with ImageNet preprocessing. Its stable backend
name is `"PyTorch MPS"` and its stable result identity is
`"pytorch_mps_resnet18"`. It is automatically available when
`torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` both
report true. Current policy scores are `PERFORMANCE = 37`, `BALANCED = 57`,
and `LOW_POWER = 0`; LOW_POWER is zero because no power-efficiency advantage
has been measured. The registered production backend performs two warm-up runs
once per backend instance.

## Routing

AI Router provides three policies:

- `performance`: prioritizes execution performance.
- `balanced`: the default compromise between backend preferences.
- `low_power`: prefers backends with an evidence-backed low-power score. Timing alone is not treated as power evidence.

On a cold start, ordinary routing explores available, capable backends with a positive policy score until each backend/task pair has five records. It selects the least-sampled eligible backend first, with deterministic score and registration-order tie-breaking. Passing `benchmark_backend="Backend Name"` remains an explicit way to seed a chosen backend up to the same five-record threshold.

After five records, recent performance contributes to routing. The score uses the median of the latest four end-to-end execution times, remains bounded at 25 points, distinguishes fast backends below 60 ms, and preserves reciprocal scoring at 60 ms and above. Combined policy and performance scores select the normal-route winner.

To keep a deterministic loser from retaining old evidence indefinitely, each policy/task pair gets one periodic refresh after ten successful normal scoring routes. The next ordinary route selects the available, compatible positive-base non-winner with the oldest matching benchmark evidence, with registration order breaking equal-age ties, and then resets the counter. The cadence repeats as ten normal routes followed by one refresh. Zero-base candidates, including LOW_POWER accelerators with no positive policy score, cannot be refreshed; explicit `benchmark_backend` behavior remains separate and unchanged. History and refresh counters are in memory per `AIRouter` instance.

## Installation

The packaged dependency set currently targets Python 3.11.

```bash
git clone https://github.com/TabiUK/ai-router.git
cd ai-router
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Install optional OpenVINO support with:

```bash
python -m pip install -e ".[openvino]"
```

CUDA requires a compatible NVIDIA driver and CUDA-enabled PyTorch/Torchvision wheels appropriate for the platform. MPS requires no additional Python dependency beyond the existing base PyTorch/Torchvision stack, but is available only when `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` both report true. See [BUILD.md](BUILD.md) before replacing the pinned base wheels. The pretrained ResNet18 weights are downloaded by Torchvision on first use if they are not already cached.

## Minimal Python example

```python
from core.policy import RoutingPolicy
from core.router import AIRouter
from core.task import Task
from core.task_types import TaskType

router = AIRouter(policy=RoutingPolicy.BALANCED)
result = router.route(
    Task(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        payload="image.png",
    )
)

print(result["routing"]["backend"])
print(result["result"]["predictions"])
```

## ComfyUI

The thin adapter in [`integrations/comfyui/`](integrations/comfyui/) exposes:

- AI Router Device Info
- AI Router Show Device Info
- AI Router Image Classification
- AI Router Show Classification

For a development checkout, link that directory into ComfyUI:

```text
ComfyUI/custom_nodes/ComfyUI-AIRouter
    -> /path/to/ai-router/integrations/comfyui
```

Restart ComfyUI and find the nodes under the `AI Router` category. Routing policy, benchmark history, and backend execution remain in AI Router rather than the ComfyUI adapter.

Automated router tests and the actual RunPod RTX 4090 workflow validated cold-start, historical scoring, and periodic refresh. After initial exploration, BALANCED preferred Torchvision CPU; the UI then observed ten normal CPU routes, one PyTorch CUDA evidence refresh, another ten normal CPU routes, and a second CUDA refresh. Refresh did not force CUDA to win: CUDA inference was about 1.2 ms but its refresh total was about 16.8-17.0 ms, while later CPU totals were about 11.5-13 ms. Because routing scores total execution time, BALANCED correctly continued to prefer CPU for this small workload.

## Validated platforms

Current real-hardware validation includes:

- Intel x86_64 macOS: Torchvision CPU and OpenVINO CPU.
- Apple M1 Pro arm64 macOS 15.7.7: PyTorch MPS ResNet18 inference and router participation on Python 3.11.9, PyTorch 2.2.2, and Torchvision 0.17.2.
- 64-bit Windows: OpenVINO on Intel Iris Xe Graphics and PyTorch CUDA on an NVIDIA RTX A1000 Laptop GPU.
- 64-bit Linux on RunPod with an NVIDIA GeForce RTX 4090: AI Router source,
  the CUDA backend, routing tests, and ComfyUI integration were validated on
  Python 3.12.3 with Torch 2.10.0+cu128 and Torchvision 0.25.0+cu128. This
  externally provisioned environment is not represented by the dependency
  versions pinned in `pyproject.toml`.

On the validated M1 Pro, direct backend comparison showed representative warm inference around 8.5 ms on MPS versus about 13 ms on Torchvision CPU, with representative warm total execution around 35 ms on MPS versus about 38-43 ms on CPU. MPS had a larger first-run cold-start cost. In a 15-route BALANCED routing validation, cold-start exploration collected five records for each backend; after evidence was available, Torchvision CPU remained the BALANCED winner because its three-point higher base score outweighed MPS's modest historical-performance advantage. These timing and routing results are hardware- and load-specific, not universal guarantees.

Availability on other hardware depends on the installed runtime, drivers, and framework wheels and should not be assumed from device family alone. See [REQUIREMENTS.md](REQUIREMENTS.md) for the detailed compatibility record.

## Project structure

```text
backends/             Backend implementations and registrations
core/                 Router, policies, tasks, registry, and benchmark scoring
examples/             Minimal discovery and routing examples
integrations/comfyui/ ComfyUI custom nodes
tests/                Standalone regression and hardware-evidence tests
```

## Documentation

- [Backend contributor guide](BACKEND_GUIDE.md)
- [Build and setup guide](BUILD.md)
- [Validated requirements](REQUIREMENTS.md)

## License

Copyright 2026 TabiUK. Licensed under the [Apache License 2.0](LICENSE).
