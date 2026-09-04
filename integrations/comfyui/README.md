# AI Router ComfyUI Integration

This thin integration exposes the normal AI Router APIs as ComfyUI custom nodes. Routing decisions, cold-start exploration, periodic evidence refresh, and benchmark history remain in `AIRouter`.

## Current nodes

- **AI Router Device Info** discovers visible backends and returns device information.
- **AI Router Show Device Info** accepts that string and acts as a ComfyUI output node.
- **AI Router Image Classification** accepts an image and returns `backend`, `predictions`, `inference_time_ms`, and `execution_time_ms`.
- **AI Router Show Classification** accepts and displays those four classifier outputs and acts as a ComfyUI output node.

Connect the classification workflow as follows:

```text
Load Image
-> AI Router Image Classification
-> AI Router Show Classification
```

Ending the workflow with Show Classification fixes the `Prompt has no outputs` error that occurs when the classifier is used without an output node.

## Installation

For development, link this directory into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/ComfyUI-AIRouter
    -> ai-router/integrations/comfyui
```

Restart ComfyUI. The nodes appear under the `AI Router` category.

## RunPod RTX 4090 validation

The ComfyUI integration has been validated on RunPod with an NVIDIA GeForce RTX 4090.

The following UI flows were validated:

```text
AI Router Device Info
-> AI Router Show Device Info

Load Image
-> AI Router Image Classification
-> AI Router Show Classification
```

Device discovery correctly reported:

- generic x86_64 CPU
- PyTorch CUDA on NVIDIA GeForce RTX 4090
- Torchvision ResNet18 CPU

OpenVINO CPU and OpenVINO Intel GPU were unavailable in that environment.

Image classification completed successfully and returned the selected backend, predictions, inference time, and total execution time.

BALANCED routing was validated with cold-start exploration and periodic evidence refresh.

During cold start, available compatible positive-base-score CPU and CUDA backend/task pairs were sampled until both had sufficient benchmark history. Normal historical scoring then resumed.

Periodic evidence refresh followed the expected cadence:

```text
10 normal CPU routes
-> 1 PyTorch CUDA refresh
-> 10 normal CPU routes
-> 1 PyTorch CUDA refresh
```

Periodic refresh updates stale benchmark evidence for an eligible non-winning backend. It does not force CUDA, or any other backend, to become the routing winner.

Refresh is deterministic and excludes zero-base-score candidates, such as CUDA under `LOW_POWER`. Explicit `benchmark_backend` behavior remains separate.

Refresh counters are held in memory per `AIRouter` instance and are keyed by policy and task type.

Cold-start exploration and periodic refresh behavior are also covered by automated router tests.