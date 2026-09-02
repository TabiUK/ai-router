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

Ending the workflow with Show Classification fixes the previous `Prompt has no outputs` error encountered when the classifier was used without an output node.

## Installation

For development, link this directory into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/ComfyUI-AIRouter
    -> ai-router/integrations/comfyui
```

Restart ComfyUI. The nodes appear under the `AI Router` category.

## RunPod RTX 4090 validation

The following UI flows were validated on the RunPod NVIDIA GeForce RTX 4090 environment:

```text
AI Router Device Info
-> AI Router Show Device Info

Load Image
-> AI Router Image Classification
-> AI Router Show Classification
```

Discovery displayed the generic x86_64 CPU, available PyTorch CUDA on NVIDIA GeForce RTX 4090, and available Torchvision ResNet18 CPU. OpenVINO CPU and OpenVINO Intel GPU were unavailable in this environment. Classification predictions and both timing values displayed successfully.

BALANCED routing alternated CPU and CUDA while their positive-base backend/task histories were below five records. After both reached five records, normal historical scoring resumed and preferred Torchvision CPU.

The UI then validated periodic stale-evidence refresh:

```text
10 normal CPU routes
-> 1 PyTorch CUDA refresh
-> 10 normal CPU routes
-> 1 PyTorch CUDA refresh
```

At the first refresh, CPU still had the higher combined score (approximately `82.52` versus CUDA `78.95`), but CUDA was selected to update its older evidence. The next refresh occurred after exactly ten more normal CPU routes. CUDA inference was approximately 1.2 ms, while CUDA refresh totals were approximately 16.8-17.0 ms and later CPU totals were roughly 11.5-13 ms. Because routing uses total execution time, BALANCED correctly continued preferring CPU for this workload. Refresh provides current evidence to an eligible losing backend; it does not force CUDA or any device to win.

Refresh is deterministic, excludes base-score-zero candidates such as CUDA under LOW_POWER, and leaves explicit `benchmark_backend` behavior separate. Its counters are in memory per `AIRouter`, keyed by policy and task type. The behavior is covered by automated router tests and was validated through this RunPod RTX 4090 ComfyUI workflow; it should not be generalized to every platform.
