# AI Router

AI Router is an experimental modular framework for routing AI workloads between different compute devices.

The long-term goal is to allow a machine to intelligently use combinations of:

- CPU
- Intel NPU
- Intel integrated GPU
- NVIDIA CUDA GPU
- AMD GPU
- Apple Silicon / MPS
- Other future accelerators

Rather than sending every AI workload to the most powerful GPU, AI Router aims to select an appropriate device based on:

- Task type
- Device capabilities
- Performance
- Power usage
- Current load
- Historical benchmark data
- User-selected routing policy

For example, a future workstation might route:

```text
Speech recognition        → NPU
Image classification      → NPU
Object detection          → NPU / iGPU
Image generation          → NVIDIA GPU
Video generation          → NVIDIA GPU
General processing        → CPU
```

The project is designed to be modular so that additional hardware backends can eventually be contributed independently.

---

# Current Status

AI Router is an early experimental prototype with a working modular routing core and one production workload: pretrained ResNet18 image classification. It is not production ready and does not yet provide persistent history, live hardware-load monitoring, remote execution, or broad model/task coverage.

Implemented routing targets:

- Generic CPU backend for the `general` and `classification` examples
- Torchvision ResNet18 CPU image classification
- OpenVINO CPU ResNet18 image classification
- OpenVINO Intel GPU ResNet18 image classification with dynamic Intel GPU discovery
- PyTorch CUDA ResNet18 image classification on CUDA device 0
- PyTorch MPS ResNet18 image classification when PyTorch reports MPS as built and available

The Mock Accelerator is test-only and does not self-register. OpenVINO NPU is diagnostic-only, not a production routing target. Core ML, AMD GPU, LLM, image-generation, and distributed/remote backends remain future work.

Implemented routing and measurement behavior:

- Automatic backend registration, discovery, availability checks, and capability filtering
- `PERFORMANCE`, `BALANCED`, and `LOW_POWER` policies
- Per-`AIRouter` in-memory benchmark history scoped by backend and task type
- Total execution and backend-reported inference timing
- Historical scoring after five records using the median of the latest four total times
- Continuous piecewise performance bonus bounded at 25 points
- Automatic cold-start exploration of available, compatible positive-score backends until each backend/task pair has five records
- Deterministic stale-evidence refresh after each ten successful normal combined-score routes for a policy/task pair
- Explicit `benchmark_backend` seeding while the requested backend/task pair has fewer than five records
- Exclusion of zero-score backends from automatic exploration, preserving LOW_POWER behavior
- Mock Accelerator available only through explicit test injection

ComfyUI integration is implemented under `integrations/comfyui/` and uses the same `AIRouter` instance, routing, exploration, and history behavior as the Python API. It provides Device Info, Show Device Info, Image Classification, and Show Classification nodes. Externally provisioned RunPod RTX 4090 validation of the current public main branch is complete for node import and startup plus the complete device-info and classification UI flows.

## Validated environments

The packaged dependency set remains officially targeted at Python 3.11:

```text
Python       3.11.x
NumPy        1.26.4
PyTorch      2.2.2
Torchvision  0.17.2
OpenVINO     2025.4.1 (optional)
```

Real-hardware validation includes:

- Intel x86_64 macOS: Torchvision CPU and OpenVINO CPU
- Apple M1 Pro arm64 macOS 15.7.7: PyTorch MPS ResNet18 inference and router participation on Python 3.11.9, PyTorch 2.2.2, and Torchvision 0.17.2
- Windows: Intel Iris Xe OpenVINO GPU and NVIDIA RTX A1000 Laptop GPU CUDA

Current-public-main validation was completed on externally provisioned RunPod Linux with an NVIDIA GeForce RTX 4090, Python 3.12.3, PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128, and CUDA build 12.8. This remains external validation evidence: it does not change the packaged `requires-python = ">=3.11,<3.12"` support requirement or the dependency versions in `pyproject.toml`.

---

# Project Structure

```text
ai-router/
├── backends/
│   ├── __init__.py
│   ├── cpu.py
│   ├── cuda.py
│   ├── mps.py
│   ├── mock_accelerator.py
│   ├── openvino.py
│   └── torchvision_classifier.py
│
├── core/
│   ├── __init__.py
│   ├── backend.py
│   ├── backend_registry.py
│   ├── benchmark.py
│   ├── device_types.py
│   ├── policy.py
│   ├── registry.py
│   ├── router.py
│   ├── runtime_types.py
│   ├── task.py
│   └── task_types.py
│
├── integrations/
│   ├── __init__.py
│   └── comfyui/
│       ├── __init__.py
│       ├── ai_router_nodes.py
│       └── README.md
│
├── tasks/
│   └── __init__.py
│
├── examples/
│   ├── detect_devices.py
│   ├── run_task.py
│   └── run_task_png.py
│
├── tests/
│   └── standalone regression and hardware-evidence tests
│
├── .gitignore
├── BACKEND_GUIDE.md
├── BUILD.md
├── LICENSE
├── pyproject.toml
├── README.md
└── REQUIREMENTS.md
```

---

# Architecture

The basic routing flow is:

```text
              Task
                │
                ▼
            AI Router
                │
                ▼
       Discover Backends
                │
                ▼
       Check Availability
                │
                ▼
       Check Capabilities
                │
                ▼
         Calculate Scores
                │
                ▼
       Select Best Backend
                │
                ▼
            Run Task
                │
                ▼
       Record Benchmark
                │
                ▼
             Result
```

Applications do not need to know which hardware backend will execute a task.

For example:

```python
task = Task(
    task_type=TaskType.IMAGE_CLASSIFICATION,
    payload="test.png",
)

result = router.route(task)
```

The router decides which available backend should execute it.

---

# Backend System

All hardware implementations inherit from the common `Backend` interface.

Contributors adding or changing a backend should follow the normative
authoring contract in:

```text
BACKEND_GUIDE.md
```

A backend provides:

```text
detect()
capabilities()
score()
run()
```

Conceptually:

```text
Backend
   │
   ├── CPUBackend
   │
   ├── TorchvisionClassifierBackend
   │
   ├── OpenVINOBackend
   │
   ├── OpenVINOIntelGPUBackend
   │
   ├── PyTorchCUDABackend
   │
   ├── PyTorchMPSBackend
   │
   ├── IntelNPUBackend          (future)
   │
   ├── AMDBackend               (future)
   │
   └── CoreMLBackend            (future)
```

Backends register themselves with AI Router.

The backend registry automatically discovers modules inside:

```text
backends/
```

This means the main application does not need hard-coded knowledge of every supported accelerator.

---

# Task Types

Current task types include:

```python
GENERAL

CLASSIFICATION
EMBEDDINGS

IMAGE_CLASSIFICATION
OBJECT_DETECTION
FACE_DETECTION
IMAGE_GENERATION
IMAGE_UPSCALING

SPEECH_TO_TEXT
TEXT_TO_SPEECH

VIDEO_GENERATION
VIDEO_UPSCALING
VIDEO_FRAME_ANALYSIS
```

Task types are intentionally specific.

For example:

```text
CLASSIFICATION
```

is different from:

```text
IMAGE_CLASSIFICATION
```

This prevents a backend expecting an image filename from accidentally receiving a generic classification payload.

---

# Routing Policies

AI Router currently supports three policies:

```text
PERFORMANCE
BALANCED
LOW_POWER
```

Example:

```python
router = AIRouter(
    policy=RoutingPolicy.BALANCED
)
```

Backends can return different scores depending on the selected policy.

For example, the explicitly injected synthetic routing demonstration uses:

```text
Classification task

PERFORMANCE
    Mock Accelerator → 100
    CPU              → 10

BALANCED
    Mock Accelerator → 80
    CPU              → 20

LOW_POWER
    CPU              → 50
    Mock Accelerator → 30
```

This results in:

```text
performance → Mock Accelerator
balanced    → Mock Accelerator
low_power   → CPU
```

This proves that the same workload can be routed to different hardware depending on policy. Mock Accelerator is not automatically registered and does not participate in normal production routing.

---

# Example 1 - Routing Test

Run:

```bash
python examples/run_task.py
```

This example uses:

```text
TaskType.CLASSIFICATION
```

with the payload:

```text
picture of a cat
```

It explicitly injects a Mock Accelerator and tests routing decisions between it and the CPU backend.

Expected behaviour:

```text
performance → Mock Accelerator
balanced    → Mock Accelerator
low_power   → CPU
```

The mock backend exists purely for testing routing architecture.

---

# Example 2 - Real AI Image Classification

The project now contains a real AI backend using:

```text
PyTorch
Torchvision
ResNet18
ImageNet pretrained weights
```

Run:

```bash
python examples/run_task_png.py
```

The example processes:

```text
test.png
```

using:

```text
TaskType.IMAGE_CLASSIFICATION
```

The router currently selects:

```text
Torchvision ResNet18 CPU
```

The image is genuinely passed through the pretrained ResNet18 neural network.

Example prediction output:

```text
maze           4.45%
accordion      3.96%
digital clock  3.95%
tennis ball    3.81%
mailbox        3.64%
```

These particular results have low confidence, but they demonstrate real model inference rather than simulated output.

---

# Benchmarking

AI Router now records execution performance automatically.

Two timing measurements are currently used.

## Total Execution Time

Measured by `AIRouter`.

This includes:

```text
backend execution
model loading when required
image loading
preprocessing
inference
post-processing
result construction
```

## Inference Time

Measured inside the AI backend.

This attempts to measure only the neural-network inference operation.

This distinction is important because accelerator performance alone does not determine real-world task latency.

For example, an NPU could perform inference very quickly but have significant model setup or data-transfer overhead.

---

# Current Real Benchmark

Five consecutive ResNet18 CPU classifications were performed using the same router/backend instance. The latest representative run before the dynamic-routing experiment produced:

Results:

```text
Run 1 total: 238.88 ms
Run 2 total:  60.08 ms
Run 3 total:  57.16 ms
Run 4 total:  59.89 ms
Run 5 total:  56.15 ms
```

Inference times:

```text
Run 1: 29.41 ms
Run 2: 26.35 ms
Run 3: 25.25 ms
Run 4: 27.37 ms
Run 5: 24.24 ms
```

Summary:

```text
Runs:               5
Cold start:       238.88 ms
Warm average:      58.32 ms
Average total:     94.43 ms
Average inference: 26.52 ms
Performance score: 17.15
```

This demonstrates an important routing consideration:

```text
Cold-start performance != warm performance
```

The first execution includes significant initialization overhead. Once the model is loaded, total execution is roughly 56-60 ms while neural-network inference itself is roughly 24-29 ms on the current Mac test machine.

Future routing decisions should therefore consider both cold-start cost and warm performance.

---

# Python Environment

- See `BUILD.md` for installation and environment setup.
- See `REQUIREMENTS.md` for supported versions, optional runtimes, hardware,
  and compatibility requirements.

---

# Current Benchmark Architecture

Benchmark records contain:

```text
backend
task_type
total_time_ms
inference_time_ms
```

The benchmark system currently calculates:

```text
average total execution time
average inference time
cold-start time
warm average time
historical performance score
```

Benchmark records can now be filtered by both backend and task type. This prevents measurements from different hardware or different workloads being averaged together.

Historical performance requires at least five benchmark records matching the same backend and task type. Once that threshold is reached, scoring uses only the latest four matching records.

The historical performance bonus uses the median of the latest four backend/task-specific records:

```text
warm_time = median(latest_4_total_time_ms)

if warm_time >= 60:
    historical_performance_bonus = 1000 / warm_time
else:
    historical_performance_bonus = 3000 / (warm_time + 120)
```

Higher values therefore represent faster recent execution. The branches meet at 60 ms, preserving the previous reciprocal score at and above that point while distinguishing faster backends without a hard plateau. At least five matching records are still required before scoring begins. The median prevents an isolated timing outlier from dominating routing, while the fast branch remains bounded by 25.0 even for zero or extremely small timings. Older measurements remain stored for reporting but no longer dominate routing decisions.

The router currently calculates:

```text
combined_score = base_policy_score + historical_performance_score
```

Combined scores now control candidate selection and actual routing. The five-record minimum and 25.0-point historical bonus cap remain in force.

Automatic cold-start exploration resolves the initial evidence problem. During ordinary routing, available and compatible backends with positive base scores and fewer than five matching records are explored by lowest record count first. Combined score and registration order provide deterministic tie-breaking. Once no eligible under-sampled backend remains, normal combined-score routing resumes.

Periodic stale-evidence refresh addresses the later winner-lock problem: without it, the deterministic combined-score winner would receive every subsequent timing record while a losing backend could retain the same old latest-four window indefinitely. For each `(policy, task_type)` pair, the router counts successful ordinary combined-score routes. After ten such routes, the next ordinary route refreshes one non-winning candidate and resets the counter, producing this steady-state cadence when an alternative exists:

```text
10 normal scoring routes
-> 1 stale-evidence refresh
-> 10 normal scoring routes
-> 1 stale-evidence refresh
```

Refresh selection is deterministic and backend-agnostic. Candidates must be currently available and compatible, have a base score greater than zero under the active policy, and not be the current normal combined-score winner. The candidate whose latest matching backend/task record is oldest in the append-only history is selected; registration order breaks equal-age ties. Initial exploration takes precedence over periodic refresh. A successful cold-start route or refresh resets the counter, while a successful normal route increments it only when a positive-base alternative exists.

Explicit `benchmark_backend` routing bypasses refresh bookkeeping and retains its existing five-record limit. Zero-base candidates cannot enter initial exploration or periodic refresh, so LOW_POWER policy restrictions remain intact. Benchmark history and refresh counters exist only for the lifetime of each `AIRouter` instance; neither is persisted.

---

# Dynamic Routing Experiment

At that historical milestone, a controlled test used the then-current scoring formula with two competing image-classification backends:

```text
Mock Accelerator
Base BALANCED score: 50
Synthetic benchmark: 100 ms cold / 40 ms warm
Historical performance score: 25
Combined score: 75

Torchvision ResNet18 CPU
Base BALANCED score: 60
Historical performance score after warm measurements: approximately 16-17
Combined score: approximately 76-77
```

This successfully proved that historical performance can be scoped per backend/task and combined with the policy score.

A temporary experiment then changed actual routing to choose by `combined_score`. This exposed an important feedback-loop flaw. The fake Mock Accelerator executed in effectively 0 ms, so each selected run was added to its benchmark history. Its calculated performance score then grew rapidly while the ResNet backend was never selected and therefore never collected new benchmark data.

Observed behaviour included:

```text
Mock performance score: 25 -> ~50 -> ~75 -> ~100 -> ~125
ResNet performance score: None
```

This is a winner-takes-all feedback loop and is not acceptable for real routing.

The experiment was therefore rolled back. Steps 25 and 26 were introduced as protections against this failure mode:

- Step 25 requires at least five backend/task-specific benchmark records before historical performance can contribute.
- Step 26 protects zero-time measurements and bounds the historical performance bonus to a maximum of 25.0.

Both protections have been implemented and tested. Step 27 subsequently re-enabled combined-score routing and confirmed that the historical bonus remains bounded during repeated near-zero runs.

---

# Completed Dynamic-Scoring Protections

## Step 25 - Minimum Benchmark History (COMPLETED)

Historical performance now requires at least five records matching the same backend and task type. Until that threshold is reached, `performance_score()` returns `None` and the combined score remains equal to the base policy score.

## Step 26 - Bounded Historical Performance Bonus (COMPLETED)

Historical performance remains bounded at 25.0. The current fast branch adds 120.0 ms to its denominator, so zero or extremely small measurements cannot cause division errors or exploding combined scores.

Step 26 testing confirmed that a backend with zero-millisecond synthetic history receives a maximum historical performance bonus of 25.0. At that stage, actual routing still selected by base policy score even when another backend had a higher displayed combined score.

## Step 27 - Combined-Score Routing (COMPLETED)

Combined scores now control candidate sorting and actual routing. Controlled testing confirmed that insufficient history contributes no bonus, sufficient history can change the selected backend, and repeated near-zero runs cannot increase the historical bonus above 25.0.

This historical limitation was later resolved by automatic cold-start exploration, described below.

---

## Step 28 - Controlled Benchmark Seeding (COMPLETED)

Callers can pass an optional `benchmark_backend` name to `AIRouter.route()` to seed benchmark history for a specific backend/task pair. This explicit path remains deterministic and unchanged. Automatic cold-start exploration was added later for ordinary routes; it does not run background workloads.

The requested backend must exist, be available, and support the task. Explicit seeding is allowed only while that backend/task pair has fewer than five records. Once five matching records exist, further explicit seeding attempts fail before backend execution with a sufficient-history error.

The five-record limit applies only to explicit seeding. Normal combined-score routing continues to execute the selected backend and may continue adding benchmark records after seeding is complete.

---

## Step 29 - Recent-Performance Scoring Window (COMPLETED)

Historical performance still requires at least five matching backend/task records, but the score now uses only the latest four matching records. Older benchmark measurements remain stored and available for reporting while recent performance can change the combined score and routing decision. The historical bonus remains capped at 25.0.

Controlled testing confirmed that four records still produce no score, five records activate scoring, older slow or fast measurements fall out of the routing window, backend/task filtering remains intact, and recent measurements can change the backend selected by combined-score routing.

---

## Step 30 - Outlier-Resistant Historical Scoring (COMPLETED)

Historical scoring now uses the median of the latest four matching backend/task records. The five-record minimum and 25.0-point historical bonus bound remain in force.

Controlled testing confirmed that isolated slow or fast timing outliers no longer dominate routing decisions, while four consistently changed recent measurements still update the combined score and selected backend.

---

## Cold-Start Exploration, Periodic Refresh, and Continuous Fast Scoring (COMPLETED)

Ordinary routing now samples every available, compatible backend with a positive policy score until that backend/task pair reaches five records. The least-sampled eligible pair is selected first; combined score and backend registration order break ties deterministically. LOW_POWER backends with a zero base score are not explored.

Explicit `benchmark_backend` selection continues to target only the named backend and is accepted only while that backend/task pair has fewer than five records.

After exploration, normal combined-score routing resumes. After ten successful normal scoring routes for a policy/task pair, the next ordinary route refreshes the eligible non-winner with the oldest matching evidence, then resets the counter. This refresh exists to update stale evidence, not to favor an accelerator or force a different winner. Historical scoring still uses the latest-four median and the five-record minimum. The current continuous piecewise bonus is:

```text
if warm_time >= 60:
    bonus = 1000 / warm_time
else:
    bonus = 3000 / (warm_time + 120)
```

The branches meet at 60 ms, remain bounded at 25 points, and allow meaningful differences between fast backends to affect routing. Cold-start and periodic refresh behavior are covered by deterministic automated router tests and were also exercised as earlier externally provisioned evidence through the ComfyUI workflow on the RunPod RTX 4090. Current-public-main RunPod validation is complete for the CUDA, explicit benchmark, history-guard, and ComfyUI paths listed in the Integrations section.

---

# Intel OpenVINO CPU ResNet18 Inference (COMPLETED)

The OpenVINO backend now supports real ResNet18 image classification on the CPU in:

```text
backends/openvino.py
```

`OpenVINOBackend` accepts explicit `target_device` and `warmup_runs`
configuration; its generic warm-up default is zero. The
registered production backend continues to use the default `"CPU"` target and
preserves the existing `"OpenVINO"` name used by routing and benchmark history.
Detection, compilation, backend details, and result identity all use the
configured target consistently. A separate registered `OpenVINO Intel GPU`
backend dynamically discovers an Intel `GPU*` device through
`FULL_DEVICE_NAME`. NPU and other manual target configurations remain
diagnostic-only.

OpenVINO remains an optional dependency. OpenVINO 2025.4.1 converts the same pretrained torchvision ResNet18 model used by the existing Torchvision backend, compiles it specifically for `CPU`, and reuses the same preprocessing and ImageNet categories.

```text
detect()        → report the OpenVINO version and available_devices
capabilities()  → return [image_classification]
score()         → return policy-specific OpenVINO CPU base score
run()           → perform OpenVINO ResNet18 inference on CPU
```

The OpenVINO CPU base scores are:

```text
PERFORMANCE → 38
BALANCED    → 58
LOW_POWER   → 63
```

Explicit benchmark seeding with `benchmark_backend="OpenVINO"` successfully executes the backend and records benchmark history through the existing router path.

OpenVINO 2025.4.1 is installed in the existing `.venv`. The runtime version was detected successfully as:

```text
2025.4.1-20426-82bbf0292c5-releases/2025/4
```

Direct runtime detection returned:

```text
Core().available_devices == ['CPU']
```

Running `examples/detect_devices.py` reported the same `['CPU']` device list and marked the OpenVINO backend as available. Real inference testing confirmed that its top-five prediction categories match the Torchvision ResNet18 CPU backend and confidence values remain within 0.1 percentage points.

The first run includes a significant cold-start cost from model loading, conversion, and CPU compilation. The following numbers are observed evidence from one interleaved run on the Intel Mac using one router instance, the same `test.png`, and five explicitly seeded runs for each CPU backend.

```text
OpenVINO CPU
Cold total:               1950.87 ms
Median warm total:          47.46 ms
Warm inference: approximately 14-15 ms
Historical bonus:           21.07

Torchvision ResNet18 CPU
Cold total:                245.26 ms
Median warm total:          56.72 ms
Warm inference: approximately 23-25 ms
Historical bonus:           17.63
```

In that observed Mac run, OpenVINO had a substantially larger cold-start cost but faster warm total and inference times. The two backends produced matching top-five categories with confidence differences within 0.1 percentage points. These values do not guarantee the winner of a later route.

In the recorded policy-routing regression on the reference Mac, five
interleaved real CPU benchmark records were collected for both OpenVINO and
Torchvision. With that run's populated history, `PERFORMANCE` and `BALANCED`
selected OpenVINO, while `LOW_POWER` selected Torchvision. A normal `BALANCED`
route using that same history selected OpenVINO; later outcomes may differ with
recent history and system load.

The latest measured `BALANCED` combined scores were:

```text
OpenVINO:                  75.02
Torchvision ResNet18 CPU:  71.06
```

Prediction parity, CPU-only OpenVINO compilation, missing-package safety, and unsupported-task handling all passed in the full regression.

The reusable regression test is:

```text
tests/test_openvino_cpu.py
```

It validates runtime detection, optional-package safety, fair interleaved benchmark seeding for OpenVINO and Torchvision, benchmark timing and scoring, prediction parity, CPU-only compilation, unsupported-task handling, and normal routing with populated history.

---

# Current OpenVINO Production State

OpenVINO CPU support is implemented, tested, and routable for `TaskType.IMAGE_CLASSIFICATION`. Its policy scores are:

```text
PERFORMANCE = 38
BALANCED    = 58
LOW_POWER   = 63
```

Observed interleaved benchmarking on the reference Mac showed that OpenVINO CPU can outperform the Torchvision CPU backend after warm-up. Current routing can select a different eligible backend as recent history and system load change.

The reusable OpenVINO CPU regression test is:

```text
tests/test_openvino_cpu.py
```

The Intel GPU diagnostic smoke test is:

```text
tests/test_openvino_gpu.py
```

The diagnostic now discovers every OpenVINO device whose ID starts with
`"GPU"`, queries `FULL_DEVICE_NAME`, and selects an Intel GPU dynamically. It
does not assume that the Intel GPU is always `GPU.0`, and it cleanly skips when
no Intel GPU is exposed. The discovered device ID is used consistently for the
`OpenVINOBackend` target, availability checks, result identity, and compiled
execution-device assertion.

Observed evidence from a real OpenVINO ResNet18 run on a Windows laptop used
the following selected device:

```text
Device ID:        GPU.0
Full device name: Intel(R) Iris(R) Xe Graphics (iGPU)
Result identity:  openvino_resnet18_gpu.0
Inference time:   48.95 ms
EXECUTION_DEVICES: ['GPU.0']
```

The run produced exactly five predictions, matched the Torchvision CPU top-1
category and top-5 category set, and kept all confidence differences within
0.1 percentage points. The diagnostic calls the existing backend's actual
`run()` method and does not duplicate model conversion, preprocessing, or
inference logic.

The production Intel GPU backend is now automatically registered when dynamic
discovery finds an Intel GPU. Its stable routing and benchmark identity is:

```text
OpenVINO Intel GPU
```

Its base scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The generic `OpenVINOBackend` defaults to `warmup_runs=0`. The registered Intel
GPU backend uses `warmup_runs=2`, once per backend instance. Both warm-ups occur
inside the first routed execution, so initialization and stabilization remain
visible in that first router total. The first result reports the actual warm-up
count and total warm-up time; later results report `warmup_runs = 0` and
`warmup_time_ms = 0.0`.

The LOW_POWER score reflects the absence of power evidence and strongly avoids
preferring the GPU; it does not disable execution. PERFORMANCE and BALANCED
routing combine these base scores with existing backend/task history. CPU or
Intel GPU may legitimately be selected depending on recent measurements, other
eligible candidates, and current system load. The observed Iris Xe results do
not establish that Intel GPU always wins either policy.

Production routing and opt-in Mock regressions are:

```text
tests/test_openvino_gpu_routing.py
tests/test_mock_backend_opt_in.py
```

Mock Accelerator no longer self-registers. Normal `AIRouter()` discovery uses
real production backends only; tests and demonstrations that need Mock inject
it explicitly.

The Intel NPU diagnostic smoke test is:

```text
tests/test_openvino_npu.py
```

It imports OpenVINO optionally, uses
`OpenVINOBackend(target_device="NPU")` to verify the device-targeting contract,
queries `Core().available_devices`, and cleanly
skips if OpenVINO is missing or `"NPU"` is unavailable. On compatible hardware,
it uses the same ResNet18 and `test.png` path, compiles explicitly for `"NPU"`,
reports inference timing, verifies five predictions, requires CPU parity for
the top-1 category and at least four overlapping top-5 categories, reports
confidence differences for inspection, and asserts:

```text
EXECUTION_DEVICES == ['NPU']
```

The diagnostic was run on the current Mac and passed with the expected clean
skip:

```text
Core().available_devices: ['CPU']
SKIPPED: OpenVINO NPU is not available on this machine.
```

This validates the NPU-unavailable clean-skip behavior on this machine. Actual
NPU compilation and inference remain unvalidated. This NPU path is diagnostic
only, and no routable Intel NPU backend should be added without real hardware
validation.

## Current NVIDIA CUDA Production State

The `PyTorch CUDA` backend is implemented, tested, and routable for
`TaskType.IMAGE_CLASSIFICATION`. It uses the same pretrained Torchvision
ResNet18 weights and preprocessing as the Torchvision CPU backend.

The registered production backend dynamically checks PyTorch CUDA availability
and uses CUDA device 0. Its stable routing and benchmark-history identity is
`PyTorch CUDA`; its stable result identity is `pytorch_cuda_resnet18`. Neither
identity contains a device index or GPU model. The actual device name, device
index, compute capability, VRAM, PyTorch version, and compiled CUDA version are
reported separately as runtime metadata.

Current image-classification scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The production instance performs two warm-up inferences on first use only.
Their elapsed time is reported and remains inside the router's first total
execution time. Later calls report zero warm-up runs and zero warm-up time.

CUDA benchmark records participate in the existing backend/task-scoped history
model. On the validated Windows run, representative history caused CUDA to win
PERFORMANCE and BALANCED, while LOW_POWER remained CPU-oriented. This is
machine-specific evidence, not a universal routing result: hardware, drivers,
system load, desktop GPU use, scheduling, and other processes can change both
timings and winners. LOW_POWER remains zero because no NVIDIA power-efficiency
measurements have been collected.

CUDA requires a CUDA-enabled PyTorch build, a compatible NVIDIA GPU and driver,
and `torch.cuda.is_available()` with at least one reported device. A separate
CUDA Toolkit installation is not required by the validated wheel-based setup.
When CUDA is unavailable, the backend reports unavailable without constructing
the model or initializing CUDA, and normal CPU/OpenVINO routing continues.

Current real-hardware CUDA validation includes an NVIDIA RTX A1000 Laptop GPU on Windows. Externally provisioned validation of the current public main branch is also complete on an NVIDIA GeForce RTX 4090 on RunPod Linux. Reusable code does not assume either model name.

## Current PyTorch MPS Production State

The `PyTorch MPS` backend is implemented, tested, and routable for
`TaskType.IMAGE_CLASSIFICATION`. It uses Torchvision pretrained ResNet18 with
the standard ImageNet preprocessing and categories. Its stable routing and
benchmark-history identity is `PyTorch MPS`; its stable result identity is
`pytorch_mps_resnet18`.

The backend is automatically available when PyTorch reports both:

```text
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
```

Current image-classification scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The generic backend defaults to zero warm-up runs. The registered production
backend performs two warm-up inferences once per backend instance. Warm-up and
model initialization remain in the first routed total execution time; later
calls report zero warm-up runs and zero warm-up time.

MPS requires no additional Python dependency beyond the existing base
PyTorch/Torchvision stack. Real-hardware validation was performed on Apple M1
Pro, arm64, macOS 15.7.7, Python 3.11.9, PyTorch 2.2.2, and Torchvision
0.17.2. In that environment, real MPS ResNet18 inference and router
participation were validated.

In the direct backend comparison on that M1 Pro, representative warm inference
was approximately 8.5 ms on MPS versus approximately 13 ms on Torchvision CPU.
Representative warm total execution was approximately 35 ms on MPS versus
approximately 38-43 ms on CPU. MPS had a larger first-run cold-start cost.

In the 15-route BALANCED routing validation, cold-start exploration collected
five records for each backend. After evidence was available, Torchvision CPU
remained the BALANCED winner because its three-point higher base score
outweighed MPS's modest historical-performance advantage. MPS should not be
described as universally faster or as the guaranteed routing winner. LOW_POWER
is zero because no power-efficiency advantage has been measured. MPS timing and
routing results are hardware- and load-specific.

## Current Next Priorities

- Continue cross-hardware evidence collection before treating accelerator scores as broadly optimal.
- Validate real NPU hardware before considering an OpenVINO NPU production backend.
- Design persistent benchmark storage, hardware/load monitoring, and additional task backends separately.

The longer-term scoring intention remains:

```text
Backend capability
        +
Routing policy
        +
Bounded historical performance
        +
Current device load
        +
Power characteristics
        =
Final routing decision
```

---

# Backend Roadmap

Implemented production routing targets are Torchvision CPU, OpenVINO CPU, dynamically discovered OpenVINO Intel GPU, PyTorch CUDA device 0, and PyTorch MPS for ResNet18 image classification, plus the generic CPU example backend.

Future backend work may include:

```text
OpenVINO NPU (after real-hardware validation)
AMD GPU / ROCm
Core ML
Qualcomm NPU
Additional models and task types
Distributed or remote execution
```

Each backend should implement the same common interface rather than requiring changes to applications using AI Router.

---

# Long-Term Goal

A future machine might contain:

```text
CPU
Intel NPU
Intel integrated GPU
NVIDIA RTX GPU
```

Instead of sending everything to the RTX GPU, AI Router could potentially do:

```text
Prompt analysis
        ↓
NPU

Object / face detection
        ↓
NPU or iGPU

Image generation
        ↓
RTX GPU

Image classification
        ↓
NPU

Video frame analysis
        ↓
NPU / iGPU

Video generation
        ↓
RTX GPU

File handling / orchestration
        ↓
CPU
```

Potentially several devices could perform independent AI workloads simultaneously.

The goal is not necessarily to split one neural network across every available processor.

Instead, AI Router aims to intelligently distribute different AI tasks to the hardware best suited to each workload.

---

# Integrations

The ComfyUI integration is implemented in `integrations/comfyui/` as a thin adapter over the normal `AIRouter` API. It provides:

- AI Router Device Info
- AI Router Show Device Info
- AI Router Image Classification
- AI Router Show Classification

The image-classification node keeps one router instance, accepts a ComfyUI image, and returns the selected backend, predictions, inference time, and total execution time. Show Classification accepts those four outputs and displays them as a real ComfyUI output node. This resolves the `Prompt has no outputs` error produced when a workflow ended at the non-output classifier node. Routing behavior remains in the core router rather than being duplicated in the integration.

Validation of the current public main branch completed successfully on this externally provisioned environment:

```text
Platform:     RunPod Linux
GPU:          NVIDIA GeForce RTX 4090
Python:       3.12.3
PyTorch:      2.10.0+cu128
Torchvision:  0.25.0+cu128
CUDA build:   12.8
```

The completed validation covered:

- AI Router device detection
- CUDA runtime/API metadata
- `tests/test_cuda.py`
- `tests/test_cuda_resnet18.py`
- `tests/test_cuda_backend.py`
- ComfyUI AI Router node import and startup
- AI Router Device Info
- AI Router Show Device Info
- AI Router Image Classification
- AI Router Show Classification
- Explicit PyTorch CUDA `benchmark_backend` path
- Benchmark-history guard

The validated ComfyUI flows were:

```text
AI Router Device Info
-> AI Router Show Device Info

Load Image
-> AI Router Image Classification
-> AI Router Show Classification
```

Device discovery displayed the generic x86_64 CPU, available PyTorch CUDA on NVIDIA GeForce RTX 4090, and available Torchvision ResNet18 CPU. OpenVINO CPU and OpenVINO Intel GPU were unavailable in that RunPod environment. Classification predictions and both inference and execution timing values were displayed successfully.

Python 3.12 and PyTorch 2.10 in this RunPod environment remain externally provisioned validation evidence. They do not change the packaged Python requirement of `>=3.11,<3.12` or the package dependency versions.

During the earlier externally provisioned RunPod validation, the actual ComfyUI UI also validated BALANCED cold-start exploration. While CPU and CUDA were below five records, observed routing alternated CPU -> CUDA -> CPU -> CUDA. When CPU had five records (`base=60`, `performance=22.492847715384617`, `combined=82.49`) and CUDA was still under-sampled (`base=57`, `performance=None`, `combined=57`), routing selected CUDA to obtain its fifth record despite its lower displayed combined score. After both backends had five records, historical scoring resumed: CUDA scored `57 + 21.916057283692375 = 78.92`, Torchvision CPU scored `60 + 22.492847715384617 = 82.49`, and BALANCED selected Torchvision CPU.

Periodic refresh was then validated through the same UI during that earlier RunPod validation. The observed steady-state sequence was ten normal Torchvision CPU selections, one PyTorch CUDA refresh, another ten normal CPU selections, and a second CUDA refresh. On the first refresh, CUDA was approximately `base=57`, `performance=21.95`, `combined=78.95`, while CPU was approximately `base=60`, `performance=22.52`, `combined=82.52`. CUDA was deliberately routed only to refresh its older evidence even though CPU remained the normal scoring winner.

The refresh measurements did not establish that CUDA should win this workload. CUDA inference was approximately 1.2 ms, but its total refresh execution was approximately 16.8-17.0 ms; later CPU total execution was roughly 11.5-13 ms. Historical routing uses total execution time, so BALANCED correctly continued selecting Torchvision CPU for this small ResNet18 workload. The mechanism is backend-agnostic and exists to obtain fresh evidence for losing candidates, not to prefer CUDA or any device class.

In that previous RunPod environment, the integration was a symlink from ComfyUI `custom_nodes` to the repository integration directory. A duplicate older plugin copy was removed. The unrelated `comfyui-impact-pack` startup failure was not an AI Router issue.

Potential future integrations include REST APIs, local model servers, desktop/web applications, and mobile applications. None is currently implemented.

---

# Development Principle

AI Router is being developed incrementally.

Each architectural feature is tested before adding the next one.

Current progression:

```text
[✓] Project structure
[✓] Backend interface
[✓] CPU backend
[✓] Automatic backend discovery
[✓] Backend registration
[✓] Task abstraction
[✓] Task types
[✓] Capability routing
[✓] Backend scoring
[✓] Competing backends
[✓] Routing policies
[✓] Real AI workload
[✓] Real image classification
[✓] Router execution timing
[✓] Pure inference timing
[✓] Benchmark records
[✓] Cold/warm benchmark statistics
[✓] Backend/task benchmark filtering
[✓] Historical performance scoring
[✓] Combined score calculation and inspection
[✓] Synthetic competing-backend score test
[✓] Dynamic-routing experiment performed
[✓] Historical dynamic-routing feedback-loop identified and temporarily reverted

[✓] Step 25 - minimum 5 backend/task-specific benchmark records before performance scoring
[✓] Step 26 - historical performance bonus bounded to 25.0 with zero/sub-1 ms protection
[✓] Step 27 - combined scores re-enabled for routing and validated with bounded history
[✓] Step 28 - explicit controlled benchmark seeding with a 5-record limit
[✓] Step 29 - score historical performance from the latest 4 matching records
[✓] Step 30 - median scoring protects against isolated timing outliers
[✓] Automatic cold-start exploration balances under-sampled positive-score backend/task pairs
[✓] Periodic stale-evidence refresh runs after ten normal policy/task routes
[✓] Oldest-evidence selection, registration-order ties, and policy/task counter isolation regression tested
[✓] LOW_POWER zero-score backends excluded from automatic exploration
[✓] Continuous piecewise performance scoring distinguishes fast backends while remaining bounded at 25
[✓] Intel OpenVINO detection-only backend implemented and tested
[✓] OpenVINO 2025.4.1 runtime validated with matching CPU device discovery
[✓] OpenVINO CPU ResNet18 inference implemented and regression tested
[✓] Evidence-based OpenVINO CPU policy routing validated against Torchvision
[✓] Intel GPU diagnostic smoke test created
[✓] Intel GPU diagnostic passed with the expected clean skip on the current Mac
[✓] Intel Iris Xe OpenVINO GPU ResNet18 inference validated on Windows
[✓] Intel GPU diagnostic dynamically selects an Intel GPU by FULL_DEVICE_NAME
[✓] Stabilized CPU-vs-Intel-GPU evidence collected on the Windows laptop
[✓] OpenVINO Intel GPU production routing implemented with scores 37 / 57 / 0
[✓] One-time Intel GPU first-use warm-up metadata and cost validated
[✓] Mock Accelerator removed from automatic production discovery
[✓] Intel NPU diagnostic smoke test created
[✓] Intel NPU diagnostic passed with the expected clean skip on the current Mac
[ ] Intel NPU backend
[✓] NVIDIA CUDA diagnostics and four-way ResNet18 benchmark validated on Windows
[✓] NVIDIA CUDA routing-evidence experiment completed
[✓] PyTorch CUDA production backend implemented with scores 37 / 57 / 0
[✓] CUDA first-use warm-up, history isolation, routing, and fallback regression validated
[✓] Current-public-main RunPod RTX 4090 device, CUDA metadata, CUDA test, explicit benchmark, and history-guard validation completed on Python 3.12.3 / PyTorch 2.10.0+cu128 / Torchvision 0.25.0+cu128 / CUDA 12.8 as externally provisioned evidence
[✓] PyTorch MPS production backend implemented with scores 37 / 57 / 0
[✓] MPS first-use warm-up, inference, registration, and routing participation validated on Apple M1 Pro
[✓] ComfyUI Device Info, Show Device Info, Image Classification, and Show Classification nodes implemented
[✓] Current-public-main ComfyUI node import/startup, device-info, and classification validation completed on RunPod RTX 4090
[✓] Earlier externally provisioned ComfyUI BALANCED cold-start evidence recorded on RunPod RTX 4090
[✓] Earlier externally provisioned ComfyUI stale-evidence refresh evidence recorded on RunPod RTX 4090
[✓] Duplicate older ComfyUI custom-node copy removed
[✓] Root public README and Apache-2.0 release metadata prepared
[ ] Persistent benchmark database
[ ] Hardware/load monitoring
[ ] REST API
[✓] ComfyUI integration
```

---

# Project Philosophy

AI Router should remain:

- Modular
- Hardware-independent at the core
- Extensible
- Measurable
- Policy-driven
- Suitable for community-contributed backends
- Useful across different combinations of AI hardware

The core router should not need to know whether a contributor is using Intel, NVIDIA, AMD, Apple, Qualcomm, or future accelerator hardware.

Hardware-specific knowledge belongs in the backend modules.

---

## Status

**Early experimental prototype — not production ready.**

The current implementation routes real ResNet18 image classification across available Torchvision CPU, OpenVINO CPU, dynamically discovered Intel GPU, PyTorch CUDA, and PyTorch MPS backends. It performs automatic cold-start exploration, maintains recent in-memory benchmark history, and is integrated with ComfyUI. Broader workloads, persistent history, hardware-load monitoring, and additional accelerator families remain future work.
