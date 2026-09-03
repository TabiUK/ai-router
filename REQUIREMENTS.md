# AI Router — Requirements and Compatibility

## Purpose

This document defines the currently validated Python, runtime, operating-system, hardware, and compatibility requirements for AI Router.

For complete clean-system installation instructions, read:

```text
BUILD.md
```

For the latest development status, read:

```text
README_AI.md
```

For normative contributor-facing backend authoring rules, read:

```text
BACKEND_GUIDE.md
```

Runtime, operating-system, hardware, and compatibility requirements remain in
this document.

For machine-readable Python dependencies, use:

```text
pyproject.toml
```

---

# 1. Current Validated Stack

The current validated reference stack is:

```text
Python        3.11.x
NumPy         1.26.4
PyTorch       2.2.2
Torchvision   0.17.2
Pillow        required
OpenVINO      2025.4.1
```

The original reference development environment used:

```text
Python 3.11.9
macOS
Intel x86_64
```

The current project supports:

```text
Torchvision CPU inference
OpenVINO CPU inference
OpenVINO CPU routing
Intel GPU diagnostic testing
Real Intel Iris Xe OpenVINO GPU inference on Windows
PyTorch CUDA ResNet18 inference and routing on Windows
PyTorch CUDA ResNet18 inference and routing on Linux/RunPod
PyTorch MPS ResNet18 inference and routing on Apple Silicon
Routable Intel OpenVINO GPU for image classification
ComfyUI device-info and image-classification integration
```

Routable Intel GPU, NVIDIA CUDA, and Apple MPS support are implemented.
Routable Intel NPU and Core ML support are not. The packaged dependency target
remains Python 3.11; the RunPod validation used an externally provisioned
Python 3.12.3 environment and does not widen the supported `requires-python`
range.

---

# 2. Python Version

## Validated version

Use:

```text
Python 3.11
```

The current `pyproject.toml` requires:

```toml
requires-python = ">=3.11,<3.12"
```

This is intentionally narrow because Python 3.11 is the version currently validated with the project’s PyTorch, Torchvision, NumPy, and OpenVINO stack.

Do not widen this range until additional Python versions have been tested successfully.

---

# 3. pyproject.toml Is the Dependency Source of Truth

The project now declares its Python dependencies in:

```text
pyproject.toml
```

The current base dependency set is:

```toml
dependencies = [
    "numpy==1.26.4",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "pillow",
]
```

OpenVINO is declared as an optional extra:

```toml
[project.optional-dependencies]

openvino = [
    "openvino==2025.4.1",
]
```

Therefore:

```bash
python -m pip install -e .
```

installs the base AI Router Python stack.

And:

```bash
python -m pip install -e ".[openvino]"
```

installs the base stack plus OpenVINO.

You should not normally install these packages individually on a clean machine unless troubleshooting.

---

# 4. Base Python Dependencies

The currently required base packages are:

```text
numpy==1.26.4
torch==2.2.2
torchvision==0.17.2
pillow
```

They are installed automatically when running:

```bash
python -m pip install -e .
```

---

# 5. Optional OpenVINO Dependency

OpenVINO is optional at the packaging level.

To enable the current OpenVINO CPU backend:

```bash
python -m pip install -e ".[openvino]"
```

This installs:

```text
openvino==2025.4.1
```

along with the base AI Router dependencies.

OpenVINO is optional so that:

```text
non-Intel systems
non-OpenVINO users
CUDA-only or MPS-only systems
```

are not forced to install an Intel-specific runtime.

The OpenVINO backend is designed to fail gracefully when OpenVINO is not installed.

---

# 6. Why NumPy Is Pinned to 1.26.4

The project previously encountered a compatibility issue between:

```text
PyTorch 2.2.2
NumPy 2.x
```

The warning included:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

and:

```text
Failed to initialize NumPy: _ARRAY_API not found
```

The validated fix was:

```text
NumPy 1.26.4
```

Therefore the current project explicitly pins:

```text
numpy==1.26.4
```

Do not remove or loosen this pin without retesting the entire stack.

---

# 7. PyTorch and Torchvision Compatibility

The currently validated pair is:

```text
PyTorch       2.2.2
Torchvision   0.17.2
```

These versions are declared in `pyproject.toml`.

Current real image-classification backends rely on Torchvision ResNet18, so both packages are part of the base runtime requirements. PyTorch MPS requires no additional Python dependency beyond this base PyTorch/Torchvision stack.

---

# 8. Pillow

Pillow is required for image loading and preprocessing.

It is declared as:

```text
pillow
```

without an exact version pin.

The currently installed development environment has used a working Pillow version, but the project does not yet require one exact release.

Before a formal public release, it may be useful to validate and pin a known-good Pillow version if reproducibility requires it.

---

# 9. OpenVINO Version

The current validated OpenVINO version is:

```text
OpenVINO 2025.4.1
```

The development Mac reported:

```text
2025.4.1-20426-82bbf0292c5-releases/2025/4
```

The current OpenVINO backend performs real CPU ResNet18 inference.

`OpenVINOBackend` accepts an explicit `target_device` and defaults to
`warmup_runs=0`. Automatic production registration preserves the CPU routing
and benchmark identity `"OpenVINO"` and also registers an Intel GPU backend
when OpenVINO exposes one. Its stable routing identity is
`"OpenVINO Intel GPU"`; its physical target is discovered dynamically by
querying `FULL_DEVICE_NAME` for advertised `GPU*` devices. The configured
target is used for availability detection, compilation, backend details, and
result identity. NPU and other manually targeted configurations remain
diagnostic-only.

The backend has been tested for:

```text
runtime detection
CPU device detection
model conversion
CPU compilation
real inference
prediction parity
benchmark collection
routing
```

---

# 10. Current OpenVINO Device State

On the current development Mac:

```python
Core().available_devices
```

returns:

```text
['CPU']
```

This means:

```text
OpenVINO CPU      available
OpenVINO GPU      not exposed
OpenVINO NPU      not exposed
```

Device visibility depends on:

```text
hardware
operating system
drivers
OpenVINO runtime/plugin support
```

Installing the Python package alone does not guarantee that Intel GPU or NPU devices will appear.

---

# 11. OpenVINO CPU Support

Current real OpenVINO implementation:

```text
TaskType.IMAGE_CLASSIFICATION
```

Current model:

```text
ResNet18
```

Current target:

```text
CPU
```

The model is converted at runtime using OpenVINO and compiled specifically for:

```text
"CPU"
```

The reusable regression test is:

```text
tests/test_openvino_cpu.py
```

---

# 12. Intel GPU Status

Current state:

```text
Diagnostic test created
Diagnostic passed with the expected clean skip on the current Mac
Real Intel Iris Xe GPU inference validated on a Windows laptop
Routable Intel GPU backend implemented
Production routing regression validated on Windows
```

Diagnostic:

```text
tests/test_openvino_gpu.py
```

The diagnostic inspects all OpenVINO device IDs beginning with `"GPU"`, queries
`FULL_DEVICE_NAME`, and dynamically selects an Intel GPU. It does not assume
that Intel is always `GPU.0`, and it cleanly skips when no Intel GPU is present.
The discovered ID is passed to `OpenVINOBackend(target_device=...)` and used
consistently for availability, result-identity, and `EXECUTION_DEVICES` checks.
It calls the backend's actual `run()` method rather than duplicating conversion,
preprocessing, or inference logic.

Observed evidence from the validated Windows laptop run selected `GPU.0`, identified as
`Intel(R) Iris(R) Xe Graphics (iGPU)`, and confirmed:

```text
OpenVINOBackend(target_device="GPU.0")
result identity: openvino_resnet18_gpu.0
exactly five predictions
matching CPU top-1 category
matching CPU top-5 category set regardless of ordering
confidence differences no greater than 0.1 percentage points
inference time: 48.95 ms
EXECUTION_DEVICES == ['GPU.0']
no warnings
```

The production Intel GPU backend uses the same dynamic discovery and has the
stable routing/benchmark identity `OpenVINO Intel GPU`. It does not assume that
the Intel device is `GPU.0` and does not use the generic `"GPU"` alias. Its base
scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The generic OpenVINO backend default remains `warmup_runs=0`. The registered
Intel GPU backend uses `warmup_runs=2` on first use of each backend instance.
The first result reports the configured count and total warm-up time, and all
warm-up work remains part of the first routed total. Later results report
`warmup_runs = 0` and `warmup_time_ms = 0.0`.

The LOW_POWER score records that no power-efficiency advantage has been
measured and strongly discourages selection; it is not a claim that the device
cannot execute the task. Once eligible history exists, the existing bounded
historical bonus can change CPU/GPU preference. Routing may legitimately select
CPU or Intel GPU according to the current policy, recent measurements, other
eligible backends, and system load. Hardware-specific observations do not imply
that Intel GPU always wins PERFORMANCE or BALANCED.

---

# 13. Intel NPU Status

Current state:

```text
Not implemented
Diagnostic test created
Diagnostic passed with the expected clean skip on the current Mac
Routable NPU backend not implemented
```

Diagnostic:

```text
tests/test_openvino_npu.py
```

The diagnostic imports OpenVINO optionally and exits with a clear successful
skip if the package is missing. It queries `Core().available_devices` and also
cleanly skips when OpenVINO does not expose `"NPU"`.

The diagnostic constructs `OpenVINOBackend(target_device="NPU")` and verifies
that detection and metadata refer to NPU. This configuration is diagnostic
only and is not registered with the router.

On compatible Intel NPU hardware, the diagnostic is intended to:

```text
compile ResNet18 explicitly for NPU
run test.png
report inference timing
verify five predictions
require the CPU reference top-1 category to match
require at least four overlapping top-5 categories
report confidence differences for inspection
assert EXECUTION_DEVICES == ['NPU']
```

The diagnostic was run on the current Mac and reported:

```text
Core().available_devices: ['CPU']
SKIPPED: OpenVINO NPU is not available on this machine.
```

The clean-skip behavior is therefore validated on this machine. Compilation
and inference on a real NPU remain unvalidated.

NPU use will require:

```text
compatible Intel NPU hardware
supported operating system
appropriate Intel NPU drivers/plugins
OpenVINO NPU availability
```

The current development Mac does not expose an NPU.

Do not add routable NPU support until real hardware validation exists.

---

# 14. NVIDIA CUDA Status

Current state:

```text
Implemented and routable for image_classification
Validated on NVIDIA RTX A1000 Laptop GPU on Windows
Validated on NVIDIA GeForce RTX 4090 on Linux/RunPod
```

The production `PyTorch CUDA` backend dynamically requires
`torch.cuda.is_available()` and at least one CUDA device. It intentionally uses
CUDA device 0 for this milestone, while its stable routing/benchmark identity
remains `PyTorch CUDA` and its stable result identity remains
`pytorch_cuda_resnet18`. The device index and actual GPU properties are runtime
metadata, not identity components.

Supported workload:

```text
TaskType.IMAGE_CLASSIFICATION
Torchvision ResNet18 with ImageNet pretrained weights and preprocessing
```

Current scores:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The generic backend defaults to zero warm-up runs. The registered production
backend performs two warm-up runs once per instance. First-use warm-up and model
initialization remain in the router's total execution time; subsequent results
report zero warm-up runs and `0.0` ms warm-up time.

CUDA history participates in `BenchmarkStats` and is isolated under the
`PyTorch CUDA` backend/task key. On the validated Windows evidence, measured
history made CUDA preferred for PERFORMANCE and BALANCED, while LOW_POWER
remained CPU-oriented. On the later RunPod RTX 4090 ComfyUI validation,
BALANCED continued to prefer Torchvision CPU for the small ResNet18 workload
because routing uses total execution time, even though CUDA model inference
itself was faster. Periodic stale-evidence refresh still selected CUDA after
each ten successful normal routes so its timing evidence could be updated.
These outcomes are not universal: hardware, drivers, system load, desktop GPU
activity, scheduling, and other processes can change timings and routing
outcomes. LOW_POWER is zero because NVIDIA power efficiency has not been
measured.

CUDA execution requires a compatible NVIDIA GPU and driver and CUDA-enabled
PyTorch/Torchvision wheels. The validated environment used PyTorch
`2.2.2+cu121` and Torchvision `0.17.2+cu121`. `pyproject.toml` remains unchanged
because it declares versions rather than a platform-specific wheel index. No
separate CUDA Toolkit installation is required for the validated wheel-based
runtime. On CPU-only or macOS systems, the backend reports unavailable without
constructing a model or initializing CUDA, so existing CPU/OpenVINO routing is
unchanged.

---

# 15. AMD / ROCm Status

Current state:

```text
Future
Not implemented
```

No ROCm-specific dependency is currently required.

---

# 16. Apple Acceleration Status

Current state:

```text
PyTorch MPS implemented and routable for image_classification
Validated on Apple M1 Pro, arm64, macOS 15.7.7
Core ML not implemented
```

The production `PyTorch MPS` backend is automatically available when PyTorch
reports both:

```text
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
```

Its stable routing and benchmark-history identity is `PyTorch MPS`, and its
stable result identity is `pytorch_mps_resnet18`. It supports:

```text
TaskType.IMAGE_CLASSIFICATION
Torchvision ResNet18 with ImageNet pretrained weights and preprocessing
```

Current scores:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

The generic backend defaults to zero warm-up runs. The registered production
backend performs two warm-up runs once per backend instance. First-use warm-up
and model initialization remain in the router's total execution time;
subsequent results report zero warm-up runs and `0.0` ms warm-up time.

Real hardware validation was performed on:

```text
Apple M1 Pro
arm64
macOS 15.7.7
Python 3.11.9
PyTorch 2.2.2
Torchvision 0.17.2
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
```

Real MPS ResNet18 inference and router participation were validated. In the
direct backend comparison on this M1 Pro, representative warm inference was
approximately 8.5 ms on MPS versus approximately 13 ms on Torchvision CPU.
Representative warm total execution was approximately 35 ms on MPS versus
approximately 38-43 ms on CPU. MPS had a larger first-run/cold-start cost.

In the 15-route BALANCED routing validation, cold-start exploration collected
five records for each backend. After evidence was available, Torchvision CPU
remained the BALANCED winner because its three-point higher base score
outweighed MPS's modest historical-performance advantage. MPS must not be
described as universally faster or as the guaranteed routing winner.
LOW_POWER is zero because no power-efficiency advantage has been measured.
MPS timing and routing results are hardware- and load-specific.

No Core ML or MPS-specific Python dependency is currently part of the validated
AI Router setup. MPS uses the existing base PyTorch/Torchvision stack.

---

# 17. Operating System — macOS

Current development has been validated on macOS with an Intel x86_64 Mac and
an Apple M1 Pro arm64 Mac.

Current OpenVINO device result:

```text
['CPU']
```

Expected current behavior:

```text
OpenVINO CPU        works
PyTorch MPS         works on the validated Apple M1 Pro
Intel GPU production backend is unavailable and the diagnostic cleanly skips
Intel NPU diagnostic cleanly skips
```

The Python.org Python installation may require certificate setup if HTTPS model downloads fail.

---

# 18. Operating System — Windows

Recommended baseline:

```text
64-bit Windows
Python 3.11
```

Current package installation should use:

```powershell
python -m pip install -e ".[openvino]"
```

Intel GPU and NPU support may require separate Intel hardware drivers.

The OpenVINO Python package does not replace those OS-level drivers.

Routable Intel GPU support is implemented. Routable Intel NPU support is not.

Routable NVIDIA CUDA support is implemented for CUDA device 0 when a compatible
NVIDIA driver and CUDA-enabled PyTorch build are available.

OpenVINO ResNet18 inference and production routing on Intel Iris Xe Graphics
have been validated on a Windows laptop. The diagnostic and production backend
discover the Intel GPU from `FULL_DEVICE_NAME` rather than relying on a fixed
OpenVINO device ID. The recorded `GPU.0` identity and timings are observations
from that hardware/run, not portable identifiers or guaranteed performance.

---

# 19. Operating System — Linux

Recommended baseline:

```text
64-bit Linux
Python 3.11
```

Ubuntu LTS is the preferred future platform for Intel GPU/NPU validation.

Current package installation should use:

```bash
python -m pip install -e ".[openvino]"
```

Intel GPU/NPU use requires appropriate Linux drivers in addition to the Python package.

---

# 20. Standard Library Dependencies

AI Router uses Python standard-library modules including:

```text
abc
dataclasses
enum
importlib
multiprocessing
pathlib
platform
statistics
time
typing
```

These do not require separate installation.

---

# 21. Required Development Tools

For normal project setup:

```text
Git
Python 3.11
pip
venv
```

Potential future/native development may also use:

```text
CMake
C/C++ compiler
platform build tools
```

The current validated installation uses prebuilt Python wheels and does not require compiling PyTorch or OpenVINO from source.

---

# 22. Current Model Assets

Current real inference uses:

```text
Torchvision ResNet18
ImageNet pretrained weights
```

Expected pretrained weight filename:

```text
resnet18-f37072fd.pth
```

The first run may download this model automatically.

Internet access is therefore normally required for the first real inference unless the model has already been cached.

---

# 23. Required Test Image

The current tests/examples expect:

```text
test.png
```

in the repository root.

If it is not included in the repository, provide a valid PNG file with that name before running the image-classification tests.

---

# 24. Network Requirements

Internet access may be required for:

```text
cloning the repository
installing Python packages
downloading pretrained ResNet18 weights
```

After dependencies and model assets are cached, core inference can run locally without repeated downloads.

---

# 25. Current Benchmark/Router Behavior

The current adaptive routing implementation uses:

```text
minimum history for historical scoring: 5 matching records
recent scoring window: latest 4 matching records
aggregation: median total execution time
historical bonus cap: +25
history scope: backend + task type
refresh counter scope: policy + task type
periodic refresh cadence: 10 successful normal routes, then 1 refresh
```

Historical performance scoring is:

```python
warm_time = median(latest_four_total_times)

if warm_time >= 60.0:
    bonus = 1000.0 / warm_time
else:
    bonus = 3000.0 / (warm_time + 120.0)
```

There is no 1 ms timing floor in the current formula. The fast branch is
bounded naturally at 25 points when `warm_time` is zero and joins the reciprocal
branch continuously at 60 ms.

Ordinary routing first cold-start explores every available, compatible
positive-base backend/task pair until it has five records. After that, normal
base-plus-history scoring selects the winner. To prevent a deterministic loser
from retaining indefinitely stale evidence, after ten successful normal routes
for a `(policy, task_type)` pair the next ordinary route refreshes the eligible
positive-base non-winner with the oldest matching benchmark evidence;
registration order breaks equal-age ties. The refresh counter then resets.
Base-score-zero candidates are excluded from cold-start and refresh collection,
and explicit `benchmark_backend` seeding remains separate from refresh counters.

Benchmark history and refresh counters are in memory for the lifetime of one
`AIRouter` instance. These are not package dependencies, but they are important
regression expectations when validating another machine.

---

# 26. Current OpenVINO Base Scores

Current OpenVINO image-classification policy scores:

```text
PERFORMANCE = 38
BALANCED    = 58
LOW_POWER   = 63
```

The current regression has validated that with representative historical measurements:

```text
PERFORMANCE → OpenVINO
BALANCED    → OpenVINO
LOW_POWER   → Torchvision
```

Exact timing and bonus values vary by machine.

Current PyTorch CUDA image-classification policy scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

Representative Windows history has demonstrated that CUDA can overcome its
three-point base disadvantage relative to Torchvision CPU for PERFORMANCE and
BALANCED. LOW_POWER remains CPU-oriented because CUDA has no measured power
advantage. These observations are hardware- and load-specific, not guaranteed
routing outcomes.

Current PyTorch MPS image-classification policy scores are:

```text
PERFORMANCE = 37
BALANCED    = 57
LOW_POWER   = 0
```

On the validated Apple M1 Pro, MPS produced a modest historical-performance
advantage over Torchvision CPU after cold-start exploration, but Torchvision
CPU remained the 15-route BALANCED winner because its base score is three
points higher. LOW_POWER remains zero because no MPS power-efficiency
advantage has been measured. These observations are hardware- and
load-specific, not guaranteed routing outcomes.

---

# 27. OpenVINO / Torchvision Performance Characteristics

Representative interleaved benchmark result from the current Mac:

```text
OpenVINO
--------
Cold total: ~1951 ms
Median warm total: ~47 ms
Warm inference: ~13–15 ms

Torchvision
-----------
Cold total: ~245 ms
Median warm total: ~57 ms
Warm inference: ~23–25 ms
```

Important:

```text
OpenVINO has a larger cold-start cost
but faster warm inference on this workload
```

These values are examples, not universal performance guarantees.

Representative direct backend comparison from the validated Apple M1 Pro:

```text
PyTorch MPS
-----------
Warm total: ~35 ms
Warm inference: ~8.5 ms

Torchvision CPU
---------------
Warm total: ~38-43 ms
Warm inference: ~13 ms
```

Important:

```text
MPS had a larger first-run/cold-start cost
MPS was not the guaranteed BALANCED routing winner
```

In the 15-route BALANCED routing validation, cold-start exploration collected
five records for each backend. After evidence was available, Torchvision CPU
remained the BALANCED winner because its three-point higher base score
outweighed MPS's modest historical-performance advantage. These values are
hardware- and load-specific examples, not universal performance guarantees.

---

# 28. Known Compatibility Issue — NumPy 2.x

If you see:

```text
_ARRAY_API not found
```

or a warning that a module compiled for NumPy 1.x cannot run under NumPy 2.x:

verify:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Expected:

```text
1.26.4
```

Current fix:

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

---

# 29. Known macOS Issue — SSL Certificates

Possible error:

```text
SSL: CERTIFICATE_VERIFY_FAILED
```

For an official Python.org Python 3.11 installation, the validated fix is:

```bash
open "/Applications/Python 3.11/Install Certificates.command"
```

This runs the certificate setup supplied by that Python installation.

Do not edit Python SSL code.

Do not permanently disable certificate verification.

---

# 30. Known macOS Issue — Multiprocessing `<stdin>`

Running complex OpenVINO conversion/inference through:

```bash
python - <<'PY'
```

previously caused multiprocessing child processes to try to reopen:

```text
<stdin>
```

and emit `FileNotFoundError`.

Use the standalone regression test instead:

```bash
python -B tests/test_openvino_cpu.py
```

It has:

```python
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
```

---

# 31. Required Repository Structure

A current working checkout should include at least:

```text
backends/
    cpu.py
    cuda.py
    mps.py
    mock_accelerator.py
    openvino.py
    torchvision_classifier.py

core/
    backend.py
    backend_registry.py
    benchmark.py
    policy.py
    registry.py
    router.py
    task.py
    task_types.py

examples/
    detect_devices.py
    run_task.py
    run_task_png.py

tests/
    test_mps_backend.py
    test_openvino_cpu.py
    test_openvino_gpu.py
    test_openvino_npu.py

pyproject.toml
README_AI.md
BUILD.md
REQUIREMENTS.md
```

Current image tests also require:

```text
test.png
```

---

# 32. Clean Installation Commands

## Base install

```bash
python -m pip install -e .
```

Installs:

```text
NumPy
PyTorch
Torchvision
Pillow
AI Router
```

## Full current OpenVINO install

```bash
python -m pip install -e ".[openvino]"
```

Installs:

```text
NumPy
PyTorch
Torchvision
Pillow
OpenVINO
AI Router
```

This is the recommended current setup.

---

# 33. Clean Environment Verification

A clean machine is ready for the current full feature set when all applicable checks succeed:

```text
[ ] Python reports 3.11.x
[ ] AI Router installs with python -m pip install -e ".[openvino]"
[ ] NumPy reports 1.26.4
[ ] PyTorch reports 2.2.2
[ ] Torchvision reports 0.17.2
[ ] Pillow imports successfully
[ ] OpenVINO reports 2025.4.1
[ ] OpenVINO Core initializes
[ ] examples/detect_devices.py runs
[ ] examples/run_task.py runs
[ ] examples/run_task_png.py runs
[ ] tests/test_openvino_cpu.py passes
[ ] tests/test_openvino_gpu.py passes or cleanly skips
[ ] tests/test_openvino_npu.py passes or cleanly skips
```

---

# 34. Current pyproject.toml Expectation

The current dependency sections should look like:

```toml
[project]
name = "ai-router"
version = "0.1.0"
description = "Modular AI workload router for CPU, GPU and NPU devices"
requires-python = ">=3.11,<3.12"

dependencies = [
    "numpy==1.26.4",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "pillow",
]

[project.optional-dependencies]

openvino = [
    "openvino==2025.4.1",
]
```

The exact package-discovery section may follow below this.

---

# 35. Packaging Policy

Use:

```text
pyproject.toml
```

as the machine-readable dependency source.

Do not maintain a duplicate `requirements.txt` with the same package list unless there is a specific future need.

This avoids version drift between multiple dependency files.

---

# 36. Future Optional Extras

As more backends become real and validated, the packaging model may expand to include extras such as:

```text
ai-router[openvino]
ai-router[cuda]
ai-router[dev]
```

Possible future combined installation:

```bash
python -m pip install -e ".[openvino,cuda]"
```

Do not add these extras until the corresponding backend and dependency strategy have been validated.

---

# 37. Requirements vs OS Drivers

`pyproject.toml` installs Python packages.

It cannot install or fully manage all operating-system hardware drivers.

Examples of requirements that may remain external:

```text
Intel GPU driver
Intel NPU driver
NVIDIA driver
CUDA system/runtime requirements
OS-specific OpenCL components
```

These must be documented separately as each hardware backend is implemented.

---

# 38. Backend and Device Identity

AI Router represents backend identity, physical device class, runtime, and
accelerator API as separate metadata.

`BackendInfo.name` is the stable backend identity used for routing and benchmark
history.

`BackendInfo.device_type` identifies the physical device class, such as `cpu`,
`gpu`, `npu`, or `accelerator`.

`BackendInfo.runtime` identifies the software execution layer, such as `native`,
`pytorch`, or `openvino`.

`BackendInfo.accelerator_api` identifies an optional hardware acceleration API,
such as `cuda` or `mps`. Backends without a distinct accelerator API use
`None`.

Backend contributions should follow the identity and metadata requirements in
`BACKEND_GUIDE.md`.

Routing output must preserve these identities separately so that backend,
physical-device, runtime, and accelerator information are not overloaded or
ambiguous.

---

# 39. Source of Truth

Use:

```text
pyproject.toml
```

for Python dependency declarations.

Use:

```text
BUILD.md
```

for complete clean-machine setup instructions.

Use:

```text
REQUIREMENTS.md
```

for software/hardware/platform compatibility information.

Use:

```text
README_AI.md
```

for current development state and next-step handover.

If these files and the repository disagree, inspect the actual code first and update stale documentation before making further architectural changes.
