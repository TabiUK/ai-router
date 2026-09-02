# AI Router — Clean System Build & Setup Guide

## Purpose

This guide is intended to let someone start with a clean macOS, Windows, or Linux machine, clone AI Router, create an isolated Python environment, install the project and its dependencies, and run the currently implemented examples and regression tests.

For dependency/platform details, also read:

```text
REQUIREMENTS.md
```

For the latest development state, read:

```text
README_AI.md
```

Contributors adding or changing a backend should follow:

```text
BACKEND_GUIDE.md
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

Reference development environment:

```text
Python 3.11.9
macOS
Intel x86_64
```

The project currently has:

```text
Torchvision CPU inference       ✅
OpenVINO CPU inference          ✅
OpenVINO CPU routing            ✅
Intel GPU diagnostic            ✅ created
Intel GPU routable backend      ✅ implemented and validated
Intel NPU support               ⬜ not implemented
NVIDIA CUDA backend             ✅ implemented and validated on Windows
RunPod RTX 4090 source validation ✅ CUDA routing + ComfyUI on Linux
```

---

# 2. What Must Be Installed Before AI Router

Install these first:

```text
Git
Python 3.11 (64-bit)
pip
Python venv support
Internet access for first-time package/model downloads
```

The current Python packages are installed through `pyproject.toml`.

You should NOT normally install NumPy, Torch, Torchvision, Pillow, and OpenVINO manually one by one.

---

# 3. Clone the Repository

Clone the public repository:

```bash
git clone https://github.com/TabiUK/ai-router.git
cd ai-router
```

You should see files/folders similar to:

```text
backends/
core/
examples/
integrations/
tasks/
tests/
pyproject.toml
README_AI.md
BUILD.md
REQUIREMENTS.md
```

---

# 4. Create a Virtual Environment

AI Router should be run inside a Python virtual environment.

## macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not available under that exact name, verify your Python installation:

```bash
python3 --version
```

Use Python 3.11 for the current validated setup.

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

## Windows Command Prompt

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 5. Upgrade pip

With the virtual environment activated:

```bash
python -m pip install --upgrade pip
```

---

# 6. Install AI Router

AI Router now declares its base Python dependencies in:

```text
pyproject.toml
```

## Base installation

Use this if you do NOT need OpenVINO:

```bash
python -m pip install -e .
```

This installs:

```text
AI Router
NumPy 1.26.4
PyTorch 2.2.2
Torchvision 0.17.2
Pillow
```

## Recommended current installation

To install AI Router with OpenVINO support:

```bash
python -m pip install -e ".[openvino]"
```

This installs:

```text
AI Router
NumPy 1.26.4
PyTorch 2.2.2
Torchvision 0.17.2
Pillow
OpenVINO 2025.4.1
```

This is the recommended command for the current full feature set.

---

# 7. Why OpenVINO Is Optional

OpenVINO is declared as an optional dependency.

That means:

```bash
python -m pip install -e .
```

installs the base project without OpenVINO.

While:

```bash
python -m pip install -e ".[openvino]"
```

installs the base project plus OpenVINO.

This allows AI Router to remain modular and prevents Intel-specific runtime support from becoming mandatory for every installation.

---

# 8. Verify AI Router Package Metadata

Run:

```bash
python -m pip show ai-router
```

Expected base requirements:

```text
Requires: numpy, pillow, torch, torchvision
```

OpenVINO does not appear in the normal `Requires:` list because it is an optional extra.

To inspect optional extras:

```bash
python - <<'PY'
from importlib.metadata import metadata

m = metadata("ai-router")

print("Provides-Extra:", m.get_all("Provides-Extra"))
print("Requires-Dist:")

for item in m.get_all("Requires-Dist") or []:
    print(" ", item)
PY
```

Expected:

```text
Provides-Extra: ['openvino']

Requires-Dist:
  numpy==1.26.4
  torch==2.2.2
  torchvision==0.17.2
  pillow
  openvino==2025.4.1; extra == "openvino"
```

---

# 9. Verify Python Dependencies

Run:

```bash
python - <<'PY'
import numpy
import torch
import torchvision
import PIL

print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Pillow:", PIL.__version__)
PY
```

Expected core versions:

```text
NumPy: 1.26.4
PyTorch: 2.2.2
Torchvision: 0.17.2
```

Pillow is currently required but not pinned to one exact version.

---

# 10. Verify OpenVINO

If installed with:

```bash
python -m pip install -e ".[openvino]"
```

run:

```bash
python - <<'PY'
import openvino as ov

core = ov.Core()

print("OpenVINO version:", ov.__version__)
print("Available devices:", core.available_devices)
PY
```

On the reference Intel Mac this reports:

```text
OpenVINO version:
2025.4.1-20426-82bbf0292c5-releases/2025/4

Available devices:
['CPU']
```

Your machine may report:

```text
['CPU']
['CPU', 'GPU']
['CPU', 'GPU', 'NPU']
```

depending on hardware, OS support, drivers, and OpenVINO plugins.

Installing OpenVINO with pip does NOT automatically install every GPU/NPU driver required by the operating system.

---

# 11. Required Test Image

Current image-classification examples/tests expect:

```text
test.png
```

in the project root.

Verify:

```bash
ls test.png
```

Windows PowerShell:

```powershell
Get-Item test.png
```

If it is not included in the repository, copy a normal PNG file into the project root and name it:

```text
test.png
```

---

# 12. First-Run ResNet18 Download

Torchvision uses pretrained ResNet18 ImageNet weights.

On the first run it may download:

```text
resnet18-f37072fd.pth
```

The reference Mac cached it under a path similar to:

```text
~/.cache/torch/hub/checkpoints/
```

Internet access is normally required for the first model download.

Later runs use the local cache.

---

# 13. macOS Certificate Fix

On some macOS systems using the official Python.org Python installer, the first Torchvision model download may fail with:

```text
SSL: CERTIFICATE_VERIFY_FAILED
unable to get local issuer certificate
```

This normally means the Python installation has not yet installed or linked its trusted certificate bundle correctly.

## Fix

Open Terminal and run:

```bash
open "/Applications/Python 3.11/Install Certificates.command"
```

This launches the certificate setup script installed with Python 3.11.

You do NOT need to edit Python source code.

You do NOT need to manually change SSL settings.

The script updates the certificate bundle used by that Python installation.

After it finishes, reactivate the virtual environment if required:

```bash
cd /path/to/ai-router
source .venv/bin/activate
```

Then rerun the command that previously failed, for example:

```bash
python examples/run_task_png.py
```

## If the certificate script does not exist

Check:

```bash
which python3
python3 --version
```

If you installed Python using the official Python.org installer, also check:

```bash
ls "/Applications/Python 3.11/"
```

You should normally see:

```text
Install Certificates.command
```

If Python was installed using Homebrew, Conda, pyenv, or another package manager, use the certificate setup appropriate for that Python distribution.

Do not create `Install Certificates.command` manually.

## Do not disable certificate verification

Do not work around the problem with code such as:

```python
ssl._create_default_https_context = ssl._create_unverified_context
```

Do not permanently disable TLS/SSL verification.

---

# 14. Verify Backend Discovery

Run:

```bash
python examples/detect_devices.py
```

This should detect registered backends without crashing.

With OpenVINO installed, OpenVINO should report its detected runtime devices.

---

# 15. Run Generic Routing Example

Run:

```bash
python examples/run_task.py
```

This intentionally injects and exercises the Mock Accelerator as a synthetic
routing demonstration. The Mock backend is not included in normal production
backend discovery.

The Mock Accelerator is synthetic test hardware and should not be interpreted as a real performance result.

---

# 16. Run Image Classification Example

Run:

```bash
python examples/run_task_png.py
```

This uses:

```text
TaskType.IMAGE_CLASSIFICATION
```

with:

```text
test.png
```

and exercises the real image-classification path.

---

# 17. Run OpenVINO CPU Regression Test

Run:

```bash
python -B tests/test_openvino_cpu.py
```

This is the reusable real OpenVINO CPU regression.

It currently checks:

```text
OpenVINO runtime detection
missing-package safety
OpenVINO CPU ResNet18 inference
Torchvision/OpenVINO prediction parity
benchmark recording
historical performance scoring
policy routing
CPU-only OpenVINO compilation
unsupported-task behavior
```

A successful run should complete without an exception.

Do not expect exact timing values to match the reference development Mac.

---

# 18. Run Intel GPU Diagnostic

Run:

```bash
python -B tests/test_openvino_gpu.py
```

This direct diagnostic is separate from the production routing regression. It
enumerates OpenVINO `GPU*` device IDs, queries `FULL_DEVICE_NAME`, and selects
only an Intel GPU. It never assumes that Intel is `GPU.0` and never uses the
generic `"GPU"` alias.

On a machine without an OpenVINO GPU device, expected behavior is a clean successful skip such as:

```text
SKIPPED: OpenVINO GPU is not available on this machine.
```

On compatible Intel GPU hardware, the diagnostic compiles ResNet18 for the
dynamically selected physical device ID and validates real inference and CPU
prediction parity.

The registered production backend has the stable routing identity
`OpenVINO Intel GPU`. The generic `OpenVINOBackend` defaults to
`warmup_runs=0`; the registered Intel GPU backend uses `warmup_runs=2`. Those
warm-ups happen once inside its first routed execution, so their cost remains
visible in the first router total. Later results report zero warm-ups.

Run the production routing regression with:

```bash
python -B tests/test_openvino_gpu_routing.py
```

Routing may select OpenVINO CPU or Intel GPU according to policy, recent
backend/task benchmark history, and current system load. Observed results from
one machine or run are evidence, not a promise that either backend always wins.

---

# 19. Recommended Verification Sequence

After installation:

```bash
python --version
```

Then:

```bash
python - <<'PY'
import numpy
import torch
import torchvision
import PIL

print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Pillow:", PIL.__version__)
PY
```

If using OpenVINO:

```bash
python - <<'PY'
import openvino as ov

core = ov.Core()

print("OpenVINO:", ov.__version__)
print("Devices:", core.available_devices)
PY
```

Then run:

```bash
python examples/detect_devices.py
python examples/run_task.py
python examples/run_task_png.py
python -B tests/test_router_cold_start.py
python -B tests/test_openvino_cpu.py
python -B tests/test_openvino_gpu.py
python -B tests/test_openvino_gpu_routing.py
python -B tests/test_mock_backend_opt_in.py
```

---

# 20. macOS Clean Install — Quick Start

Install:

```text
Git
Python 3.11
```

If Apple command-line tools are required:

```bash
xcode-select --install
```

Then:

```bash
git clone https://github.com/TabiUK/ai-router.git
cd ai-router

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -e ".[openvino]"
```

Then verify:

```bash
python examples/detect_devices.py
python -B tests/test_router_cold_start.py
python -B tests/test_openvino_cpu.py
python -B tests/test_openvino_gpu.py
python -B tests/test_openvino_gpu_routing.py
```

If the first model download fails with an SSL certificate error, use the macOS certificate fix documented above.

---

# 21. Linux Clean Install — Quick Start

Install:

```text
Git
Python 3.11
Python venv support
pip
```

Then:

```bash
git clone https://github.com/TabiUK/ai-router.git
cd ai-router

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -e ".[openvino]"
```

Then:

```bash
python examples/detect_devices.py
python -B tests/test_router_cold_start.py
python -B tests/test_openvino_cpu.py
python -B tests/test_openvino_gpu.py
python -B tests/test_openvino_gpu_routing.py
```

## Intel GPU/NPU on Linux

If Intel GPU or NPU support is required, install the appropriate Intel Linux driver stack separately.

Do not assume:

```bash
pip install openvino
```

installs all required GPU/NPU OS drivers.

---

# 22. Windows Clean Install — Quick Start

Install:

```text
Git for Windows
64-bit Python 3.11
```

Open PowerShell:

```powershell
git clone https://github.com/TabiUK/ai-router.git
cd ai-router

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -e ".[openvino]"
```

Then:

```powershell
python examples\detect_devices.py
python -B tests\test_router_cold_start.py
python -B tests\test_openvino_cpu.py
python -B tests\test_openvino_gpu.py
python -B tests\test_openvino_gpu_routing.py
```

If PowerShell blocks activation scripts, either use Command Prompt activation or follow your organization’s approved PowerShell execution-policy process.

Intel GPU/NPU use may require separate Intel Windows driver installation.

---

# 23. Base Installation Without OpenVINO

For systems where OpenVINO is not wanted:

```bash
python -m pip install -e .
```

This installs only the base dependencies declared in `pyproject.toml`.

The OpenVINO backend must fail gracefully when OpenVINO is absent.

---

# 24. Troubleshooting

## `ModuleNotFoundError: No module named 'backends'`

Verify you are in the repository root and `.venv` is active.

Then run:

```bash
python -m pip install -e .
```

or, for the OpenVINO setup:

```bash
python -m pip install -e ".[openvino]"
```

---

## NumPy / `_ARRAY_API not found`

Verify:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Expected:

```text
1.26.4
```

If necessary:

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

Then rerun the failing test.

---

## `SSL: CERTIFICATE_VERIFY_FAILED`

Use the macOS certificate procedure in this document if you installed Python from Python.org.

Do not disable SSL globally.

---

## `FileNotFoundError: test.png`

Ensure:

```text
test.png
```

exists in the project root.

---

## Multiprocessing `<stdin>` errors on macOS

Complex OpenVINO conversion tests previously produced repeated errors when run using:

```bash
python - <<'PY'
```

because spawned child processes attempted to reopen `<stdin>`.

Use:

```bash
python -B tests/test_openvino_cpu.py
```

instead.

The standalone test contains a proper `__main__` guard and `multiprocessing.freeze_support()`.

---

## OpenVINO only reports `['CPU']`

This can be completely normal.

GPU/NPU visibility depends on:

```text
hardware
operating system
drivers
OpenVINO plugins
```

The reference Intel Mac reports only:

```text
['CPU']
```

---

# 24. NVIDIA CUDA Runtime

The production `PyTorch CUDA` backend supports `image_classification` with the
same pretrained Torchvision ResNet18 weights and preprocessing as the CPU path.
It detects CUDA dynamically through PyTorch and is available only when
`torch.cuda.is_available()` is true and at least one CUDA device is reported.
The registered production backend intentionally selects CUDA device 0; the
actual device name and properties are reported at runtime.

The validated Windows environment used:

```text
PyTorch       2.2.2+cu121
Torchvision   0.17.2+cu121
Compiled CUDA 12.1
GPU           NVIDIA RTX A1000 Laptop GPU
```

After the normal editable install, the matching CUDA-enabled wheel variants can
be selected on a compatible Windows/NVIDIA system with:

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip check
```

A compatible NVIDIA driver is required. The validated wheel-based setup does
not require a separate CUDA Toolkit installation. On CPU-only or macOS systems,
retain the normal PyTorch wheels; the CUDA backend reports unavailable without
initializing CUDA and the existing CPU/OpenVINO candidates continue normally.

Production CUDA policy scores are PERFORMANCE 37, BALANCED 57, and LOW_POWER
0. The registered backend performs two warm-up runs on first use only, and its
records participate in the existing `PyTorch CUDA` benchmark history. In the
validated Windows regression, measured history caused CUDA to win PERFORMANCE
and BALANCED while LOW_POWER remained CPU-oriented. These results are not
universal: hardware, drivers, system load, desktop GPU use, scheduling, and
other processes can change timings and routing outcomes.

CUDA-specific regression commands are:

```powershell
.venv\Scripts\python.exe -B tests\test_cuda.py
.venv\Scripts\python.exe -B tests\test_cuda_resnet18.py
.venv\Scripts\python.exe -B tests\test_cuda_backend.py
```

---

# 25. Current pyproject.toml Role

`pyproject.toml` is now the machine-readable dependency source for the current project.

It should include:

```toml
[project]
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

Therefore the normal clean install should use:

```bash
python -m pip install -e ".[openvino]"
```

Do not maintain a duplicate `requirements.txt` containing the same dependency list unless the project later has a concrete reason to do so.

---

# 26. Future Dependency Extras

Packaging may later evolve toward optional extras such as:

```text
ai-router[openvino]
ai-router[cuda]
ai-router[dev]
```

Potential future combined install:

```bash
python -m pip install -e ".[openvino,cuda]"
```

No CUDA extra exists yet. Keep the current version declarations in
`pyproject.toml`; platform-specific CUDA wheel selection remains an explicit
environment setup step until a portable packaging design is chosen and
validated.

---

# 27. Clean-System Readiness Checklist

```text
[ ] Git installed
[ ] Python 3.11 installed
[ ] Repository cloned
[ ] .venv created
[ ] .venv activated
[ ] pip upgraded
[ ] python -m pip install -e ".[openvino]" completed
[ ] Python reports 3.11.x
[ ] NumPy reports 1.26.4
[ ] PyTorch reports 2.2.2
[ ] Torchvision reports 0.17.2
[ ] Pillow imports successfully
[ ] OpenVINO imports successfully
[ ] OpenVINO devices inspected
[ ] test.png exists
[ ] examples/detect_devices.py runs
[ ] examples/run_task.py runs
[ ] examples/run_task_png.py runs
[ ] tests/test_router_cold_start.py passes
[ ] tests/test_openvino_cpu.py passes
[ ] tests/test_openvino_gpu.py passes or cleanly skips
[ ] tests/test_openvino_gpu_routing.py passes or cleanly skips
[ ] tests/test_mock_backend_opt_in.py passes
[ ] tests/test_openvino_npu.py passes or cleanly skips
```

If all applicable checks pass, the clean system is ready for the currently implemented AI Router feature set.

---

# 28. Source of Truth

Use:

```text
pyproject.toml
```

for machine-readable Python dependencies.

Use:

```text
BUILD.md
```

for clean installation and setup instructions.

Use:

```text
REQUIREMENTS.md
```

for dependency/platform/hardware compatibility notes.

Use:

```text
README_AI.md
```

for the latest development status.

If documentation and code disagree, inspect the repository and update the stale documentation before relying on it.
