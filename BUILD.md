# AI Router Build Guide

`README.md` is the source of truth. This is the shortest supported setup path.

## 1. Clone

```bash
git clone https://github.com/TabiUK/ai-router.git
cd ai-router
```

## 2. Create a Python 3.11 virtual environment

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Confirm that the active interpreter is Python 3.11:

```bash
python --version
python -m pip install --upgrade pip
```

## 3. Install

Standard installation:

```bash
python -m pip install -e .
```

### Windows with NVIDIA CUDA

Install these wheels before AI Router to avoid selecting CPU-only PyTorch:

```powershell
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

This requires a compatible NVIDIA GPU and driver, but not a separate CUDA Toolkit.

### Optional OpenVINO

Install the OpenVINO extra when OpenVINO CPU or Intel GPU support is wanted:

```bash
python -m pip install -e ".[openvino]"
```

On a Windows NVIDIA system, install the CUDA wheels first, then install this
extra. OpenVINO GPU availability also depends on compatible Intel hardware and
drivers.

## 4. Run device detection

```bash
python examples/detect_devices.py
```

Unavailable optional runtimes should not prevent other backends from loading.

## 5. Run tests

Core regressions:

```bash
python -B tests/test_router_cold_start.py
python -B tests/test_mock_backend_opt_in.py
```

Run only the hardware-specific tests applicable to the current machine:

```bash
python -B tests/test_openvino_cpu.py
python -B tests/test_openvino_gpu.py
python -B tests/test_cuda.py
python -B tests/test_cuda_backend.py
python -B tests/test_mps_backend.py
```

## Done

AI Router is ready when device detection runs, applicable tests pass, and the
pretrained ResNet18 weights have downloaded successfully on first use.
