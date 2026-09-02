# Standalone PyTorch CUDA tensor smoke test.
import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print(
            "SKIPPED: CUDA is not available through the installed "
            "PyTorch build."
        )
        return

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)

    device_name = torch.cuda.get_device_name(device)
    device_capability = torch.cuda.get_device_capability(device)
    device_properties = torch.cuda.get_device_properties(device)

    print(f"Selected CUDA device index: {device_index}")
    print(f"Selected CUDA device name: {device_name}")
    print(
        "Selected CUDA compute capability: "
        f"{device_capability[0]}.{device_capability[1]}"
    )
    print(
        "Selected CUDA total memory: "
        f"{device_properties.total_memory} bytes"
    )

    left = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        device=device,
    )
    right = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        device=device,
    )

    result = left @ right
    torch.cuda.synchronize(device)

    expected = torch.tensor(
        [
            [19.0, 22.0],
            [43.0, 50.0],
        ]
    )
    actual = result.cpu()

    torch.testing.assert_close(actual, expected)

    print(f"CUDA tensor result: {actual.tolist()}")
    print("PyTorch CUDA tensor smoke test passed.")


if __name__ == "__main__":
    main()
