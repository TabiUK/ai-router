from core.registry import discover_backends


print("AI Router")
print("---------")

for backend in discover_backends():
    info = backend.detect()

    print()
    print(f"Device:       {info.name}")
    print(f"Type:         {info.device_type.value}")
    print(f"Runtime:      {info.runtime.value}")
    print(
        "Accelerator API: "
        f"{info.accelerator_api.value if info.accelerator_api is not None else None}"
    )
    print(f"Available:    {info.available}")
    print(f"Capabilities: {backend.capabilities()}")
    print(f"Details:      {info.details}")
