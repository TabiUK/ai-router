from core.registry import discover_backends


print("AI Router")
print("---------")

for backend in discover_backends():
    info = backend.detect()

    print()
    print(f"Device:       {info.name}")
    print(f"Type:         {info.device_type}")
    print(f"Available:    {info.available}")
    print(f"Capabilities: {backend.capabilities()}")
    print(f"Details:      {info.details}")