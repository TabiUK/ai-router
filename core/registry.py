import importlib
import pkgutil

import backends
from core.backend_registry import get_registered_backends


def _load_backend_modules() -> None:
    for module_info in pkgutil.iter_modules(backends.__path__):
        importlib.import_module(f"backends.{module_info.name}")


def discover_backends():
    _load_backend_modules()
    return get_registered_backends()