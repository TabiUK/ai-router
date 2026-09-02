from core.backend import Backend


_backends: list[type[Backend]] = []


def register_backend(backend_class: type[Backend]) -> None:
    if backend_class not in _backends:
        _backends.append(backend_class)


def get_registered_backends() -> list[Backend]:
    return [backend_class() for backend_class in _backends]