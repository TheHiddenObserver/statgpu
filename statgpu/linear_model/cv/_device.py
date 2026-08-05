"""Shared strict device/backend resolution for dedicated CV routines."""

from __future__ import annotations

from statgpu._config import Device
from statgpu.backends import get_backend
from statgpu.glm_core._validation import validate_glm_sample_weight


def normalize_cv_device(device):
    """Normalize string/enum device values without silently accepting typos."""
    if isinstance(device, Device):
        return device.value
    name = str(device).strip().lower()
    if name.startswith("device."):
        name = name.split(".", 1)[1]
    valid = {item.value for item in Device}
    if name not in valid:
        expected = ", ".join(sorted(valid))
        raise ValueError(f"Invalid device {device!r}. Expected one of: {expected}")
    return name


def _array_gpu_backend(value):
    """Return the GPU array library already owning *value*, if any."""
    module = type(value).__module__
    if module.startswith("cupy"):
        return "cupy"
    if module.startswith("torch"):
        try:
            device = str(value.device)
        except Exception:
            return None
        return "torch" if device.startswith("cuda") else None
    return None


def _backend_name(backend):
    name = type(backend).__name__.lower()
    if "torch" in name:
        return "torch"
    if "cupy" in name:
        return "cupy"
    return "numpy"


def resolve_cv_backend(device, X):
    """Resolve a dedicated-CV backend while preserving explicit library choice."""
    device_name = normalize_cv_device(device)
    input_backend = _array_gpu_backend(X)

    if device_name == Device.CPU.value:
        if input_backend is not None:
            raise ValueError(
                "device='cpu' cannot consume a GPU-resident design matrix; "
                "move X to CPU or request the matching GPU backend."
            )
        backend_name = "numpy"
        backend = get_backend(backend="numpy", device="cpu")
    elif device_name == Device.TORCH.value:
        if input_backend not in (None, "torch"):
            raise ValueError(
                "device='torch' cannot silently switch a CuPy design matrix "
                "to another GPU library."
            )
        backend_name = "torch"
        backend = get_backend(backend="torch", device="cuda")
    elif device_name == Device.CUDA.value:
        if input_backend not in (None, "cupy"):
            raise ValueError(
                "device='cuda' selects CuPy and cannot silently switch a "
                "Torch CUDA design matrix to another GPU library."
            )
        backend_name = "cupy"
        backend = get_backend(backend="cupy", device="cuda")
    else:
        if input_backend is not None:
            backend_name = input_backend
            backend = get_backend(backend=input_backend, device="cuda")
        else:
            backend = get_backend(backend="auto", device="auto")
            backend_name = _backend_name(backend)

    use_gpu = backend_name in {"cupy", "torch"}
    return (
        device_name,
        backend_name,
        backend,
        use_gpu,
        input_backend == "cupy",
        input_backend == "torch",
    )


def validate_cv_sample_weight(sample_weight, n_samples):
    """Validate analytic CV weights before any grid or degenerate return."""
    if sample_weight is None:
        return None
    return validate_glm_sample_weight(sample_weight, n_samples)
