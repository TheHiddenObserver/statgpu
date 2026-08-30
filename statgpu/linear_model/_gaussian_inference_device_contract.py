"""Host-only Torch device selection contract for shared Gaussian inference."""

from __future__ import annotations

from . import _gaussian_inference as _gaussian_inference


def _install_host_torch_device_contract() -> None:
    """Use CPU when explicit Torch inference has no native/device authority."""
    current = _gaussian_inference._as_backend_array
    if getattr(current, "_statgpu_host_torch_device_contract", False):
        return

    def _as_backend_array(value, backend: str, *, like=None, device=None):
        if backend == "torch" and device is None:
            import torch

            native_authority = isinstance(like, torch.Tensor) or isinstance(
                value, torch.Tensor
            )
            if not native_authority:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        return current(value, backend, like=like, device=device)

    _as_backend_array._statgpu_host_torch_device_contract = True
    _as_backend_array._statgpu_original = current
    _gaussian_inference._as_backend_array = _as_backend_array


_install_host_torch_device_contract()
