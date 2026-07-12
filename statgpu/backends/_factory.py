"""
Backend factory: select the appropriate compute backend automatically or
explicitly by name.
"""

from statgpu.backends._base import BackendBase
from statgpu.backends._numpy import NumpyBackend
from statgpu.backends._cupy import CuPyBackend
from statgpu.backends._torch import TorchBackend

# Module-level singletons (one instance per library, shared across calls).
_numpy_backend = NumpyBackend()
_cupy_backend = CuPyBackend()
_torch_backend = TorchBackend(device="cuda")
_torch_cpu_backend = TorchBackend(device="cpu")


def get_backend(backend: str = "auto", device: str = "auto") -> BackendBase:
    """
    Return a compute backend instance.

    Parameters
    ----------
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Which array library to use.

        * ``'numpy'`` – always use NumPy (CPU).
        * ``'cupy'``  – use CuPy (requires a CUDA GPU and the ``cupy`` package).
        * ``'torch'`` – use PyTorch (requires the ``torch`` package; defaults
          to CUDA if available, else CPU).
        * ``'auto'``  – pick automatically: CuPy if available, else PyTorch
          CUDA if available, else NumPy.

    device : {'auto', 'cpu', 'cuda'}, default='auto'
        Hint about the target device. Ignored when *backend* is explicitly
        set to a non-``'auto'`` value. When ``'cpu'``, always returns the
        NumPy backend regardless of GPU availability.

    Returns
    -------
    BackendBase
        A backend instance that can be used to create/convert arrays.

    Examples
    --------
    >>> from statgpu.backends import get_backend
    >>> xp = get_backend().xp      # numpy, cupy, or torch depending on hw
    >>> arr = xp.zeros((3, 3))
    """
    backend = str(backend).strip().lower()
    device = str(device).strip().lower()

    if backend not in {"auto", "numpy", "cupy", "torch"}:
        raise ValueError(
            "backend must be one of: 'auto', 'numpy', 'cupy', 'torch'"
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'")

    if backend == "numpy":
        return _numpy_backend
    if backend == "cupy":
        return _cupy_backend
    if backend == "torch":
        # ``backend='torch'`` selects the array library, while ``device``
        # selects CPU versus CUDA.  Keep estimator ``device='torch'`` strict by
        # passing device='cuda' from BaseEstimator, but allow functional APIs to
        # use a Torch CPU backend when CUDA is unavailable, as documented.
        if device == "cpu":
            return _torch_cpu_backend
        if device == "cuda":
            return _torch_backend
        return _torch_backend if _torch_backend.is_available() else _torch_cpu_backend

    # --- auto-selection ---
    if device == "cpu":
        return _numpy_backend

    # Prefer CuPy → PyTorch CUDA → NumPy
    if _cupy_backend.is_available():
        return _cupy_backend
    if _torch_backend.is_available():
        return _torch_backend
    return _numpy_backend
