"""
Base classes for statgpu estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

from statgpu._config import Device, get_device
from statgpu.backends import get_backend, BackendBase, _get_torch_device_str


class BaseEstimator(ABC):
    """
    Base class for all statgpu estimators.
    
    Provides common functionality for device management and input validation.
    """
    
    def __init__(
        self,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize base estimator.
        
        Parameters
        ----------
        device : str or Device, default='auto'
            Computation device: 'cpu', 'cuda', or 'auto'.
        n_jobs : int, optional
            Number of parallel jobs for CPU computation.
            -1 means using all processors.
        """
        self.device = device if isinstance(device, Device) else Device(device)
        self.n_jobs = n_jobs
        self._fitted = False
    
    def _get_compute_device(self) -> Device:
        """Resolve device for actual computation."""
        if self.device == Device.AUTO:
            return get_device()
        return self.device

    def _get_backend(self, backend: str = "auto") -> BackendBase:
        """
        Return the compute backend appropriate for this estimator's device.

        Parameters
        ----------
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Override which array library to use.  When ``'auto'``, the backend
            is chosen based on :attr:`device` and GPU availability.

        Returns
        -------
        BackendBase
            A backend instance whose :attr:`~BackendBase.xp` attribute is the
            underlying array module (NumPy, CuPy, or PyTorch).
        """
        compute_device = self._get_compute_device()
        device_str = compute_device.value  # 'cpu', 'cuda', or 'torch'

        if (
            self.device != Device.AUTO
            and compute_device == Device.CUDA
            and backend == "auto"
        ):
            cupy_backend = get_backend(backend="cupy", device="cuda")
            if not cupy_backend.is_available():
                raise RuntimeError(
                    "device='cuda' requires a working CuPy CUDA backend. "
                    "Use device='auto' to allow automatic backend selection."
                )
            return cupy_backend

        # Handle Device.TORCH explicitly - use torch backend
        if compute_device == Device.TORCH:
            try:
                import torch
            except Exception as exc:
                raise RuntimeError(
                    "device='torch' requires PyTorch with CUDA support."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='torch' requires torch.cuda.is_available() to be True. "
                    "Use device='auto' or device='cpu' if CUDA is unavailable."
                )
            return get_backend(backend="torch", device="cuda")

        return get_backend(backend=backend, device=device_str)
    
    def _to_array(self, X, device: Optional[Device] = None, backend: Optional[str] = None) -> Any:
        """
        Convert input to appropriate array type for device.

        Parameters
        ----------
        X : array-like
            Input data.
        device : Device, optional
            Target device. If None, uses self._get_compute_device().
        backend : {'numpy', 'cupy', 'torch'}, optional
            Explicit backend selection. If None, uses device-based default.

        Returns
        -------
        array
            NumPy array (CPU), CuPy array (GPU), or Torch tensor.
        """
        target_device = device or self._get_compute_device()

        # If backend is explicitly specified, use it
        if backend == "torch":
            torch_target = "cpu" if target_device == Device.CPU else "cuda"
            return self._to_torch(X, device=torch_target)
        elif backend == "cupy":
            return self._to_cupy(X)
        elif backend == "numpy":
            # Handle torch tensors that may be on CUDA — must move to CPU first
            if hasattr(X, 'cpu') and hasattr(X, 'numpy'):
                return X.detach().cpu().numpy()
            return np.asarray(X)

        # Otherwise, use device-based default
        if target_device == Device.TORCH:
            return self._to_torch(X, device="cuda")

        if target_device == Device.CUDA:
            # Strict CUDA means CuPy-backed arrays. Torch has its own explicit
            # device mode and is not used as an implicit CUDA fallback.
            try:
                import cupy as cp
                if isinstance(X, cp.ndarray):
                    return X
            except Exception:
                pass

        # Convert to numpy first for CPU paths or for non-CuPy inputs.
        if hasattr(X, "get"):  # CuPy-like array
            X_np = X.get()
        elif hasattr(X, "cpu"):  # PyTorch tensor
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)

        if target_device == Device.CUDA:
            try:
                import cupy as cp
                return cp.asarray(X_np)
            except Exception as exc:
                raise RuntimeError(
                    "device='cuda' requires a working CuPy CUDA backend; "
                    "no CPU/Torch fallback is performed for explicit CUDA."
                ) from exc

        return X_np

    def _to_torch(self, X, device=None):
        """Convert input to Torch tensor."""
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='torch' requires torch.cuda.is_available() to be True; "
                "no Torch CPU fallback is performed for explicit Torch GPU."
            )

        if isinstance(X, torch.Tensor):
            if device and X.device.type != device:
                return X.to(device)
            return X

        if hasattr(X, "get"):  # CuPy
            X_np = X.get()
        elif hasattr(X, "cpu"):  # Tensor-like
            X_cpu = X.cpu()
            X_np = X_cpu.numpy() if hasattr(X_cpu, "numpy") else np.asarray(X_cpu)
        else:
            X_np = np.asarray(X)

        target_device = device or _get_torch_device_str()
        return torch.from_numpy(X_np).to(target_device)

    def _to_cupy(self, X):
        """Convert input to CuPy array."""
        import cupy as cp

        if isinstance(X, cp.ndarray):
            return X

        if hasattr(X, "cpu"):  # PyTorch
            X_np = X.detach().cpu().numpy()
        elif hasattr(X, "get"):  # CuPy (shouldn't happen, but handle it)
            X_np = X.get()
        else:
            X_np = np.asarray(X)

        return cp.asarray(X_np)
    
    def _to_numpy(self, X) -> np.ndarray:
        """Convert array back to numpy."""
        if hasattr(X, 'get'):  # CuPy
            return X.get()
        elif hasattr(X, 'cpu'):  # PyTorch
            return X.cpu().numpy()
        return np.asarray(X)

    def adjust_pvalues(
        self,
        pvalues=None,
        method: str = "bh",
        alpha: float = 0.05,
        axis: Optional[int] = 0,
        backend: str = "auto",
    ):
        """
        Adjust p-values for multiple testing (FDR/FWER controls).

        Parameters
        ----------
        pvalues : array-like, optional
            Raw p-values. If omitted, uses this estimator's ``_pvalues``.
        method : str, default='bh'
            Adjustment method: ``bh``, ``by``, ``holm``, ``bonferroni``
            (aliases accepted).
        alpha : float, default=0.05
            Rejection threshold in (0, 1).
        axis : int or None, default=0
            Axis along which to adjust. ``None`` flattens all entries.
        backend : {'auto', 'numpy', 'cupy'}, default='auto'
            Compute backend. ``'auto'`` uses CuPy when estimator device is CUDA.

        Returns
        -------
        dict
            Contains ``pvalues``, ``pvalues_adjusted``, ``reject``,
            ``method``, ``alpha``, and ``axis``.
        """
        from statgpu.inference import adjust_pvalues as _adjust_pvalues

        source = pvalues
        if source is None:
            source = getattr(self, "_pvalues", None)
        if source is None:
            raise RuntimeError(
                "No p-values available. Fit with inference enabled or pass pvalues explicitly."
            )

        backend_name = str(backend).strip().lower()
        if backend_name == "auto" and self._get_compute_device() == Device.CUDA:
            backend_name = "cupy"

        if backend_name == "cupy":
            pvals = self._to_array(source, Device.CUDA)
        else:
            pvals = self._to_numpy(source)

        reject, pvals_adj = _adjust_pvalues(
            pvals,
            method=method,
            alpha=alpha,
            axis=axis,
            backend=backend_name,
        )

        return {
            "method": method,
            "alpha": float(alpha),
            "axis": axis,
            "backend": backend_name,
            "pvalues": pvals,
            "pvalues_adjusted": pvals_adj,
            "reject": reject,
        }

    def combine_pvalues(
        self,
        pvalues=None,
        method: str = "fisher",
        weights=None,
        axis: Optional[int] = None,
        backend: str = "auto",
    ):
        """
        Combine p-values into a global p-value.

        Parameters
        ----------
        pvalues : array-like, optional
            Raw p-values. If omitted, uses this estimator's ``_pvalues``.
        method : str, default='fisher'
            Combination method: ``fisher`` or ``cauchy`` (aliases accepted).
        weights : array-like, optional
            Optional non-negative weights for cauchy combination.
        axis : int or None, default=None
            Axis along which to combine p-values. ``None`` flattens input.
        backend : {'auto', 'numpy', 'cupy'}, default='auto'
            Compute backend. ``'auto'`` uses CuPy when estimator device is CUDA.

        Returns
        -------
        dict
            Contains ``pvalues``, ``statistic``, ``pvalue``,
            ``method``, ``axis``, and ``backend``.
        """
        from statgpu.inference import combine_pvalues as _combine_pvalues

        source = pvalues
        if source is None:
            source = getattr(self, "_pvalues", None)
        if source is None:
            raise RuntimeError(
                "No p-values available. Fit with inference enabled or pass pvalues explicitly."
            )

        backend_name = str(backend).strip().lower()
        if backend_name == "auto" and self._get_compute_device() == Device.CUDA:
            backend_name = "cupy"

        if backend_name == "cupy":
            pvals = self._to_array(source, Device.CUDA)
            w_cast = None if weights is None else self._to_array(weights, Device.CUDA)
        elif backend_name == "numpy":
            pvals = self._to_numpy(source)
            w_cast = None if weights is None else self._to_numpy(weights)
        else:
            pvals = source
            w_cast = weights

        statistic, pvalue = _combine_pvalues(
            pvals,
            method=method,
            weights=w_cast,
            axis=axis,
            backend=backend_name,
        )

        return {
            "method": method,
            "axis": axis,
            "backend": backend_name,
            "pvalues": pvals,
            "weights": w_cast,
            "statistic": statistic,
            "pvalue": pvalue,
        }

    def bootstrap_statistic(
        self,
        statistic,
        *arrays,
        n_resamples: int = 200,
        strategy: str = "iid",
        strata=None,
        clusters=None,
        block_size: Optional[int] = None,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        statistic_name: str = "statistic",
        backend: str = "auto",
    ):
        """
        Run unified bootstrap engine from model context.

        This is a thin wrapper over ``statgpu.inference.bootstrap_statistic``.
        """
        from statgpu.inference import bootstrap_statistic as _bootstrap_statistic

        arrays_use = arrays
        if len(arrays_use) == 0:
            X_cache = getattr(self, "_X_design", None)
            y_cache = getattr(self, "_y", None)
            if X_cache is None or y_cache is None:
                raise RuntimeError(
                    "No cached training arrays available. Pass arrays explicitly or fit first."
                )
            arrays_use = (X_cache, y_cache)

        backend_name = str(backend).strip().lower()
        if backend_name == "auto" and self._get_compute_device() == Device.CUDA:
            backend_name = "cupy"

        if backend_name == "cupy":
            arrays_cast = tuple(self._to_array(a, Device.CUDA) for a in arrays_use)
            strata_cast = None if strata is None else self._to_array(strata, Device.CUDA)
            clusters_cast = None if clusters is None else self._to_array(clusters, Device.CUDA)
        elif backend_name == "numpy":
            arrays_cast = tuple(self._to_numpy(a) for a in arrays_use)
            strata_cast = None if strata is None else self._to_numpy(strata)
            clusters_cast = None if clusters is None else self._to_numpy(clusters)
        else:
            arrays_cast = arrays_use
            strata_cast = strata
            clusters_cast = clusters

        return _bootstrap_statistic(
            statistic,
            *arrays_cast,
            n_resamples=n_resamples,
            strategy=strategy,
            strata=strata_cast,
            clusters=clusters_cast,
            block_size=block_size,
            confidence_level=confidence_level,
            random_state=random_state,
            statistic_name=statistic_name,
            backend=backend_name,
        )

    def permutation_test(
        self,
        statistic,
        X,
        y,
        n_resamples: int = 1000,
        strategy: str = "iid",
        strata=None,
        groups=None,
        alternative: str = "two-sided",
        random_state: Optional[int] = None,
        statistic_name: str = "statistic",
        backend: str = "auto",
    ):
        """
        Run unified permutation test engine from model context.

        This is a thin wrapper over ``statgpu.inference.permutation_test``.
        """
        from statgpu.inference import permutation_test as _permutation_test

        backend_name = str(backend).strip().lower()
        if backend_name == "auto" and self._get_compute_device() == Device.CUDA:
            backend_name = "cupy"

        if backend_name == "cupy":
            X_cast = self._to_array(X, Device.CUDA)
            y_cast = self._to_array(y, Device.CUDA)
            strata_cast = None if strata is None else self._to_array(strata, Device.CUDA)
            groups_cast = None if groups is None else self._to_array(groups, Device.CUDA)
        elif backend_name == "numpy":
            X_cast = self._to_numpy(X)
            y_cast = self._to_numpy(y)
            strata_cast = None if strata is None else self._to_numpy(strata)
            groups_cast = None if groups is None else self._to_numpy(groups)
        else:
            X_cast = X
            y_cast = y
            strata_cast = strata
            groups_cast = groups

        return _permutation_test(
            statistic,
            X_cast,
            y_cast,
            n_resamples=n_resamples,
            strategy=strategy,
            strata=strata_cast,
            groups=groups_cast,
            alternative=alternative,
            random_state=random_state,
            statistic_name=statistic_name,
            backend=backend_name,
        )
    
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the estimator."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def _check_is_fitted(self):
        """Check if estimator has been fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'device': self.device.value,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key == 'device':
                self.device = Device(value) if isinstance(value, str) else value
            else:
                setattr(self, key, value)
        return self
