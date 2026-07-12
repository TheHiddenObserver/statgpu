"""
Base classes for statgpu estimators.
"""

from __future__ import annotations

__all__ = ["BaseEstimator"]

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

from statgpu._config import Device, get_device
from statgpu.backends import (
    get_backend,
    BackendBase,
    _get_torch_device_str,
    _cupy_to_torch_dlpack,
    _torch_to_cupy_dlpack,
    _numpy_to_torch_tensor,
    _move_torch_tensor,
)


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
            if hasattr(X, "get"):
                return X.get()
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
            X_cpu = X.detach().cpu() if hasattr(X, "detach") else X.cpu()
            X_np = X_cpu.numpy() if hasattr(X_cpu, "numpy") else np.asarray(X_cpu)
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
            return _move_torch_tensor(X, device=device) if device else X

        if hasattr(X, "get"):  # CuPy
            tensor = _cupy_to_torch_dlpack(X, device=device)
            if tensor is not None:
                return tensor
            X_np = X.get()
        elif hasattr(X, "cpu"):  # Tensor-like
            X_cpu = X.detach().cpu() if hasattr(X, "detach") else X.cpu()
            X_np = X_cpu.numpy() if hasattr(X_cpu, "numpy") else np.asarray(X_cpu)
        else:
            target_device = device or _get_torch_device_str()
            return _numpy_to_torch_tensor(
                X,
                device=target_device,
                pin_memory=str(target_device).startswith("cuda"),
            )

        target_device = device or _get_torch_device_str()
        return _numpy_to_torch_tensor(
            X_np,
            device=target_device,
            pin_memory=str(target_device).startswith("cuda"),
        )

    def _to_cupy(self, X):
        """Convert input to CuPy array."""
        import cupy as cp

        if isinstance(X, cp.ndarray):
            return X

        if hasattr(X, "cpu"):  # PyTorch
            arr = _torch_to_cupy_dlpack(X)
            if arr is not None:
                return arr
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
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _resolve_inference_backend(self, backend: str) -> str:
        """Resolve model-context inference backend from the estimator device."""
        backend_name = str(backend).strip().lower()
        if backend_name == "auto":
            compute_device = self._get_compute_device()
            if compute_device == Device.CUDA:
                return "cupy"
            if compute_device == Device.TORCH:
                return "torch"
        return backend_name

    def _cast_inference_array(self, value, backend_name: str):
        """Cast an inference input to the explicitly resolved backend."""
        if backend_name == "cupy":
            return self._to_array(value, Device.CUDA, backend="cupy")
        if backend_name == "torch":
            return self._to_array(value, Device.TORCH, backend="torch")
        if backend_name == "numpy":
            return self._to_numpy(value)
        return value

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
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Compute backend. ``'auto'`` follows the estimator's resolved device.

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

        backend_name = self._resolve_inference_backend(backend)

        pvals = self._cast_inference_array(source, backend_name)

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
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Compute backend. ``'auto'`` follows the estimator's resolved device.

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

        backend_name = self._resolve_inference_backend(backend)

        pvals = self._cast_inference_array(source, backend_name)
        w_cast = (
            None
            if weights is None
            else self._cast_inference_array(weights, backend_name)
        )

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

        backend_name = self._resolve_inference_backend(backend)

        arrays_cast = tuple(
            self._cast_inference_array(a, backend_name) for a in arrays_use
        )
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        clusters_cast = (
            None
            if clusters is None
            else self._cast_inference_array(clusters, backend_name)
        )

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

        backend_name = self._resolve_inference_backend(backend)

        X_cast = self._cast_inference_array(X, backend_name)
        y_cast = self._cast_inference_array(y, backend_name)
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        groups_cast = (
            None
            if groups is None
            else self._cast_inference_array(groups, backend_name)
        )

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
    
    def __sklearn_clone__(self):
        """Return an unfitted estimator clone for scikit-learn >= 1.3.

        Several statgpu constructors validate or canonicalize immutable strings
        and copy mutable dictionaries. scikit-learn's legacy identity check
        treats those defensive copies as constructor mutation. The explicit
        clone protocol preserves the public constructor values while discarding
        fitted state.
        """
        from copy import deepcopy

        return type(self)(**deepcopy(self.get_params(deep=False)))

    def get_params(self, deep=True):
        """Get constructor parameters for this estimator.

        Nested estimator parameters are exposed as ``name__param`` when
        ``deep=True``, matching the scikit-learn estimator contract.
        """
        import inspect

        params = {}
        try:
            sig = inspect.signature(type(self).__init__)
        except (ValueError, TypeError):
            return params

        for name, parameter in sig.parameters.items():
            if name == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                params[name] = getattr(self, f"_{name}")

        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params"):
                    for sub_name, sub_value in value.get_params(deep=True).items():
                        params[f"{name}__{sub_name}"] = sub_value
        return params


    def set_params(self, **params):
        """Set estimator parameters, validating names and nesting."""
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = {}

        for key, value in params.items():
            root, delimiter, sub_key = key.partition("__")
            if root not in valid_params:
                valid_names = sorted(name for name in valid_params if "__" not in name)
                raise ValueError(
                    f"Invalid parameter {root!r} for estimator "
                    f"{self.__class__.__name__}. Valid parameters are: "
                    f"{', '.join(valid_names)}."
                )

            if delimiter:
                nested_params.setdefault(root, {})[sub_key] = value
                continue

            if root == "device" and isinstance(value, str):
                value = Device(value)
            if hasattr(self, root):
                setattr(self, root, value)
            else:
                setattr(self, f"_{root}", value)

        for root, sub_params in nested_params.items():
            nested_estimator = getattr(self, root, None)
            if nested_estimator is None:
                nested_estimator = getattr(self, f"_{root}", None)
            if not hasattr(nested_estimator, "set_params"):
                raise ValueError(
                    f"Parameter {root!r} of {self.__class__.__name__} "
                    "does not support nested parameters."
                )
            nested_estimator.set_params(**sub_params)

        return self
