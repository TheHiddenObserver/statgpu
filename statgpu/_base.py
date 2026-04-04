"""
Base classes for statgpu estimators.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

from ._config import Device, get_device, cuda_available
from .backends import get_backend, BackendBase


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
            return Device.CUDA if cuda_available() else Device.CPU
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
        device_str = self._get_compute_device().value  # 'cpu' or 'cuda'
        return get_backend(backend=backend, device=device_str)
    
    def _to_array(self, X, device: Optional[Device] = None) -> Any:
        """
        Convert input to appropriate array type for device.
        
        Parameters
        ----------
        X : array-like
            Input data.
        device : Device, optional
            Target device. If None, uses self._get_compute_device().
        
        Returns
        -------
        array
            NumPy array (CPU) or CuPy array (GPU).
        """
        target_device = device or self._get_compute_device()
        
        # If target is CUDA and X is already a CuPy array, keep it on GPU.
        # This avoids an expensive host-device roundtrip when the user
        # pre-constructs data on GPU (torch-like workflow).
        if target_device == Device.CUDA:
            try:
                import cupy as cp

                if isinstance(X, cp.ndarray):
                    return X
            except Exception:
                # If CuPy isn't available / type check fails, fall back below.
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
            except ImportError:
                # Fall back to numpy if CuPy isn't available.
                return X_np

        return X_np
    
    def _to_numpy(self, X) -> np.ndarray:
        """Convert array back to numpy."""
        if hasattr(X, 'get'):  # CuPy
            return X.get()
        elif hasattr(X, 'cpu'):  # PyTorch
            return X.cpu().numpy()
        return np.asarray(X)
    
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
