"""Nystroem kernel approximation with GPU acceleration."""

from __future__ import annotations

__all__ = ["Nystroem"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray
from statgpu.nonparametric.kernel_methods._kernels import pairwise_kernels


class Nystroem(BaseEstimator):
    """Nystroem kernel approximation.

    Approximates a kernel feature map using a random subset of the
    training data as landmarks.  Useful for scaling kernel methods
    to large datasets.

    The approximation is Z = K_nm @ normalization_, with
    normalization_ = U @ diag(1/sqrt(max(s, 1e-12))) @ Vt from the
    NumPy SVD K_mm = U @ diag(s) @ Vt. For a positive-semidefinite kernel,
    this is the symmetric inverse-square-root feature orientation.

    Landmark kernel construction and SVD run on CPU even for GPU fitting;
    query-to-landmark kernels and returned features use the selected array
    library, but Torch CPU inputs are not consistently moved to CUDA.
    This output-placement limitation is separate from the CPU landmark SVD.
    A custom kernel must also accept NumPy landmark inputs.

    Finite input does not guarantee a finite computed kernel. Polynomial
    kernels can overflow; nonfinite kernel/decomposition values can currently
    produce a fitted object with NaN learned arrays and predictions. Use finite
    applicable kernel controls, check learned arrays and outputs for finiteness,
    and discard a nonfinite fit. Increasing regularization does not repair an
    already nonfinite kernel; revise feature scaling/kernel settings and refit.

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel function name or callable.
    n_components : int, default=100
        Requested landmark count, capped at the number of training rows:
        m = min(n_components, n_samples).
    gamma : float, optional
        Kernel coefficient (for rbf, poly, etc.).
    degree : int, default=3
        Polynomial degree (for poly kernel).
    coef0 : float, default=1
        Independent term (for poly and sigmoid kernels).
    random_state : int or None, default=None
        Random seed for landmark selection.
    device : str or Device, default='auto'
        Requested device: 'cpu' (NumPy), 'cuda' (CuPy CUDA), 'torch'
        (Torch CUDA), or 'auto'. Unavailable explicit backends raise.
        After the Torch CUDA availability check succeeds, NumPy or Torch
        CPU input can still stay on CPU. Inspect returned features with
        .device/.is_cuda; the selected backend does not prove placement.
        Public fitted arrays are deliberately NumPy, so inspect outputs
        instead. For a predictable CPU path use NumPy input, device='cpu'.
    n_jobs : int or None, default=None
        Shared estimator option; does not parallelize fitting.

    Attributes
    ----------
    components_ : numpy.ndarray, shape (m, n_features)
        Host selected landmarks, where m = min(n_components, n_samples).
    component_indices_ : numpy.ndarray, shape (m,)
        Host indices of selected landmarks in the training data.
    normalization_ : numpy.ndarray, shape (m, m)
        Host NumPy SVD inverse-square-root normalization matrix.
    eigenvalues_ : numpy.ndarray, shape (m,)
        Host NumPy singular values, floored at 1e-12; not signed eigenvalues.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        n_components: int = 100,
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.kernel = kernel
        self.n_components = n_components
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the Nystroem approximation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
            raise ValueError("X must contain only finite values")
        if isinstance(self.n_components, bool) or int(self.n_components) < 1:
            raise ValueError("n_components must be a positive integer")

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])
        self.n_features_in_ = n_features

        n_comp = min(self.n_components, n_samples)

        # Select landmarks (do NOT sort — must match sklearn's order for reproducibility)
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_samples, size=n_comp, replace=False)
        self.component_indices_ = indices

        landmarks = X_arr[indices]  # (n_comp, n_features)
        self.components_ = _to_numpy(landmarks)

        # Build kernel params
        kernel_params = self._get_kernel_params()

        # ---- K_mm decomposition on CPU (small matrix, avoid GPU overhead) ----
        # Use eigendecomposition for PSD kernel matrix: K = V diag(λ) V^T
        # K^{-1/2} = V diag(1/sqrt(λ)) V^T
        landmarks_np = _to_numpy(landmarks)
        K_mm_np = pairwise_kernels(landmarks_np, metric=self.kernel, xp=np, **kernel_params)

        # SVD is stable for both PSD and indefinite kernels (for example,
        # sigmoid).  Clipping negative eigenvalues from eigh would otherwise
        # create enormous artificial features.
        U, singular_values, Vt = np.linalg.svd(K_mm_np, full_matrices=False)
        singular_values = np.maximum(singular_values, 1e-12)
        self.normalization_ = (U / np.sqrt(singular_values)[None, :]) @ Vt
        self.eigenvalues_ = singular_values
        self._landmarks = landmarks
        self._landmarks_np = landmarks_np
        self._kernel_params = kernel_params
        self._backend_name = backend.name
        self._fitted = True
        return self

    def transform(self, X):
        """Approximate feature map for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_transformed : backend array, shape (n_samples, m)
            Approximate features, where m = min(n_components, n_training_samples).
            The selected backend does not guarantee placement: device='torch'
            rejects unavailable CUDA, but after that check CPU input can still
            yield CPU tensors. Inspect output .device/.is_cuda. Small singular
            values are floored, not removed; fitted public arrays remain NumPy
            regardless of numerical device.
        """
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
            raise ValueError("X must contain only finite values")

        # Compute K_nm on the same device as X
        if xp is np:
            # CPU path: use numpy landmarks
            landmarks = self._landmarks_np
        else:
            # GPU path: use GPU landmarks
            landmarks = self._landmarks
            if hasattr(X_arr, 'device') and getattr(xp, '__name__', '') == 'torch':
                # torch: ensure landmarks on same device
                landmarks = xp.asarray(_to_numpy(landmarks), dtype=xp.float64,
                                       device=X_arr.device)

        K_nm = pairwise_kernels(X_arr, landmarks, metric=self.kernel,
                                xp=xp, **self._kernel_params)

        # Apply normalization
        norm = xp.asarray(self.normalization_, dtype=xp.float64)
        if hasattr(K_nm, 'is_cuda'):
            norm = norm.to(device=K_nm.device)

        Z = K_nm @ norm

        # GPU in → GPU out, CPU in → CPU out
        return Z

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """Alias for transform (required by BaseEstimator)."""
        return self.transform(X)

    def _get_kernel_params(self):
        """Collect kernel parameters."""
        params = {}
        if self.gamma is not None:
            params["gamma"] = self.gamma
        if self.kernel in ("poly", "polynomial"):
            params["degree"] = self.degree
            params["coef0"] = self.coef0
        elif self.kernel == "sigmoid":
            params["coef0"] = self.coef0
        return params

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["kernel"] = self.kernel
        params["n_components"] = self.n_components
        params["gamma"] = self.gamma
        params["degree"] = self.degree
        params["coef0"] = self.coef0
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        for key in ["kernel", "n_components", "gamma", "degree", "coef0", "random_state"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
