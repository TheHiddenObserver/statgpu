"""Nystroem kernel approximation with GPU acceleration."""

from __future__ import annotations

__all__ = ["Nystroem"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray, xp_maximum
from statgpu.nonparametric.kernel_methods._kernels import pairwise_kernels


class Nystroem(BaseEstimator):
    """Nystroem kernel approximation.

    Approximates a kernel feature map using a random subset of the
    training data as landmarks.  Useful for scaling kernel methods
    to large datasets.

    The approximation is Z = K_nm @ normalization_, with
    normalization_ = U @ diag(1/sqrt(max(s, 1e-12))) @ Vt from the
    selected-backend SVD K_mm = U @ diag(s) @ Vt. For a positive-semidefinite kernel,
    this is the symmetric inverse-square-root feature orientation.

    Landmark and query kernels and the landmark SVD use the selected backend.
    Seeded landmark-index selection remains on the host, and public fitted
    arrays are exported to NumPy for reporting. A custom kernel must accept
    arrays from the selected backend.

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
        Inputs are placed on the selected backend device. Public fitted arrays
        are deliberately NumPy; transformed features use the selected device.
    n_jobs : int or None, default=None
        Shared estimator option; does not parallelize fitting.

    Attributes
    ----------
    components_ : numpy.ndarray, shape (m, n_features)
        Host selected landmarks, where m = min(n_components, n_samples).
    component_indices_ : numpy.ndarray, shape (m,)
        Host indices of selected landmarks in the training data.
    normalization_ : numpy.ndarray, shape (m, m)
        Host inverse-square-root normalization matrix from the native SVD.
    eigenvalues_ : numpy.ndarray, shape (m,)
        Host singular values, floored at 1e-12; not signed eigenvalues.
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

        X_arr = backend.asarray(X, dtype=backend.float64)
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

        # Keep the landmark Gram and its decomposition on the numerical backend.
        # SVD preserves the established normalization for indefinite kernels too.
        K_mm = pairwise_kernels(landmarks, metric=self.kernel, xp=xp, **kernel_params)
        U, singular_values, Vt = xp.linalg.svd(K_mm, full_matrices=False)
        singular_values = xp_maximum(singular_values, 1e-12, xp)
        normalization = (U / xp.sqrt(singular_values)[None, :]) @ Vt
        self.normalization_ = _to_numpy(normalization)
        self.eigenvalues_ = _to_numpy(singular_values)
        self._landmarks = landmarks
        self._landmarks_np = self.components_
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
            Inputs and features use the selected backend device. Small singular
            values are floored, not removed; fitted public arrays remain NumPy
            regardless of numerical device.
        """
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = backend.asarray(X, dtype=backend.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
            raise ValueError("X must contain only finite values")

        # Re-align retained landmarks when the requested backend/device changes.
        landmarks = backend.asarray(self._landmarks, dtype=backend.float64)

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
