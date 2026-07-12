"""Empirical covariance estimation with GPU support."""

from __future__ import annotations

__all__ = ["EmpiricalCovariance"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import (
    _LINALG_ERRORS,
    _get_xp,
    _is_cupy_array,
    _is_torch_array,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
    _torch_dev,
    xp_zeros,
    xp_asarray,
)


def _detect_backend(X, device: Device) -> str:
    """Resolve backend from input array type, falling back to device setting."""
    if _is_torch_array(X):
        return "torch"
    if _is_cupy_array(X):
        return "cupy"
    # For numpy input, use device-based resolution
    if device == Device.TORCH:
        return "torch"
    if device == Device.CUDA:
        try:
            import cupy as cp  # noqa: F401
            return "cupy"
        except ImportError:
            raise RuntimeError(
                "CuPy is required for device='cuda' but is not installed. "
                "Use device='auto' to fall back to CPU automatically."
            )
    return "numpy"


def _torch_device_from_data(X) -> Optional[str]:
    """Extract torch device from tensor, or None for non-torch inputs."""
    try:
        import torch
        if isinstance(X, torch.Tensor):
            return str(X.device)
    except (ImportError, AttributeError):
        pass
    return None


class EmpiricalCovariance(BaseEstimator):
    """
    Maximum likelihood covariance estimator with GPU acceleration.

    Computes the sample covariance matrix, its inverse (precision), and
    provides log-likelihood scoring and Mahalanobis distance computation.

    Parameters
    ----------
    assume_centered : bool, default=False
        If True, data is assumed to be already centered. If False, the
        mean is estimated and subtracted before computing the covariance.
    device : str or Device, default='auto'
        Computation device: ``'cpu'``, ``'cuda'``, ``'torch'``, or ``'auto'``.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : array, shape (n_features, n_features)
        Estimated covariance matrix.
    location_ : array, shape (n_features,)
        Estimated location (mean) vector.
    precision_ : array, shape (n_features, n_features)
        Estimated precision matrix (inverse covariance).
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.assume_centered = assume_centered

    def fit(self, X, y=None):
        """Fit the covariance model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        self
        """
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)

        # For torch backend, ensure arrays land on CUDA (not CPU)
        _ref = None
        if backend_name == "torch":
            import torch
            _dev = self._get_compute_device()
            _cuda_dev = "cuda" if _dev.value in ("torch", "cuda") else "cpu"
            _ref = torch.empty(0, dtype=torch.float64, device=_cuda_dev)

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=_ref)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])

        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 samples to estimate covariance, got {n_samples}"
            )

        # Center if needed
        if self.assume_centered:
            location = xp_zeros(n_features, xp.float64, xp, X_arr)
        else:
            location = xp.mean(X_arr, axis=0)
            X_arr = X_arr - location

        # Sample covariance: S = X^T X / n
        S = (X_arr.T @ X_arr) / float(n_samples)

        # Compute precision (inverse) with jitter stabilization
        precision = _stable_inv(S, xp, backend_name)

        self.covariance_ = S
        self.location_ = location
        self.precision_ = precision
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self._backend_name = backend_name
        self._fitted = True
        return self

    def predict(self, X):
        """Return Mahalanobis distances for *X* under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        distances : ndarray of shape (n_samples,)
        """
        return self.mahalanobis(X)

    def score(self, X, y=None):
        """Compute the average log-likelihood of *X* under the fitted Gaussian.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : ignored

        Returns
        -------
        ll : float
            Average log-likelihood per observation.
        """
        self._check_is_fitted()
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = int(X_arr.shape[0])
        p = int(X_arr.shape[1])
        if p != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features, got {p}")
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")

        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        prec = xp_asarray(self.precision_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        cov = xp_asarray(self.covariance_, dtype=xp.float64, xp=xp, ref_arr=X_arr)

        X_centered = X_arr - loc

        # Mahalanobis term: sum of (x-mu)^T S^{-1} (x-mu)
        M = X_centered @ prec
        mahal_sum = _to_float_scalar(xp.sum(M * X_centered))

        # log(det(S)) via slogdet for numerical stability
        sign, logdet = xp.linalg.slogdet(cov)
        sign_val = _to_float_scalar(sign)
        if sign_val <= 0:
            return float("-inf")
        logdet_val = _to_float_scalar(logdet)

        # Average log-likelihood:
        #   LL = -(1/2) * (p * log(2*pi) + log(det(S)) + (1/n) * sum(mahal))
        ll = -0.5 * (p * np.log(2.0 * np.pi) + logdet_val + mahal_sum / n_samples)
        return float(ll)

    def mahalanobis(self, X):
        """Compute Mahalanobis distances of observations in *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Squared Mahalanobis distances.
        """
        self._check_is_fitted()
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_:
            got = X_arr.shape[1] if X_arr.ndim == 2 else "invalid"
            raise ValueError(f"X must have {self.n_features_} features, got {got}")

        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        prec = xp_asarray(self.precision_, dtype=xp.float64, xp=xp, ref_arr=X_arr)

        X_centered = X_arr - loc

        # Efficient: row-wise (x-mu)^T prec (x-mu)
        M = X_centered @ prec
        mahal = xp.sum(M * X_centered, axis=1)

        return _to_numpy(mahal)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["assume_centered"] = self.assume_centered
        return params

    def set_params(self, **params):
        for key, value in list(params.items()):
            if key == "assume_centered":
                self.assume_centered = value
                del params[key]
        if params:
            super().set_params(**params)
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_inv(S, xp, backend_name: str):
    """Invert *S* with jitter-boosted diagonal for numerical stability.

    Tries the exact inverse first; if that fails or produces non-finite
    values, adds progressively larger diagonal jitter.
    """
    p = int(S.shape[0])

    trace_S = _to_float_scalar(xp.trace(S))
    base = max(abs(trace_S) / max(p, 1), 1.0) * 1e-10

    torch_dev = None
    if backend_name == "torch":
        try:
            import torch
            if isinstance(S, torch.Tensor):
                torch_dev = S.device
        except (ImportError, AttributeError):
            pass

    # Pre-allocate identity matrix once
    if torch_dev is not None:
        eye = xp.eye(p, dtype=xp.float64, device=torch_dev)
    else:
        eye = xp.eye(p, dtype=xp.float64)

    # Preserve the exact estimator whenever the covariance is invertible.
    # Jitter is a fallback, not part of the empirical covariance definition.
    try:
        inv_S = xp.linalg.inv(S)
        test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
        if np.isfinite(test_val):
            return inv_S
    except _LINALG_ERRORS + (ValueError,):
        pass

    jitter = base
    for _ in range(12):
        try:
            inv_S = xp.linalg.inv(S + jitter * eye)
            test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
            if np.isfinite(test_val):
                return inv_S
        except _LINALG_ERRORS + (ValueError,):
            pass
        jitter *= 10.0

    raise ValueError(
        "Covariance matrix is singular and cannot be inverted even with "
        "diagonal jitter. Consider using LedoitWolf or OAS shrinkage."
    )
