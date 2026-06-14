"""Ledoit-Wolf and Oracle Approximating Shrinkage (OAS) covariance estimators."""

from __future__ import annotations

__all__ = ["LedoitWolf", "OAS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_float_scalar, _torch_dev, xp_zeros, xp_eye

from statgpu.covariance._empirical import EmpiricalCovariance, _detect_backend, _stable_inv


class LedoitWolf(EmpiricalCovariance):
    """
    Ledoit-Wolf shrinkage covariance estimator with GPU support.

    Computes a shrinkage estimator that is a convex combination of the
    sample covariance and a structured target (scaled identity). The
    optimal shrinkage intensity is derived from the Ledoit & Wolf (2004)
    analytical formula.

    Parameters
    ----------
    assume_centered : bool, default=False
        If True, data is assumed to be already centered.
    device : str or Device, default='auto'
        Computation device: ``'cpu'``, ``'cuda'``, ``'torch'``, or ``'auto'``.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated shrunk covariance matrix.
    location_ : ndarray of shape (n_features,)
        Estimated mean (zero if *assume_centered* is True).
    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse of the covariance (precision matrix).
    shrinkage_ : float
        Shrinkage intensity in [0, 1].
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.
    """

    def fit(self, X, y=None):
        """Fit the Ledoit-Wolf covariance model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)

        # Ensure torch tensors land on CUDA
        _ref = None
        if backend_name == "torch":
            import torch
            _ref = torch.empty(0, dtype=torch.float64, device="cuda")
        if _ref is not None:
            X_arr = xp.asarray(X, dtype=xp.float64, device=_ref.device)
        else:
            X_arr = xp.asarray(X, dtype=xp.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n = int(X_arr.shape[0])
        p = int(X_arr.shape[1])

        if n < 2:
            raise ValueError(
                f"Need at least 2 samples to estimate covariance, got {n}"
            )

        # Center
        if self.assume_centered:
            location = xp_zeros(p, xp.float64, xp, X_arr)
        else:
            location = xp.mean(X_arr, axis=0)
            X_arr = X_arr - location

        # Sample covariance
        S = (X_arr.T @ X_arr) / float(n)

        # ---- Ledoit-Wolf shrinkage intensity ----
        mu = _to_float_scalar(xp.trace(S)) / p

        # Efficient formula (LW2004):
        #   beta = (1/n^2) * [sum_k ||X_k||_2^4  -  n * ||S||_F^2]
        #   delta = ||S - mu*I||_F^2 = ||S||_F^2 - tr(S)^2 / p
        #   alpha = clip(beta / delta, 0, 1)
        X_sq = X_arr * X_arr
        norm_sq = xp.sum(X_sq, axis=1)  # ||X_k||^2 for each observation k
        sum_norm_sq_sq = _to_float_scalar(xp.sum(norm_sq * norm_sq))

        frob_S_sq = _to_float_scalar(xp.sum(S * S))
        tr_S = _to_float_scalar(xp.trace(S))

        beta = (sum_norm_sq_sq - float(n) * frob_S_sq) / (float(n) * float(n))
        delta = frob_S_sq - tr_S * tr_S / float(p)

        if delta <= 0.0:
            # Degenerate case: all eigenvalues equal
            alpha = 1.0
        else:
            alpha = beta / delta
            alpha = max(0.0, min(1.0, alpha))

        # Shrunk covariance: (1 - alpha) * S + alpha * mu * I
        shrunk_S = (1.0 - alpha) * S + alpha * mu * xp_eye(p, xp.float64, xp, S)

        # Precision of shrunk covariance
        precision = _stable_inv(shrunk_S, xp, backend_name)

        self.covariance_ = shrunk_S
        self.location_ = location
        self.precision_ = precision
        self.shrinkage_ = alpha
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self


class OAS(EmpiricalCovariance):
    """
    Oracle Approximating Shrinkage (OAS) covariance estimator with GPU support.

    Uses the analytical formula from Chen, Wiesel, Eldar & Hero (2010)
    to compute the optimal shrinkage intensity under a Gaussian assumption.

    Parameters
    ----------
    assume_centered : bool, default=False
        If True, data is assumed to be already centered.
    device : str or Device, default='auto'
        Computation device: ``'cpu'``, ``'cuda'``, ``'torch'``, or ``'auto'``.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated shrunk covariance matrix.
    location_ : ndarray of shape (n_features,)
        Estimated mean (zero if *assume_centered* is True).
    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo-inverse of the covariance (precision matrix).
    shrinkage_ : float
        Shrinkage intensity in [0, 1].
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.
    """

    def fit(self, X, y=None):
        """Fit the OAS covariance model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)

        # Ensure torch tensors land on CUDA
        _ref = None
        if backend_name == "torch":
            import torch
            _ref = torch.empty(0, dtype=torch.float64, device="cuda")
        if _ref is not None:
            X_arr = xp.asarray(X, dtype=xp.float64, device=_ref.device)
        else:
            X_arr = xp.asarray(X, dtype=xp.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n = int(X_arr.shape[0])
        p = int(X_arr.shape[1])

        if n < 2:
            raise ValueError(
                f"Need at least 2 samples to estimate covariance, got {n}"
            )

        # Center
        if self.assume_centered:
            location = xp_zeros(p, xp.float64, xp, X_arr)
        else:
            location = xp.mean(X_arr, axis=0)
            X_arr = X_arr - location

        # Sample covariance
        S = (X_arr.T @ X_arr) / float(n)

        # ---- OAS shrinkage intensity ----
        # Follows sklearn's implementation of the OAS formula (Chen et al. 2010).
        # Note: sklearn omits the 2/p factor from Eq. 23 in the original paper
        # because it negligibly affects the estimator for large p.
        tr_S = _to_float_scalar(xp.trace(S))
        alpha_mean = _to_float_scalar(xp.mean(S * S))  # mean of squared elements
        mu = tr_S / float(p)
        mu_squared = mu * mu

        numerator = alpha_mean + mu_squared
        denominator = (float(n) + 1.0) * (alpha_mean - mu_squared / float(p))

        if denominator <= 0.0:
            alpha = 1.0
        else:
            alpha = numerator / denominator
            alpha = max(0.0, min(1.0, alpha))

        # Shrunk covariance: (1 - alpha) * S + alpha * mu * I
        shrunk_S = (1.0 - alpha) * S + alpha * mu * xp_eye(p, xp.float64, xp, S)

        # Precision of shrunk covariance
        precision = _stable_inv(shrunk_S, xp, backend_name)

        self.covariance_ = shrunk_S
        self.location_ = location
        self.precision_ = precision
        self.shrinkage_ = alpha
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self
