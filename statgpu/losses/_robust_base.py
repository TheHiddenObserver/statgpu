"""
Base class for robust losses with scale estimation (Huber, Bisquare, Fair).

Provides shared methods: scale estimation (MAD, Huber Proposal 2),
cached Lipschitz constant, and solver wrappers that call _ensure_scale
before delegating to the base class or subclass implementations.

Subclasses must implement:
    - per_sample_value(eta, y)
    - per_sample_gradient(eta, y)
    - _fused_impl(X, y, coef, sample_weight=None)
    - _hessian_impl(X, y, coef, sample_weight=None)
"""

__all__ = ["RobustLossBase"]

import numpy as np
from statgpu.backends._utils import _to_numpy as _backends_to_numpy
from ._base import LossBase


class RobustLossBase(LossBase):
    """Base for robust losses with automatic scale estimation.

    Subclasses (HuberLoss, BisquareLoss, FairLoss) share:
    - ``estimate_scale()``: MAD-based scale from residuals
    - ``_update_scale_prop2()``: Huber Proposal 2 fixed-point iteration
    - ``_ensure_scale()``: lazy scale trigger on first gradient call
    - ``lipschitz()``: cached max eigenvalue of X'X / n
    - ``value()``, ``gradient()``, ``fused_value_and_gradient()``, ``hessian()``:
      ensure scale is estimated before computing loss/gradient/hessian
    """

    # ── Array conversion (backend-agnostic) ────────────────────────────

    @staticmethod
    def _to_numpy(arr):
        """Convert cupy/torch/numpy array to numpy.

        Delegates to statgpu.backends._utils._to_numpy which handles
        .detach() for torch tensors with autograd graphs.
        """
        return _backends_to_numpy(arr)

    # ── Scale estimation ───────────────────────────────────────────────

    def estimate_scale(self, X, y, coef=None):
        """Estimate scale from residuals via MAD: scale = MAD(y - X@coef) / 0.6745."""
        y_np = self._to_numpy(y)
        X_np = self._to_numpy(X)
        if coef is not None:
            coef_np = self._to_numpy(coef)
        else:
            coef_np = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
        r = y_np - X_np @ coef_np
        mad = float(np.median(np.abs(r)))
        scale = max(mad / 0.6745, 1e-10)
        self.delta = self.epsilon * scale
        self._scale_estimated = True
        return scale

    def _update_scale_prop2(self, X, y, coef):
        """Huber Proposal 2: re-estimate sigma via fixed-point iteration.

        Standard M-estimator scale (Huber 1981):
            sigma^2 = (1/n) * [sum_{inliers} r_i^2 + sigma^2 * eps^2 * n_outliers]

        Freezes when sigma changes < 1% between calls.
        """
        y_np = self._to_numpy(y)
        X_np = self._to_numpy(X)
        coef_np = self._to_numpy(coef)
        r = y_np - X_np @ coef_np
        n = len(r)
        eps = self.epsilon
        eps2 = eps * eps

        # Initialize sigma from MAD of residuals
        mad = float(np.median(np.abs(r)))
        sigma = max(mad / 0.6745, 1e-10)

        for _ in range(50):
            abs_r = np.abs(r)
            inliers = abs_r <= eps * sigma
            n_outliers = n - np.sum(inliers)
            sum_sq_in = np.sum(r[inliers] ** 2)
            sigma_new = np.sqrt((sum_sq_in + sigma**2 * eps2 * n_outliers) / n)
            if abs(sigma_new - sigma) < 1e-10 * max(sigma, 1e-10):
                break
            sigma = sigma_new

        self.delta = eps * max(sigma, 1e-10)
        # Mark as estimated after first convergence or stabilization
        if hasattr(self, '_prev_sigma') and self._prev_sigma > 0:
            if abs(sigma - self._prev_sigma) / self._prev_sigma < 0.01:
                self._scale_estimated = True
        else:
            # First call: mark as estimated (scale has been computed)
            self._scale_estimated = True
        self._prev_sigma = sigma

    def _ensure_scale(self, X, y, coef):
        """Lazy scale estimation on first gradient/value call."""
        if not self._auto_scale or self._scale_estimated:
            return
        if self._method == "mad":
            self.estimate_scale(X, y, coef)
        elif self._method == "huber_prop2":
            self._update_scale_prop2(X, y, coef)

    def precompute_scale(self, X, y, coef=None):
        """Pre-estimate scale before solver starts (avoids per-iteration overhead).

        Call this once before fitting to avoid _ensure_scale() being called
        on every gradient/hessian evaluation.  If coef is None, uses OLS.

        Parameters
        ----------
        X : array (n, p)
        y : array (n,)
        coef : array (p,), optional
        """
        if not self._auto_scale or self._scale_estimated:
            return
        if coef is None:
            _X = self._to_numpy(X)
            _y = self._to_numpy(y)
            coef = np.linalg.lstsq(_X, _y, rcond=None)[0]
        if self._method == "mad":
            self.estimate_scale(X, y, coef)
        elif self._method == "huber_prop2":
            self._update_scale_prop2(X, y, coef)

    # ── Lipschitz constant (cached) ────────────────────────────────────

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        """Cached Lipschitz constant: max eigenvalue of X'X / n."""
        from statgpu.backends._array_ops import _max_eigval_power
        cache_key = id(X)
        if not hasattr(self, '_lipschitz_cache'):
            self._lipschitz_cache = {}
        if cache_key in self._lipschitz_cache:
            return self._lipschitz_cache[cache_key]
        XtX = X.T @ X
        L = _max_eigval_power(XtX) / X.shape[0]
        self._lipschitz_cache[cache_key] = L
        return L

    # ── Solver-facing methods (ensure scale before computing) ──────────

    def value(self, X, y, coef, sample_weight=None):
        """Loss value with lazy scale estimation."""
        self._ensure_scale(X, y, coef)
        return super().value(X, y, coef, sample_weight=sample_weight)

    def gradient(self, X, y, coef, sample_weight=None):
        """Gradient with lazy scale estimation."""
        self._ensure_scale(X, y, coef)
        return super().gradient(X, y, coef, sample_weight=sample_weight)

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Fused value+gradient with lazy scale estimation."""
        self._ensure_scale(X, y, coef)
        return self._fused_impl(X, y, coef, sample_weight=sample_weight)

    def hessian(self, X, y, coef, sample_weight=None):
        """Hessian with lazy scale estimation."""
        self._ensure_scale(X, y, coef)
        return self._hessian_impl(X, y, coef, sample_weight=sample_weight)

    def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
        """Compute gradient and Hessian in one pass (avoids redundant X@coef).

        Returns (gradient, hessian) tuple.  Subclasses should override
        ``_fused_grad_hess_impl()`` for the actual computation.
        """
        self._ensure_scale(X, y, coef)
        return self._fused_grad_hess_impl(X, y, coef, sample_weight=sample_weight)

    def _fused_grad_hess_impl(self, X, y, coef, sample_weight=None):
        """Default: call gradient() and hessian() separately."""
        return (
            self.gradient(X, y, coef, sample_weight=sample_weight),
            self.hessian(X, y, coef, sample_weight=sample_weight),
        )
