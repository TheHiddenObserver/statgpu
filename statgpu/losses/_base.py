"""
Base class for all loss functions in statgpu.

LossBase is the generic base for loss functions including GLM losses
(GLMLoss), quantile regression, robust regression (Huber, Bisquare),
and survival models (Cox partial likelihood).

Subclasses implement per-sample formulas as the single source of truth.
The base class derives ``value()``, ``gradient()``, and
``fused_value_and_gradient()`` from them automatically.

For GLM-specific features (link functions, IRLS, _mu_from_eta), use
``GLMLoss(LossBase)`` in ``statgpu.glm_core._base``.
"""

__all__ = ["LossBase"]


from abc import ABC
from typing import Optional

import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp


class LossBase(ABC):
    """Generic loss function base class.

    Objective: minimize: loss(X, y, coef) + penalty(coef)

    Subclasses implement per-sample formulas as the single source of truth.
    The base class derives ``value()``, ``gradient()``, and
    ``fused_value_and_gradient()`` from them automatically.

    Subclass API (implement these):
        - ``per_sample_value(eta, y)`` — per-sample loss ℓ(η, y)
        - ``per_sample_gradient(eta, y)`` — per-sample gradient ∂ℓ/∂η

    Optional overrides:
        - ``hessian(X, y, coef, sample_weight=None)`` — Hessian for Newton solver
        - ``lipschitz(X, coef, y=None, sample_weight=None)`` — Lipschitz constant
        - ``preprocess(X, y)`` — preprocess data before fitting
        - ``predict(X, coef)`` — map coefficients to predictions
    """

    name: str = "base"
    y_type: str = "continuous"
    smooth_gradient: bool = True
    has_hessian: bool = False

    # ── Optimization hints (solvers read these, subclasses can override) ──
    _lipschitz_safety: float = 1.0       # Lipschitz safety factor
    _lipschitz_safety_cv: float = 1.0    # Extra safety factor in CV mode
    _lipschitz_uses_y: bool = False      # Whether Lipschitz needs y-scaling
    _momentum_beta_cap: Optional[float] = None  # Nesterov momentum cap (None=unlimited)
    _skip_momentum: bool = False         # Disable momentum entirely
    _has_constant_hessian: bool = False  # Hessian is constant (Newton fast path)
    _prefer_fista_over_bb: bool = False  # Prefer FISTA over FISTA-BB for smooth penalties
    _conservative_momentum_with_nonsmooth: bool = False  # Cap momentum when penalty is non-smooth
    _supports_irls: bool = False         # Whether loss has irls() method (Quantile, Bisquare, Fair)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        """Per-sample loss: ℓ(η, y). Returns array of shape (n,)."""
        raise NotImplementedError(f"{self.name} does not implement per_sample_value")

    def per_sample_gradient(self, eta, y):
        """Per-sample gradient: ∂ℓ/∂η. Returns array of shape (n,)."""
        raise NotImplementedError(f"{self.name} does not implement per_sample_gradient")

    # ── Derived methods (implemented once in base class) ──────────────

    def value(self, X, y, coef, sample_weight=None) -> float:
        """Loss value: (1/n) Σ ℓ(ηᵢ, yᵢ)."""
        xp = _get_xp(X)
        eta = X @ coef
        ps = self.per_sample_value(eta, y)
        if sample_weight is not None:
            return float(xp.dot(sample_weight, ps)) / float(sample_weight.sum())
        return float(xp.sum(ps)) / X.shape[0]

    def gradient(self, X, y, coef, sample_weight=None) -> np.ndarray:
        """Gradient: X' ∂ℓ/∂η / n."""
        xp = _get_xp(X)
        eta = X @ coef
        resid = self.per_sample_gradient(eta, y)
        if sample_weight is not None:
            return X.T @ (sample_weight * resid) / float(sample_weight.sum())
        return X.T @ resid / X.shape[0]

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Compute value and gradient in one pass (avoids redundant X @ coef).

        Returns (value, gradient) tuple.
        """
        xp = _get_xp(X)
        eta = X @ coef
        ps = self.per_sample_value(eta, y)
        resid = self.per_sample_gradient(eta, y)
        if sample_weight is not None:
            sw_sum = float(sample_weight.sum())
            val = float(xp.dot(sample_weight, ps)) / sw_sum
            grad = X.T @ (sample_weight * resid) / sw_sum
        else:
            n = X.shape[0]
            val = float(xp.sum(ps)) / n
            grad = X.T @ resid / n
        return val, grad

    def hessian(self, X, y, coef, sample_weight=None) -> np.ndarray:
        """Hessian matrix (for Newton solver).

        Raises NotImplementedError when auto solver falls back to FISTA.
        """
        raise NotImplementedError(
            f"{self.name} does not support Hessian."
        )

    def lipschitz(self, X, coef, y=None, sample_weight=None) -> float:
        """Lipschitz constant (for FISTA step size step=1/L)."""
        from statgpu.backends._array_ops import _max_eigval_power
        XtX = X.T @ X
        return _max_eigval_power(XtX) / X.shape[0]

    def preprocess(self, X, y):
        """Preprocess data. Default returns as-is."""
        return X, y

    def predict(self, X, coef):
        """Map from X @ coef to prediction. Default X @ coef."""
        return X @ coef
