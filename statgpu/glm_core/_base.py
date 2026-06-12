"""
Base class for GLM loss functions in statgpu.

The GLM core loss framework supports:
- Squared error (linear regression)
- Logistic loss (logistic regression)
- Poisson loss (count data)

Structured models such as Cox, panel, and time-series models should use a
future objective layer rather than this GLM-specific interface.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp_mod
from statgpu.backends._utils import _to_float_scalar


class GLMLoss(ABC):
    """GLM loss function base class.

    Objective: minimize: loss(X, y, w) + penalty(w)

    Subclasses implement per-sample formulas as the single source of truth.
    The base class derives ``value()``, ``gradient()``, and
    ``fused_value_and_gradient()`` from them automatically.

    Subclass API (implement these):
        - ``per_sample_value(eta, y)`` — per-sample loss ℓ(η, y)
        - ``per_sample_gradient(eta, y)`` — per-sample gradient ∂ℓ/∂η
        - ``_mu_from_eta(eta)`` — link inverse μ = g⁻¹(η), with clipping
    """

    name: str = "base"
    y_type: str = "continuous"
    smooth_gradient: bool = True
    has_hessian: bool = False

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        """Per-sample loss: ℓ(η, y). Returns array of shape (n,)."""
        raise NotImplementedError(f"{self.name} does not implement per_sample_value")

    def per_sample_gradient(self, eta, y):
        """Per-sample gradient: ∂ℓ/∂η. Returns array of shape (n,)."""
        raise NotImplementedError(f"{self.name} does not implement per_sample_gradient")

    def _mu_from_eta(self, eta):
        """Link inverse: μ = g⁻¹(η). Override for clipping."""
        return eta  # default: identity link

    # ── Derived methods (implemented once in base class) ──────────────

    def value(self, X, y, coef, sample_weight=None) -> float:
        """Loss value: (1/n) Σ ℓ(ηᵢ, yᵢ)."""
        xp = _get_xp_mod(X)
        eta = X @ coef
        ps = self.per_sample_value(eta, y)
        if sample_weight is not None:
            return float(xp.sum(sample_weight * ps)) / float(sample_weight.sum())
        return float(xp.sum(ps)) / X.shape[0]

    def gradient(self, X, y, coef, sample_weight=None) -> np.ndarray:
        """Gradient: X' ∂ℓ/∂η / n."""
        xp = _get_xp_mod(X)
        eta = X @ coef
        resid = self.per_sample_gradient(eta, y)
        if sample_weight is not None:
            return X.T @ (sample_weight * resid) / float(sample_weight.sum())
        return X.T @ resid / X.shape[0]

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Compute value and gradient in one pass (avoids redundant X @ coef).

        Returns (value, gradient) tuple.
        """
        xp = _get_xp_mod(X)
        eta = X @ coef
        ps = self.per_sample_value(eta, y)
        resid = self.per_sample_gradient(eta, y)
        if sample_weight is not None:
            sw_sum = float(sample_weight.sum())
            val = float(xp.sum(sample_weight * ps)) / sw_sum
            grad = X.T @ (sample_weight * resid) / sw_sum
        else:
            n = X.shape[0]
            val = float(xp.sum(ps)) / n
            grad = X.T @ resid / n
        return val, grad

    def hessian(self, X, y, coef) -> np.ndarray:
        """Hessian matrix (for IRLS/Newton).

        Raises NotImplementedError when auto solver falls back to FISTA.
        """
        raise NotImplementedError(
            f"{self.name} does not support Hessian."
        )

    def lipschitz(self, X, coef, y=None) -> float:
        """Lipschitz constant (for FISTA step size step=1/L)."""
        from statgpu.backends._array_ops import _max_eigval_power
        XtX = X.T @ X
        return _max_eigval_power(XtX) / X.shape[0]

    def preprocess(self, X, y):
        """Preprocess y. Default returns as-is."""
        return X, y

    def predict(self, X, coef):
        """Map from X @ coef to prediction. Default X @ coef."""
        return X @ coef


# ─── Registry ──────────────────────────────────────────────────────────────

_GLM_LOSS_REGISTRY: dict = {}


def get_glm_loss(name: str, **kwargs) -> GLMLoss:
    """
    Get a GLM loss by name from the registry.

    Parameters
    ----------
    name : str
        GLM loss name: 'squared_error', 'logistic', 'poisson', etc.
    **kwargs
        Arguments passed to the loss constructor.

    Returns
    -------
    GLMLoss
        Instantiated GLM loss object.

    Raises
    ------
    ValueError
        If loss name is not in the registry.
    """
    if name not in _GLM_LOSS_REGISTRY:
        available = list(_GLM_LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown GLM loss: {name}. Available GLM losses: {available}"
        )
    return _GLM_LOSS_REGISTRY[name](**kwargs)


def register_glm_loss(name: str):
    """
    Decorator to register a custom GLM loss class.

    Parameters
    ----------
    name : str
        Name to register the GLM loss under.

    Returns
    -------
    callable
        Decorator function that registers the loss class.
    """
    def decorator(cls):
        if not issubclass(cls, GLMLoss):
            raise TypeError(
                f"GLM loss class must inherit from GLMLoss, got {cls.__bases__}"
            )
        _GLM_LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def list_glm_losses() -> list:
    """List all registered GLM loss names."""
    return list(_GLM_LOSS_REGISTRY.keys())
