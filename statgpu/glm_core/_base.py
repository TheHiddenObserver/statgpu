"""
Base class for GLM loss functions in statgpu.

The GLM core loss framework supports 7 families:
- Squared error (linear regression)
- Logistic loss (binary classification)
- Poisson loss (count data)
- Gamma loss (positive continuous)
- Inverse Gaussian loss (positive continuous)
- Negative Binomial loss (overdispersed count data)
- Tweedie loss (generalized GLM family)

GLMLoss extends LossBase with GLM-specific features:
- ``_mu_from_eta(eta)`` — link inverse μ = g⁻¹(η)
- ``_poisson_like``, ``_gamma_like`` — family-specific solver hints
- IRLS support (only for GLMLoss subclasses)

Structured models such as Cox, panel, and time-series models should use
LossBase directly rather than this GLM-specific interface.
"""

__all__ = ["GLMLoss", "get_glm_loss", "register_glm_loss", "list_glm_losses"]


from typing import Optional

from statgpu.losses._base import LossBase


class GLMLoss(LossBase):
    """GLM loss function base class.

    Extends LossBase with GLM-specific features (link functions, IRLS hints).

    Objective: minimize: loss(X, y, w) + penalty(w)

    Subclasses implement per-sample formulas as the single source of truth.
    The base class derives ``value()``, ``gradient()``, and
    ``fused_value_and_gradient()`` from them automatically.

    Subclass API (implement these):
        - ``per_sample_value(eta, y)`` — per-sample loss ℓ(η, y)
        - ``per_sample_gradient(eta, y)`` — per-sample gradient ∂ℓ/∂η
        - ``_mu_from_eta(eta)`` — link inverse μ = g⁻¹(η), with clipping
    """

    # ── GLM-specific optimization hints ──
    # (solvers read these via getattr(..., False) — safe if absent)
    _is_quadratic: bool = False          # True for squared_error (XtX constant, no y-scaling)
    _supports_cholesky: bool = False     # True for squared_error (ADMM can use Cholesky)
    _gpu_loop_excluded: bool = False     # True for logistic (async GPU loop not suitable)
    _inverse_gaussian: bool = False      # True for inverse Gaussian (special BB handling)
    _tweedie: bool = False               # True for Tweedie (special BB handling)
    _poisson_like: bool = False          # True for Poisson (conservative momentum burn-in)
    _gamma_like: bool = False            # True for Gamma (adjusted BB/momentum params)

    def _mu_from_eta(self, eta):
        """Link inverse: μ = g⁻¹(η). Override for clipping."""
        return eta  # default: identity link

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Fused value+gradient using GLM-specific optimized kernels.

        Dispatches to family-specific fused implementations (logistic, poisson,
        gamma, etc.) that compute eta = X @ coef once and derive both value
        and gradient from it, avoiding redundant matmul.
        """
        if sample_weight is not None:
            from statgpu.glm_core._fused import _weighted_loss_and_grad
            return _weighted_loss_and_grad(self, X, y, coef, sample_weight)

        from statgpu.glm_core._fused import _fused_glm_value_and_gradient
        return _fused_glm_value_and_gradient(self, X, y, coef)


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
        # Also register in the generic loss registry
        from statgpu.losses._registry import _LOSS_REGISTRY
        if name not in _LOSS_REGISTRY:
            _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def list_glm_losses() -> list:
    """List all registered GLM loss names."""
    return list(_GLM_LOSS_REGISTRY.keys())
