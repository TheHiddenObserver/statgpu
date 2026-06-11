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


class GLMLoss(ABC):
    """GLM loss function base class.

    Objective: minimize: loss(X, y, w) + penalty(w)
    """

    name: str = "base"
    y_type: str = "continuous"
    smooth_gradient: bool = True
    has_hessian: bool = False

    @abstractmethod
    def value(self, X, y, coef, sample_weight=None) -> float:
        """Loss value (不含 penalty)."""
        pass

    @abstractmethod
    def gradient(self, X, y, coef, sample_weight=None) -> np.ndarray:
        """Gradient of loss w.r.t. w."""
        pass

    def hessian(self, X, y, coef) -> np.ndarray:
        """Hessian matrix (for IRLS/Newton).

        Raises NotImplementedError when auto solver falls back to FISTA.
        """
        raise NotImplementedError(
            f"{self.name} does not support Hessian."
        )

    def lipschitz(self, X, coef, y=None) -> float:
        """Lipschitz constant (for FISTA step size step=1/L)."""
        XtX = X.T @ X
        return float(np.linalg.eigvalsh(XtX)[-1]) / X.shape[0]

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
