"""
L2 penalty (Ridge) implementation.

P(w) = (α/2) * ||w||²₂
"""

from typing import Optional
import numpy as np
from ._base import Penalty


class L2Penalty(Penalty):
    """
    L2 penalty (Ridge): P(w) = (α/2) * ||w||²₂

    The proximal operator has a closed-form solution:
        prox_{λ·||·||²/2}(z) = z / (1 + λ*step)
    """

    name = "l2"
    is_convex = True

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength.
        """
        self.alpha = alpha

    def value(self, coef: np.ndarray) -> float:
        """P(w) = (α/2) * Σw_j²"""
        return 0.5 * self.alpha * np.sum(coef ** 2)

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        """∇P(w) = α * w"""
        return self.alpha * coef

    def proximal(
        self,
        w: np.ndarray,
        step: float,
        backend: str = "numpy"
    ) -> np.ndarray:
        """
        Closed-form for L2: w / (1 + α*step)

        Parameters
        ----------
        w : array
            Input array.
        step : float
            Step size.
        backend : str
            Backend: 'numpy', 'cupy', or 'torch'.

        Returns
        -------
        array
            Scaled result.
        """
        scale = 1.0 / (1.0 + self.alpha * step)

        if backend == "cupy":
            import cupy as cp
            return scale * cp.asarray(w)
        elif backend == "torch":
            import torch
            return scale * torch.as_tensor(w)
        else:
            return scale * np.asarray(w)

    def get_params(self) -> dict:
        params = super().get_params()
        params["alpha"] = self.alpha
        return params
