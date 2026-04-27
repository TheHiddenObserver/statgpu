"""
L1 penalty (Lasso) implementation.

P(w) = α * ||w||₁
"""

from typing import Optional
import numpy as np
from ._base import Penalty


class L1Penalty(Penalty):
    """
    L1 penalty: P(w) = α * ||w||₁

    The proximal operator is the soft thresholding function:
        prox_{λ·||·||₁}(z) = sign(z) * max(|z| - λ, 0)
    """

    name = "l1"
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
        """P(w) = α * Σ|w_j|"""
        return self.alpha * np.sum(np.abs(coef))

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        """∇P(w) = α * sign(w)"""
        return self.alpha * np.sign(coef)

    def proximal(
        self,
        w: np.ndarray,
        step: float,
        backend: str = "numpy"
    ) -> np.ndarray:
        """
        Soft thresholding: sign(z) * max(|z| - α*step, 0)

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
            Soft-thresholded result.
        """
        thresh = self.alpha * step

        if backend == "cupy":
            import cupy as cp
            return cp.sign(w) * cp.maximum(cp.abs(w) - thresh, 0.0)
        elif backend == "torch":
            import torch
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh)
        else:
            # numpy
            return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

    def get_params(self) -> dict:
        params = super().get_params()
        params["alpha"] = self.alpha
        return params
