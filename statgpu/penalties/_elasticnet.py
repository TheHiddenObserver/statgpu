"""
Elastic Net penalty implementation.

P(w) = α * l1_ratio * ||w||₁ + (α/2) * (1 - l1_ratio) * ||w||²₂
"""

from typing import Optional
import numpy as np
from ._base import Penalty


class ElasticNetPenalty(Penalty):
    """
    Elastic Net penalty: P(w) = α * l1_ratio * ||w||₁ + (α/2) * (1-l1_ratio) * ||w||²₂

    Combines L1 (sparsity) and L2 (grouping effect) penalties.
    - l1_ratio = 1: Pure L1 (Lasso)
    - l1_ratio = 0: Pure L2 (Ridge)
    - 0 < l1_ratio < 1: Combination

    The proximal operator combines soft thresholding with L2 scaling:
        prox(w) = soft_threshold(w, α*l1_ratio*step) / (1 + α*(1-l1_ratio)*step)
    """

    name = "elasticnet"
    is_convex = True

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength.
        l1_ratio : float, default=0.5
            Mixing parameter between L1 and L2 penalties.
            - l1_ratio = 1: L1 only (Lasso)
            - l1_ratio = 0: L2 only (Ridge)
            - 0 < l1_ratio < 1: Combined
        """
        self.alpha = alpha
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
        self.l1_ratio = l1_ratio

    def value(self, coef: np.ndarray) -> float:
        """P(w) = α*l1_ratio*||w||₁ + (α/2)*(1-l1_ratio)*||w||²₂"""
        l1 = self.alpha * self.l1_ratio * np.sum(np.abs(coef))
        l2 = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(coef ** 2)
        return l1 + l2

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        """∇P(w) = α*l1_ratio*sign(w) + α*(1-l1_ratio)*w"""
        grad_l1 = self.alpha * self.l1_ratio * np.sign(coef)
        grad_l2 = self.alpha * (1 - self.l1_ratio) * coef
        return grad_l1 + grad_l2

    def proximal(
        self,
        w: np.ndarray,
        step: float,
        backend: str = "numpy"
    ) -> np.ndarray:
        """
        Soft thresholding + L2 scaling.

        prox(w) = sign(w) * max(|w| - α*l1_ratio*step, 0) / (1 + α*(1-l1_ratio)*step)

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
            Proximal result.
        """
        thresh = self.alpha * self.l1_ratio * step
        l2_scale = 1.0 + self.alpha * (1 - self.l1_ratio) * step

        from statgpu.backends._array_ops import _soft_threshold
        return _soft_threshold(w, thresh) / l2_scale

    def get_params(self) -> dict:
        params = super().get_params()
        params["alpha"] = self.alpha
        params["l1_ratio"] = self.l1_ratio
        return params
