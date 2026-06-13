"""
L1 penalty (Lasso) implementation.

P(w) = α * ||w||₁
"""

__all__ = ["L1Penalty"]


from typing import Optional
import numpy as np
from statgpu.penalties._base import Penalty

# ---- torch.compile lazy-loader (fuses elementwise ops into 1 kernel) ---------
_L1_PROXIMAL_TORCH_COMPILED = None


def _get_l1_torch_compiled():
    global _L1_PROXIMAL_TORCH_COMPILED
    if _L1_PROXIMAL_TORCH_COMPILED is not None:
        return _L1_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _L1_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w, thresh):
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh)
        _L1_PROXIMAL_TORCH_COMPILED = torch.compile(_prox, mode='reduce-overhead')
    except Exception:
        _L1_PROXIMAL_TORCH_COMPILED = None
    return _L1_PROXIMAL_TORCH_COMPILED


class L1Penalty(Penalty):
    """
    L1 penalty: P(w) = α * ||w||₁

    The proximal operator is the soft thresholding function:
        prox_{λ·||·||₁}(z) = sign(z) * max(|z| - λ, 0)
    """

    name = "l1"
    is_convex = True

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
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

        # torch.compile fast path (performance optimization)
        if backend == "torch":
            compiled_fn = _get_l1_torch_compiled()
            if compiled_fn is not None:
                return compiled_fn(w, thresh)

        # Unified fallback across numpy/cupy/torch
        from statgpu.backends._array_ops import _soft_threshold
        return _soft_threshold(w, thresh)

    def get_params(self) -> dict:
        params = super().get_params()
        params["alpha"] = self.alpha
        return params
