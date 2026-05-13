"""
Adaptive L1 penalty (Adaptive Lasso).

Zou, JASA 2006. Convex penalty with data-driven per-coordinate weights.

The penalty is:
    P(w) = alpha * sum(weights_j * |w_j|)
where weights_j = 1 / (|init_coef_j| + eps)^nu.

The weights are set via set_weights() using an initial OLS or Ridge estimate.
"""
from typing import Optional
import numpy as np
from ._base import Penalty

# ---- torch.compile lazy-loader (fuses elementwise ops into 1 kernel) ---------
_ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = None


def _get_adaptive_l1_torch_compiled():
    global _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED
    if _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED is not None:
        return _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w, thresh_tensor):
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh_tensor)
        _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = torch.compile(_prox, dynamic=True, mode='reduce-overhead')
    except Exception:
        _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = None
    return _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED


class AdaptiveL1Penalty(Penalty):
    """Adaptive L1 penalty (Adaptive Lasso).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    nu : float, default=1.0
        Exponent for weight computation (1 or 2, per Zou 2006).
    eps : float, default=1e-4
        Small constant to avoid division by zero.
    init_method : str, default='auto'
        Method for initial coefficient estimates:
        - 'auto': OLS if n > p, Ridge otherwise
        - 'ols': forced OLS (errors if p > n)
        - 'ridge': forced Ridge (always works)
    normalize : bool, default=True
        If True, normalize weights by their mean to match R glmnet's
        penalty.factor convention (R normalizes so mean(pf) = 1).
        Set to False to use raw 1/|coef| weights with no normalization.
    weights : array-like, optional
        Pre-computed per-coordinate weights.  When provided, ``set_weights``
        is a no-op — the external weights are used as-is.  When ``None``,
        weights are computed from an initial fit via ``set_weights``.

    Notes
    -----
    With fixed weights adaptive_l1 is convex (``is_convex=True``). However,
    when used as the inner solver for non-convex penalties (SCAD, MCP) via
    LLA, the overall optimization is non-convex and may converge to different
    local minima depending on the solver and initialization. In standalone use
    (fixed weights from a pre-fit), results are deterministic and reproducible.
    """

    name = "adaptive_l1"
    is_convex = True
    requires_init = True

    def __init__(
        self,
        alpha: float = 1.0,
        nu: float = 1.0,
        eps: float = 1e-8,
        init_method: str = "auto",
        normalize: bool = True,
        weights: Optional[np.ndarray] = None,
    ):
        self.alpha = alpha
        self.nu = nu
        self.eps = eps
        self.init_method = init_method
        self.normalize = normalize
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            # Normalize by mean to match R glmnet's penalty.factor convention.
            # R's glmnet internally normalizes penalty.factor so that
            # mean(penalty.factor) = 1.  Without this, raw 1/|coef| weights
            # (which can have mean >> 1) make the penalty orders of magnitude
            # stronger than intended.
            mean_w = float(np.mean(w))
            if mean_w > 0:
                self._weights = w / mean_w
                self._norm_factor = mean_w
            else:
                self._weights = w
                self._norm_factor = 1.0
        else:
            self._weights = None

    def set_weights(self, coef: np.ndarray):
        """Compute adaptive weights from initial coefficient estimates.

        weights_j = 1 / (|coef_j| + eps)^nu

        If ``weights`` was passed to __init__, the external weights are kept
        and this method is a no-op (normalization is handled in __init__).

        When ``normalize=True`` (default), weights are divided by their
        mean to match R glmnet's penalty.factor convention (R normalizes
        penalty factors so that mean(pf) = 1 internally).

        When ``normalize=False``, raw 1/|coef| weights are used (no
        normalization).
        """
        if self._weights is not None:
            return
        # If the init coef is all-zero (e.g., ridge init diverged),
        # fall back to uniform weights so adaptive_l1 reduces to L1.
        if not np.any(np.abs(coef) > 1e-12):
            self._weights = np.ones_like(coef)
            self._norm_factor = 1.0
            return
        raw = 1.0 / (np.abs(coef) + self.eps) ** self.nu
        self._norm_factor = 1.0
        if self.normalize:
            mean_w = float(np.mean(raw))
            if mean_w > 0:
                raw = raw / mean_w
                self._norm_factor = mean_w
        self._weights = raw

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef: np.ndarray) -> float:
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(coef)
        return self.alpha * np.sum(self._weights * np.abs(coef))

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(coef)
        return self.alpha * self._weights * np.sign(coef)

    # ----------------------------------------------------------------
    # Proximal operator (FISTA path)
    # ----------------------------------------------------------------

    # Lazy-loaded fused CuPy kernel
    _ADAPTIVE_L1_PROXIMAL_CUPY = None

    def proximal(
        self,
        w,
        step: float,
        backend: str = "numpy",
    ):
        """Per-coordinate soft-threshold with per-coordinate thresholds."""
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(np.asarray(w))

        weights = np.asarray(self._weights, dtype=float)
        thresh_arr = self.alpha * weights * step

        if backend == "cupy":
            import cupy as cp
            if AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY is None:
                AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY = cp.ElementwiseKernel(
                    'float64 w, float64 thresh',
                    'float64 result',
                    '''
                    double abs_w = abs(w);
                    double sign_w = (w > 0.0) ? 1.0 : ((w < 0.0) ? -1.0 : 0.0);
                    if (abs_w > thresh) {
                        result = sign_w * (abs_w - thresh);
                    } else {
                        result = 0.0;
                    }
                    ''',
                    'adaptive_l1_proximal',
                )
            thresh_gpu = cp.asarray(thresh_arr)
            return AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY(w, thresh_gpu)
        elif backend == "torch":
            import torch
            thresh_t = torch.tensor(thresh_arr, device=w.device, dtype=torch.float64)
            compiled_fn = _get_adaptive_l1_torch_compiled()
            if compiled_fn is not None:
                return compiled_fn(w, thresh_t)
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh_t)
        else:
            return np.sign(w) * np.maximum(np.abs(w) - thresh_arr, 0.0)

    # ----------------------------------------------------------------
    # LLA weights (identity: this is already a weighted L1 penalty)
    # ----------------------------------------------------------------

    def lla_weights(self, coef: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(coef)
        return self._weights.copy()

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "alpha": self.alpha,
            "nu": self.nu,
        })
        return params
