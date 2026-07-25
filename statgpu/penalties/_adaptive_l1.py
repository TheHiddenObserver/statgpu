"""
Adaptive L1 penalty (Adaptive Lasso).

Zou, JASA 2006. Convex penalty with data-driven per-coordinate weights.

The penalty is:
    P(w) = alpha * sum(weights_j * |w_j|)
where weights_j = 1 / (|init_coef_j| + eps)^nu.

The weights are set via set_weights() using an initial OLS or Ridge estimate.
"""

__all__ = ["AdaptiveL1Penalty"]

from typing import Optional
import numpy as np
from statgpu.backends._array_ops import _xp
from statgpu.penalties._base import Penalty

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
    eps : float, default=1e-8
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
            self._norm_factor = 1.0
            if self.normalize:
                # Normalize by mean to match R glmnet's penalty.factor convention.
                mean_w = float(np.mean(w))
                if mean_w > 0:
                    w = w / mean_w
                    self._norm_factor = mean_w
            self._weights = w
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
        # Convert to numpy for weight computation (weights are always stored as numpy)
        from statgpu.backends._utils import _to_numpy
        coef_np = np.asarray(_to_numpy(coef), dtype=np.float64).ravel()
        # If the init coef is all-zero (e.g., ridge init diverged),
        # fall back to uniform weights so adaptive_l1 reduces to L1.
        if not np.any(np.abs(coef_np) > 1e-12):
            self._weights = np.ones_like(coef_np)
            self._norm_factor = 1.0
            return
        raw = 1.0 / (np.abs(coef_np) + self.eps) ** self.nu
        self._norm_factor = 1.0
        if self.normalize:
            mean_w = float(np.mean(raw))
            if mean_w > 0:
                raw = raw / mean_w
                self._norm_factor = mean_w
        self._weights = raw
        # Invalidate cached device tensors so proximal recomputes them.
        for _k in ('_alpha_w_torch', '_alpha_w_cupy',
                    '_alpha_w_torch_src', '_alpha_w_cupy_src'):
            if hasattr(self, _k):
                delattr(self, _k)

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef) -> float:
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(np.asarray(coef))
        mod = type(coef).__module__
        if mod.startswith("torch"):
            import torch
            # Reuse cached device tensor from proximal() if available
            _cached = getattr(self, '_alpha_w_torch', None)
            _src = getattr(self, '_alpha_w_torch_src', None)
            if _cached is not None and _src is self._weights:
                return (_cached * torch.abs(coef)).sum().item()
            w = self._weights
            _is_dev = type(w).__module__.startswith("torch")
            if _is_dev:
                if _cached is None or _src is not w:
                    _cached = self.alpha * w.to(device=coef.device, dtype=torch.float64)
                    self._alpha_w_torch = _cached
                    self._alpha_w_torch_src = w
            else:
                if _cached is None or _src is not w:
                    _cached = torch.tensor(self.alpha * np.asarray(w, dtype=float),
                                           device=coef.device, dtype=torch.float64)
                    self._alpha_w_torch = _cached
                    self._alpha_w_torch_src = w
            return (_cached * torch.abs(coef)).sum().item()
        elif mod.startswith("cupy"):
            import cupy as cp
            _cached = getattr(self, '_alpha_w_cupy', None)
            _src = getattr(self, '_alpha_w_cupy_src', None)
            w = self._weights
            _is_dev = type(w).__module__.startswith("cupy")
            if _is_dev:
                if _cached is None or _src is not w:
                    _cached = self.alpha * w
                    self._alpha_w_cupy = _cached
                    self._alpha_w_cupy_src = w
            else:
                if _cached is None or _src is not w:
                    _cached = cp.asarray(self.alpha * np.asarray(w, dtype=float))
                    self._alpha_w_cupy = _cached
                    self._alpha_w_cupy_src = w
            return float((_cached * cp.abs(coef)).sum())
        else:
            return self.alpha * np.sum(self._weights * np.abs(coef))

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef):
        xp = _xp(coef)
        if not hasattr(self, "_weights"):
            self._weights = xp.ones_like(coef)
        return self.alpha * self._weights * xp.sign(coef)

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

        # Check if _weights is already a device tensor (from lla_weights on GPU)
        _w_mod = type(self._weights).__module__
        _is_device = _w_mod.startswith("torch") or _w_mod.startswith("cupy")

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
            # Cache device tensor for alpha*weights across calls.
            # Use _weights_src_id to detect when _weights is reassigned externally.
            _cache_key = '_alpha_w_cupy'
            _src_key = '_alpha_w_cupy_src'
            _cached = getattr(self, _cache_key, None)
            _src = getattr(self, _src_key, None)
            if _is_device:
                if _cached is None or _src is not self._weights:
                    _cached = self.alpha * self._weights
                    setattr(self, _cache_key, _cached)
                    setattr(self, _src_key, self._weights)
            else:
                if _cached is None or _src is not self._weights:
                    alpha_w = self.alpha * np.asarray(self._weights, dtype=float)
                    _cached = cp.asarray(alpha_w)
                    setattr(self, _cache_key, _cached)
                    setattr(self, _src_key, self._weights)
            thresh_gpu = _cached * step
            return AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY(w, thresh_gpu)
        elif backend == "torch":
            import torch
            # Cache device tensor for alpha*weights across calls.
            _cache_key = '_alpha_w_torch'
            _src_key = '_alpha_w_torch_src'
            _cached = getattr(self, _cache_key, None)
            _src = getattr(self, _src_key, None)
            if _is_device:
                if _cached is None or _src is not self._weights:
                    _cached = self.alpha * self._weights.to(device=w.device, dtype=torch.float64)
                    setattr(self, _cache_key, _cached)
                    setattr(self, _src_key, self._weights)
            else:
                if _cached is None or _src is not self._weights:
                    alpha_w = self.alpha * np.asarray(self._weights, dtype=float)
                    _cached = torch.tensor(alpha_w, device=w.device, dtype=torch.float64)
                    setattr(self, _cache_key, _cached)
                    setattr(self, _src_key, self._weights)
            thresh_t = _cached * step
            compiled_fn = _get_adaptive_l1_torch_compiled()
            if compiled_fn is not None:
                return compiled_fn(w, thresh_t)
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh_t)
        else:
            alpha_w = self.alpha * np.asarray(self._weights, dtype=float)
            thresh_arr = alpha_w * step
            return np.sign(w) * np.maximum(np.abs(w) - thresh_arr, 0.0)

    # ----------------------------------------------------------------
    # LLA weights (identity: this is already a weighted L1 penalty)
    # ----------------------------------------------------------------

    def lla_weights(self, coef):
        """Return LLA weights, converted to the same backend as coef."""
        if not hasattr(self, "_weights"):
            self._weights = np.ones_like(np.asarray(coef))
        # Convert weights to the same backend as coef to avoid device-to-host transfer
        from statgpu.backends._array_ops import _xp
        xp = _xp(coef)
        if xp is np:
            return self._weights.copy()
        return xp.asarray(self._weights, dtype=coef.dtype)

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "alpha": self.alpha,
            "nu": self.nu,
        })
        return params
