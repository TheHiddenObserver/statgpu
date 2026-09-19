"""
Adaptive L1 penalty (Adaptive Lasso).

Zou, JASA 2006. Convex penalty with data-driven per-coordinate weights.

The penalty is:
    P(w) = alpha * sum(weights_j * |w_j|)
where weights_j = 1 / (|init_coef_j| + eps)^nu.

The weights are set via set_weights() using an initial OLS or Ridge estimate.
"""

__all__ = ["AdaptiveL1Penalty"]

from statgpu.backends._torch_compile import compile_torch
from typing import Optional
import numpy as np
from statgpu.backends._array_ops import _xp, _xp_asarray
from statgpu.penalties._base import Penalty

# ---- torch.compile lazy-loader (fuses elementwise ops into 1 kernel) ---------
_ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = None


def _get_adaptive_l1_torch_compiled():
    global _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED
    if _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED is not None:
        return _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED
    import torch
    def _prox(w, thresh_tensor):
        return torch.sign(w) * torch.relu(torch.abs(w) - thresh_tensor)
    _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = compile_torch(_prox, dynamic=True, workload="iterative")
    return _ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED

def _normalize_adaptive_controls(alpha, nu, eps, init_method, normalize):
    """Validate public Adaptive-L1 constructor controls."""
    from numbers import Real

    def numeric(value, name, *, allow_zero):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.number)
        ):
            raise TypeError(f"{name} must be a finite numeric scalar")
        result = float(value)
        if not np.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {qualifier} scalar")
        return result

    alpha_value = numeric(alpha, "alpha", allow_zero=True)
    nu_value = numeric(nu, "nu", allow_zero=False)
    eps_value = numeric(eps, "eps", allow_zero=False)

    if not isinstance(init_method, str):
        raise TypeError("init_method must be one of 'auto', 'ols', or 'ridge'")
    normalized_method = init_method.lower()
    if normalized_method not in ("auto", "ols", "ridge"):
        raise ValueError("init_method must be one of 'auto', 'ols', or 'ridge'")
    # Preserve already-canonical strings by identity for sklearn<=1.2 clone.
    method_value = init_method if init_method == normalized_method else normalized_method

    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be boolean")
    normalize_value = bool(normalize)

    return alpha_value, nu_value, eps_value, method_value, normalize_value


def _normalize_external_weights(weights):
    """Validate external adaptive weights and return a clone-safe snapshot."""
    if weights is None:
        return None

    raw = np.asarray(weights)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if raw.dtype.kind in ("b", "S", "U"):
        raise TypeError("weights must contain real numeric values")
    if raw.dtype.kind == "O":
        from numbers import Real
        for value in raw:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (Real, np.number)
            ):
                raise TypeError("weights must contain real numeric values")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("weights must contain real numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")

    normalized = tuple(float(value) for value in values)
    if (
        isinstance(weights, tuple)
        and len(weights) == len(normalized)
        and all(type(value) is float for value in weights)
        and weights == normalized
    ):
        return weights
    return normalized


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
        (
            self.alpha,
            self.nu,
            self.eps,
            self.init_method,
            self.normalize,
        ) = _normalize_adaptive_controls(
            alpha, nu, eps, init_method, normalize
        )
        self.weights = _normalize_external_weights(weights)
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
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
        for _k in (
            '_alpha_w_torch', '_alpha_w_cupy',
            '_alpha_w_torch_src', '_alpha_w_cupy_src',
            '_alpha_w_torch_alpha', '_alpha_w_cupy_alpha',
            '_alpha_w_torch_dtype', '_alpha_w_cupy_dtype',
        ):
            if hasattr(self, _k):
                delattr(self, _k)

    def _require_weights(self):
        """Return initialized adaptive weights or fail before numerical use."""
        weights = getattr(self, "_weights", None)
        if weights is None:
            raise RuntimeError(
                "AdaptiveL1Penalty weights are not initialized; call "
                "set_weights() or pass weights=... before numerical use."
            )
        return weights

    def _cached_alpha_weights(self, ref, backend: str):
        """Return alpha*weights on ref's concrete device/dtype with truthful cache."""
        cache_key = f"_alpha_w_{backend}"
        src_key = f"_alpha_w_{backend}_src"
        alpha_key = f"_alpha_w_{backend}_alpha"
        dtype_key = f"_alpha_w_{backend}_dtype"
        cached = getattr(self, cache_key, None)
        source = self._require_weights()
        cached_source = getattr(self, src_key, None)
        cached_alpha = getattr(self, alpha_key, None)
        cached_dtype = getattr(self, dtype_key, None)
        alpha_value = float(self.alpha)

        if backend == "torch":
            import torch
            target_dtype = (
                ref.dtype if torch.is_floating_point(ref) else torch.float64
            )
            dtype_token = str(target_dtype)
        elif backend == "cupy":
            import cupy as cp
            try:
                ref_kind = np.dtype(ref.dtype).kind
            except (TypeError, ValueError):
                ref_kind = "f"
            target_dtype = ref.dtype if ref_kind in "fc" else cp.float64
            dtype_token = str(np.dtype(target_dtype))
        else:
            ref_arr = np.asarray(ref)
            target_dtype = (
                ref_arr.dtype if ref_arr.dtype.kind in "fc" else np.float64
            )
            dtype_token = str(np.dtype(target_dtype))

        same_device = True
        if cached is not None and backend == "torch":
            same_device = (
                getattr(cached, "device", None) == getattr(ref, "device", None)
            )
        elif cached is not None and backend == "cupy":
            cached_device = getattr(getattr(cached, "device", None), "id", None)
            ref_device = getattr(getattr(ref, "device", None), "id", None)
            same_device = (
                cached_device is not None
                and ref_device is not None
                and int(cached_device) == int(ref_device)
            )

        if (
            cached is not None
            and cached_source is source
            and cached_alpha == alpha_value
            and cached_dtype == dtype_token
            and same_device
        ):
            return cached

        if backend == "torch":
            import torch
            if type(source).__module__.startswith("torch"):
                aligned = source.to(device=ref.device, dtype=target_dtype)
            else:
                aligned = torch.as_tensor(
                    np.asarray(source, dtype=np.float64),
                    device=ref.device,
                    dtype=target_dtype,
                )
            alpha_scalar = torch.as_tensor(
                alpha_value, device=ref.device, dtype=target_dtype
            )
        elif backend == "cupy":
            import cupy as cp
            aligned = _xp_asarray(source, target_dtype, ref)
            alpha_scalar = np.dtype(target_dtype).type(alpha_value)
        else:
            aligned = np.asarray(source, dtype=target_dtype)
            alpha_scalar = np.dtype(target_dtype).type(alpha_value)

        cached = alpha_scalar * aligned
        setattr(self, cache_key, cached)
        setattr(self, src_key, source)
        setattr(self, alpha_key, alpha_value)
        setattr(self, dtype_key, dtype_token)
        return cached

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef) -> float:
        self._require_weights()
        mod = type(coef).__module__
        if mod.startswith("torch"):
            import torch
            alpha_w = self._cached_alpha_weights(coef, "torch")
            return (alpha_w * torch.abs(coef)).sum().item()
        elif mod.startswith("cupy"):
            import cupy as cp
            alpha_w = self._cached_alpha_weights(coef, "cupy")
            return float((alpha_w * cp.abs(coef)).sum())
        else:
            return self.alpha * np.sum(self._weights * np.abs(coef))

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef):
        xp = _xp(coef)
        self._require_weights()
        weights = self.lla_weights(coef)
        return self.alpha * weights * xp.sign(coef)

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
        self._require_weights()

        # Check if _weights is already a device tensor (from lla_weights on GPU)
        _w_mod = type(self._weights).__module__
        _is_device = _w_mod.startswith("torch") or _w_mod.startswith("cupy")

        if backend == "cupy":
            import cupy as cp
            try:
                kind = np.dtype(w.dtype).kind
            except (TypeError, ValueError):
                kind = "f"
            w_work = w if kind in "fc" else _xp_asarray(w, cp.float64, w)
            if AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY is None:
                AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY = cp.ElementwiseKernel(
                    'T w, T thresh',
                    'T result',
                    '''
                    T abs_w = abs(w);
                    T sign_w = (w > (T)0) ? (T)1 : ((w < (T)0) ? (T)-1 : (T)0);
                    if (abs_w > thresh) {
                        result = sign_w * (abs_w - thresh);
                    } else {
                        result = (T)0;
                    }
                    ''',
                    'adaptive_l1_proximal',
                )
            _cached = self._cached_alpha_weights(w_work, "cupy")
            step_value = _xp_asarray(step, w_work.dtype, w_work)
            thresh_gpu = _cached * step_value
            return AdaptiveL1Penalty._ADAPTIVE_L1_PROXIMAL_CUPY(
                w_work, thresh_gpu
            )
        elif backend == "torch":
            import torch
            w_work = w if torch.is_floating_point(w) else w.to(torch.float64)
            _cached = self._cached_alpha_weights(w_work, "torch")
            step_value = torch.as_tensor(
                step, dtype=w_work.dtype, device=w_work.device
            )
            thresh_t = _cached * step_value
            compiled_fn = _get_adaptive_l1_torch_compiled()
            if compiled_fn is not None:
                return compiled_fn(w_work, thresh_t)
            return torch.sign(w_work) * torch.relu(torch.abs(w_work) - thresh_t)
        else:
            w_work = np.asarray(w)
            target_dtype = (
                w_work.dtype if w_work.dtype.kind in "fc" else np.dtype(np.float64)
            )
            w_work = np.asarray(w_work, dtype=target_dtype)
            alpha_w = (
                np.asarray(self.alpha, dtype=target_dtype)
                * np.asarray(self._weights, dtype=target_dtype)
            )
            thresh_arr = alpha_w * np.asarray(step, dtype=target_dtype)
            return np.sign(w_work) * np.maximum(
                np.abs(w_work) - thresh_arr,
                np.asarray(0.0, dtype=target_dtype),
            )

    # ----------------------------------------------------------------
    # LLA weights (identity: this is already a weighted L1 penalty)
    # ----------------------------------------------------------------

    def lla_weights(self, coef):
        """Return LLA weights, converted to the same backend as coef."""
        self._require_weights()
        # Convert weights to the same backend and concrete device as coef.
        xp = _xp(coef)
        if xp is np:
            coef_arr = np.asarray(coef)
            target_dtype = (
                coef_arr.dtype
                if coef_arr.dtype.kind in "fc"
                else np.dtype(np.float64)
            )
            return np.asarray(self._weights, dtype=target_dtype).copy()
        if xp.__name__ == "torch":
            import torch
            target_dtype = (
                coef.dtype if torch.is_floating_point(coef) else torch.float64
            )
        else:
            try:
                kind = np.dtype(coef.dtype).kind
            except (TypeError, ValueError):
                kind = "f"
            target_dtype = coef.dtype if kind in "fc" else xp.float64
        return _xp_asarray(self._weights, target_dtype, coef)

    # ----------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor params for clone or descriptive serialization."""
        if not deep:
            return {
                "alpha": self.alpha,
                "nu": self.nu,
                "eps": self.eps,
                "init_method": self.init_method,
                "normalize": self.normalize,
                "weights": self.weights,
            }

        params = super().get_params(deep=deep)
        params.update({
            "alpha": self.alpha,
            "nu": self.nu,
            "eps": self.eps,
            "init_method": self.init_method,
            "normalize": self.normalize,
            "weights": None if self.weights is None else list(self.weights),
        })
        return params
