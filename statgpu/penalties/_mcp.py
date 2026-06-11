"""
MCP penalty (Minimax Concave Penalty).

Zhang, Annals of Statistics 2010. Non-convex penalty with oracle property.

Element-wise:
    p(w_j) = {
        alpha * |w_j| - w_j^2 / (2*gamma)     if |w_j| <= gamma*alpha
        (1/2) * gamma * alpha^2                if |w_j| > gamma*alpha
    }

Supports both FISTA direct (proximal) and LLA (lla_weights) optimization.
"""
from typing import Optional
import numpy as np
from ._base import Penalty

# ---- torch.compile lazy-loader (fuses elementwise ops into 1 kernel) ---------
_MCP_PROXIMAL_TORCH_COMPILED = None


def _get_mcp_torch_compiled():
    global _MCP_PROXIMAL_TORCH_COMPILED
    if _MCP_PROXIMAL_TORCH_COMPILED is not None:
        return _MCP_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _MCP_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w, step, alpha, gamma):
            max_step = 0.9 * gamma
            step = torch.clamp(step, max=max_step)
            t = alpha * step
            abs_w = torch.abs(w)
            sign_w = torch.sign(w)
            r1 = abs_w <= t
            r3 = abs_w > gamma * alpha
            r2 = ~(r1 | r3)
            result = torch.where(r1,
                torch.zeros_like(w),
                torch.where(r2,
                    sign_w * (abs_w - t) / (1.0 - step / gamma),
                    w))
            return result
        _MCP_PROXIMAL_TORCH_COMPILED = torch.compile(_prox, dynamic=True, mode='reduce-overhead')
    except Exception:
        _MCP_PROXIMAL_TORCH_COMPILED = None
    return _MCP_PROXIMAL_TORCH_COMPILED


class MCPPenalty(Penalty):
    """MCP penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    gamma : float, default=3.0
        Concavity parameter. Zhang recommends gamma > 1 (default 3.0).

    Notes
    -----
    MCP is **non-convex** (``is_convex=False``). The objective function may
    contain multiple local minima. Different solvers (e.g. ``fista`` vs
    ``fista_bb``) can converge to different local minima with comparable
    objective values — a coefficient ``max|diff|`` up to ~1e-2 is expected
    and does not indicate a bug. The objective values should agree within
    ~1e-4 relative tolerance across runs.
    """

    name = "mcp"
    is_convex = False

    def __init__(self, alpha: float = 1.0, gamma: float = 3.0):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for MCP penalty")
        if not np.isfinite(gamma) or gamma <= 1.0:
            raise ValueError("gamma must be a finite scalar greater than 1 for MCP penalty")
        self.alpha = alpha
        self.gamma = gamma

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef: np.ndarray) -> float:
        from statgpu.backends._array_ops import _xp
        from statgpu.backends._utils import _to_float_scalar
        xp = _xp(coef)
        alpha = self.alpha
        gamma = self.gamma

        abs_w = xp.abs(coef)
        region1 = abs_w <= gamma * alpha
        region2 = ~region1
        total = xp.sum(alpha * abs_w[region1] - abs_w[region1] ** 2 / (2.0 * gamma))
        total = total + 0.5 * gamma * alpha ** 2 * xp.sum(region2)
        return _to_float_scalar(total)

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        abs_w = np.abs(coef)
        sign_w = np.sign(coef)
        alpha = self.alpha
        gamma = self.gamma

        grad = np.zeros_like(coef, dtype=float)

        mask1 = abs_w <= gamma * alpha
        grad[mask1] = sign_w[mask1] * (alpha - abs_w[mask1] / gamma)

        return grad

    # ----------------------------------------------------------------
    # Proximal operator (FISTA direct path)
    # ----------------------------------------------------------------

    # Lazy-loaded fused CuPy kernel (single launch vs ~10 intermediate arrays)
    _MCP_PROXIMAL_CUPY = None

    def proximal(
        self,
        w,
        step: float,
        backend: str = "numpy",
    ):
        """Closed-form MCP proximal operator (three regions per coordinate).

        Clamp step < gamma so the three-region formula always applies.
        """
        alpha = self.alpha
        gamma = self.gamma
        max_step = 0.9 * gamma
        if step > max_step:
            step = max_step
        t = alpha * step

        if backend == "cupy":
            import cupy as cp
            if MCPPenalty._MCP_PROXIMAL_CUPY is None:
                MCPPenalty._MCP_PROXIMAL_CUPY = cp.ElementwiseKernel(
                    'float64 w, float64 step, float64 alpha, float64 gamma',
                    'float64 result',
                    '''
                    double max_step = 0.9 * gamma;
                    double s = (step > max_step) ? max_step : step;
                    double abs_w = abs(w);
                    double t = alpha * s;
                    double sign_w = (w > 0.0) ? 1.0 : ((w < 0.0) ? -1.0 : 0.0);
                    if (abs_w <= t) {
                        result = 0.0;
                    } else if (abs_w <= gamma * alpha) {
                        result = sign_w * (abs_w - t) / (1.0 - s / gamma);
                    } else {
                        result = w;
                    }
                    ''',
                    'mcp_proximal',
                )
            return MCPPenalty._MCP_PROXIMAL_CUPY(w, step, alpha, gamma)

        elif backend == "torch":
            import torch
            compiled_fn = _get_mcp_torch_compiled()
            if compiled_fn is not None:
                step_t = torch.as_tensor(step, dtype=w.dtype, device=w.device)
                return compiled_fn(w, step_t, alpha, gamma)
            abs_w = torch.abs(w)
            sign_w = torch.sign(w)

            r1 = abs_w <= t
            r3 = abs_w > gamma * alpha
            r2 = ~(r1 | r3)
            result = torch.where(r1,
                torch.zeros_like(w),
                torch.where(r2,
                    sign_w * (abs_w - t) / (1.0 - step / gamma),
                    w))
            return result

        else:
            abs_w = np.abs(w)
            sign_w = np.sign(w)

            region1 = abs_w <= t
            region3 = abs_w > gamma * alpha
            region2 = ~(region1 | region3)

            result = np.zeros_like(w, dtype=float)
            result[region2] = (
                sign_w[region2]
                * (abs_w[region2] - t)
                / (1.0 - step / gamma)
            )
            result[region3] = w[region3]
            return result

    # ----------------------------------------------------------------
    # LLA weights (Local Linear Approximation path)
    # ----------------------------------------------------------------

    def lla_weights(self, coef):
        """
        LLA weights: w_j = P'(|coef_j|) — the subgradient of MCP at |coef_j|.

        w_j = {
            alpha - |coef_j| / gamma   if |coef_j| <= gamma*alpha
            0                           if |coef_j| > gamma*alpha
        }

        Accepts numpy, cupy, or torch arrays. Returns same backend type.
        """
        alpha = self.alpha
        gamma = self.gamma

        from statgpu.backends._array_ops import _xp
        xp = _xp(coef)
        abs_w = xp.abs(coef)
        weights = xp.zeros_like(coef)
        mask = abs_w <= gamma * alpha
        weights[mask] = alpha - abs_w[mask] / gamma
        return weights

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"alpha": self.alpha, "gamma": self.gamma})
        return params
