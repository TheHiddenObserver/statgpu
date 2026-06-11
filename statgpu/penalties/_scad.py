"""
SCAD penalty (Smoothly Clipped Absolute Deviation).

Fan & Li, JASA 2001. Non-convex penalty with oracle property.

Element-wise:
    p(w_j) = {
        alpha * |w_j|                                         if |w_j| <= alpha
        -(w_j^2 - 2*a*alpha*|w_j| + alpha^2) / (2*(a-1))     if alpha < |w_j| <= a*alpha
        (a+1)*alpha^2 / 2                                     if |w_j| > a*alpha
    }

Supports both FISTA direct (proximal) and LLA (lla_weights) optimization.
"""
from typing import Optional
import numpy as np
from ._base import Penalty
from statgpu.backends._array_ops import _xp
from statgpu.backends._utils import _to_float_scalar

# ---- torch.compile lazy-loader (fuses elementwise ops into 1 kernel) ---------
_SCAD_PROXIMAL_TORCH_COMPILED = None


def _get_scad_torch_compiled():
    global _SCAD_PROXIMAL_TORCH_COMPILED
    if _SCAD_PROXIMAL_TORCH_COMPILED is not None:
        return _SCAD_PROXIMAL_TORCH_COMPILED
    from statgpu.penalties import _torch_compile_ok
    if not _torch_compile_ok():
        _SCAD_PROXIMAL_TORCH_COMPILED = None
        return None
    try:
        import torch
        def _prox(w, step, alpha, a):
            max_step = 0.9 * (a - 1.0)
            step = torch.clamp(step, max=max_step)
            t = alpha * step
            abs_w = torch.abs(w)
            sign_w = torch.sign(w)
            r1 = abs_w <= alpha + t
            r3 = abs_w > a * alpha
            r2 = ~(r1 | r3)
            result = torch.where(r1,
                sign_w * torch.relu(abs_w - t),
                torch.where(r2,
                    sign_w * ((a - 1.0) * abs_w - a * t) / (a - 1.0 - step),
                    w))
            return result
        _SCAD_PROXIMAL_TORCH_COMPILED = torch.compile(_prox, dynamic=True, mode='reduce-overhead')
    except Exception:
        _SCAD_PROXIMAL_TORCH_COMPILED = None
    return _SCAD_PROXIMAL_TORCH_COMPILED


class SCADPenalty(Penalty):
    """SCAD penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    a : float, default=3.7
        Concavity parameter. Fan & Li recommend 3.7.

    Notes
    -----
    SCAD is **non-convex** (``is_convex=False``). The objective function may
    contain multiple local minima. Different solvers (e.g. ``fista`` vs
    ``fista_bb``) can converge to different local minima with comparable
    objective values — a coefficient ``max|diff|`` up to ~1e-2 is expected
    and does not indicate a bug. The objective values should agree within
    ~1e-4 relative tolerance across runs.
    """

    name = "scad"
    is_convex = False

    def __init__(self, alpha: float = 1.0, a: float = 3.7):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for SCAD penalty")
        if not np.isfinite(a) or a <= 2.0:
            raise ValueError("a must be a finite scalar greater than 2 for SCAD penalty")
        self.alpha = alpha
        self.a = a

    # ----------------------------------------------------------------
    # Value
    # ----------------------------------------------------------------

    def value(self, coef: np.ndarray) -> float:
        xp = _xp(coef)
        a = self.a
        alpha = self.alpha

        abs_w = xp.abs(coef)
        region1 = abs_w <= alpha
        region2 = (abs_w > alpha) & (abs_w <= a * alpha)
        region3 = abs_w > a * alpha
        total = alpha * xp.sum(abs_w[region1])
        total = total + xp.sum(
            -(abs_w[region2] ** 2 - 2 * a * alpha * abs_w[region2] + alpha ** 2)
            / (2.0 * (a - 1.0))
        )
        total = total + (a + 1.0) * alpha ** 2 / 2.0 * xp.sum(region3)
        return _to_float_scalar(total)

    # ----------------------------------------------------------------
    # Gradient
    # ----------------------------------------------------------------

    def gradient(self, coef: np.ndarray) -> np.ndarray:
        abs_w = np.abs(coef)
        sign_w = np.sign(coef)
        a = self.a
        alpha = self.alpha

        grad = np.zeros_like(coef, dtype=float)

        # Region 1: |w| <= alpha → alpha * sign(w)
        mask1 = abs_w <= alpha
        grad[mask1] = alpha * sign_w[mask1]

        # Region 2: alpha < |w| <= a*alpha → (a*alpha*sign - w) / (a-1)
        mask2 = (abs_w > alpha) & (abs_w <= a * alpha)
        grad[mask2] = (a * alpha * sign_w[mask2] - coef[mask2]) / (a - 1.0)

        # Region 3: |w| > a*alpha → 0
        return grad

    # ----------------------------------------------------------------
    # Proximal operator (FISTA direct path)
    # ----------------------------------------------------------------

    # Lazy-loaded fused CuPy kernel (single launch vs ~15 intermediate arrays)
    _SCAD_PROXIMAL_CUPY = None

    def proximal(
        self,
        w,
        step: float,
        backend: str = "numpy",
    ):
        """Closed-form SCAD proximal operator (three regions per coordinate).

        When step > a-1 the three-region formula degenerates (division by
        zero or negative denominator).  Clamp step so the three-region
        logic always applies — this matches R ncvreg's per-coordinate
        behaviour where each coordinate has its own step v_j and the
        threshold is always alpha (never alpha*v_j).
        """
        alpha = self.alpha
        a = self.a
        # Clamp step: ensure a > 1 + step (three-region condition).
        # Use 0.9*(a-1) as max to avoid the singularity at step = a-1.
        max_step = 0.9 * (a - 1.0)
        if step > max_step:
            step = max_step
        t = alpha * step

        if backend == "cupy":
            import cupy as cp
            if SCADPenalty._SCAD_PROXIMAL_CUPY is None:
                SCADPenalty._SCAD_PROXIMAL_CUPY = cp.ElementwiseKernel(
                    'float64 w, float64 step, float64 alpha, float64 a',
                    'float64 result',
                    '''
                    double max_step = 0.9 * (a - 1.0);
                    double s = (step > max_step) ? max_step : step;
                    double abs_w = abs(w);
                    double t = alpha * s;
                    double sign_w = (w > 0.0) ? 1.0 : ((w < 0.0) ? -1.0 : 0.0);
                    if (abs_w <= alpha + t) {
                        double v = abs_w - t;
                        result = sign_w * (v > 0.0 ? v : 0.0);
                    } else if (abs_w <= a * alpha) {
                        result = sign_w * ((a - 1.0) * abs_w - a * t) / (a - 1.0 - s);
                    } else {
                        result = w;
                    }
                    ''',
                    'scad_proximal',
                )
            return SCADPenalty._SCAD_PROXIMAL_CUPY(w, step, alpha, a)

        elif backend == "torch":
            import torch
            compiled_fn = _get_scad_torch_compiled()
            if compiled_fn is not None:
                step_t = torch.as_tensor(step, dtype=w.dtype, device=w.device)
                return compiled_fn(w, step_t, alpha, a)
            abs_w = torch.abs(w)
            sign_w = torch.sign(w)

            r1 = abs_w <= alpha + t
            r3 = abs_w > a * alpha
            r2 = ~(r1 | r3)
            result = torch.where(r1,
                sign_w * torch.relu(abs_w - t),
                torch.where(r2,
                    sign_w * ((a - 1.0) * abs_w - a * t) / (a - 1.0 - step),
                    w))
            return result

        else:
            abs_w = np.abs(w)
            sign_w = np.sign(w)

            region1 = abs_w <= alpha + t
            region3 = abs_w > a * alpha
            region2 = ~(region1 | region3)

            result = np.zeros_like(w, dtype=float)
            result[region1] = sign_w[region1] * np.maximum(abs_w[region1] - t, 0.0)
            result[region2] = (
                sign_w[region2]
                * ((a - 1.0) * abs_w[region2] - a * t)
                / (a - 1.0 - step)
            )
            result[region3] = w[region3]
            return result

    # ----------------------------------------------------------------
    # LLA weights (Local Linear Approximation path)
    # ----------------------------------------------------------------

    def lla_weights(self, coef):
        """
        LLA weights: w_j = P'(|coef_j|) — the subgradient of SCAD at |coef_j|.

        w_j = {
            alpha                            if |coef_j| <= alpha
            (a*alpha - |coef_j|) / (a - 1)   if alpha < |coef_j| <= a*alpha
            0                                 if |coef_j| > a*alpha
        }

        Accepts numpy, cupy, or torch arrays. Returns same backend type.
        """
        a = self.a
        alpha = self.alpha

        xp = _xp(coef)
        abs_w = xp.abs(coef)
        weights = xp.full_like(coef, alpha)
        mask2 = (abs_w > alpha) & (abs_w <= a * alpha)
        weights[mask2] = (a * alpha - abs_w[mask2]) / (a - 1.0)
        mask3 = abs_w > a * alpha
        weights[mask3] = 0.0
        return weights

    # ----------------------------------------------------------------

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"alpha": self.alpha, "a": self.a})
        return params
