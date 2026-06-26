"""L-BFGS-B solver: L-BFGS with box constraints (projected gradient variant).

Extends the standard L-BFGS solver with box constraints support.
Uses projected gradient approach: after each step, clip params to bounds.
Sufficient for joint optimization with loose bounds (e.g. log_sigma).

For full L-BFGS-B (Byrd et al. 1995) with Cauchy point + subspace
minimization, see scipy.optimize._lbfgsb_py as reference.
"""

from __future__ import annotations

__all__ = ["lbfgs_b_solver"]

import warnings
import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._array_ops import (
    _copy_arr,
    _device_gt,
    _device_leq,
    _dot_dev,
    _norm2_dev,
    _sync_scalars,
    _zeros,
)

from ._convergence import ConvergenceWarning
from ._utils import (
    _smooth_penalty_gradient,
    _smooth_penalty_value_dev,
    _validate_uniform_sample_weight,
)


def lbfgs_b_solver(
    loss,
    penalty,
    X,
    y,
    max_iter: int = 100,
    tol: float = 1e-4,
    init_coef=None,
    history_size: int = 10,
    sample_weight=None,
    lower_bounds=None,
    upper_bounds=None,
) -> tuple:
    """L-BFGS-B: L-BFGS with box constraints.

    Parameters
    ----------
    loss : object
        Loss with ``fused_value_and_gradient(X, y, coef)`` and
        ``preprocess(X, y)`` methods.
    penalty : object or None
        Smooth penalty (l2, elasticnet, none).
    X, y : array-like
        Design matrix and response vector.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on projected gradient norm and step norm.
    init_coef : array-like or None
        Initial coefficient vector.  Zeros if *None*.
    history_size : int
        Number of past (s, y) pairs to store.
    sample_weight : array-like or None
        Sample weights.  Must be uniform (all equal).
    lower_bounds : array-like or None
        Lower bounds for each parameter.  -inf if *None*.
    upper_bounds : array-like or None
        Upper bounds for each parameter.  +inf if *None*.

    Returns
    -------
    params : array
        Optimised coefficient vector (clipped to bounds).
    n_iter : int
        Number of iterations performed.
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "lbfgs_b_solver")

    # Initialize params
    if init_coef is not None:
        params = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        params = _zeros(n_features, backend, ref_tensor=X)

    # Initialize bounds
    if backend == "torch":
        import torch
        _neg_inf = torch.full((n_features,), float("-inf"), dtype=torch.float64, device=params.device)
        _pos_inf = torch.full((n_features,), float("inf"), dtype=torch.float64, device=params.device)
    else:
        _neg_inf = np.full(n_features, float("-inf"))
        _pos_inf = np.full(n_features, float("inf"))

    lb = _neg_inf if lower_bounds is None else (
        lower_bounds if hasattr(lower_bounds, "shape") else np.array(lower_bounds)
    )
    ub = _pos_inf if upper_bounds is None else (
        upper_bounds if hasattr(upper_bounds, "shape") else np.array(upper_bounds)
    )

    # Clip initial params to bounds
    params = _clip_to_bounds(params, lb, ub, backend)

    s_hist = []
    y_hist = []
    rho_hist = []

    # Initial gradient
    _init_val_dev, grad = loss.fused_value_and_gradient(X_proc, y_proc, params)
    grad = grad + _smooth_penalty_gradient(penalty, params)

    iteration = -1

    for iteration in range(max_iter):
        # Projected gradient norm: only count free components
        proj_grad = _projected_gradient(grad, params, lb, ub)
        pg_norm_dev = _norm2_dev(proj_grad)

        # Two-loop recursion
        q = _copy_arr(grad)
        alphas = []
        for s_vec, y_vec, rho in reversed(list(zip(s_hist, y_hist, rho_hist))):
            alpha = rho * _dot_dev(s_vec, q)
            alphas.append(alpha)
            q = q - alpha * y_vec

        if y_hist:
            sy = _dot_dev(s_hist[-1], y_hist[-1])
            yy = _dot_dev(y_hist[-1], y_hist[-1])
            gamma = sy / yy if _device_gt(yy, 1e-30) else 1.0
        else:
            gamma = 1.0
        r = gamma * q

        for s_vec, y_vec, rho, alpha in zip(
            s_hist, y_hist, rho_hist, reversed(alphas)
        ):
            beta = rho * _dot_dev(y_vec, r)
            r = r + s_vec * (alpha - beta)

        direction = -r
        gdd_dev = _dot_dev(grad, direction)

        pg_norm, gdd = _sync_scalars(pg_norm_dev, gdd_dev, backend=backend)
        if pg_norm < tol:
            break
        if gdd >= 0:
            direction = -grad
            gdd = -_norm2_dev(grad)
            gdd = float(gdd) if not hasattr(gdd, "item") else float(gdd.item())

        # Line search with bounds clipping
        old_val_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params)
        old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)

        step = 1.0
        params_new = params
        _ls_accepted = False
        for _ in range(25):
            candidate = _clip_to_bounds(params + step * direction, lb, ub, backend)
            cand_val_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, candidate)
            cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(penalty, candidate)
            if _device_leq(cand_val_dev, old_val_dev + 1e-4 * step * gdd):
                params_new = candidate
                _ls_accepted = True
                break
            step *= 0.5
        if not _ls_accepted:
            warnings.warn(
                "lbfgs_b_solver: line search failed to find a descent step "
                f"after 25 backtracking steps (iteration {iteration}). "
                "Solver may stagnate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Update gradient
        _, grad_new = loss.fused_value_and_gradient(X_proc, y_proc, params_new)
        grad_new = grad_new + _smooth_penalty_gradient(penalty, params_new)

        s_vec = params_new - params
        y_vec = grad_new - grad
        ys_dev = _dot_dev(y_vec, s_vec)
        s_norm_dev = _norm2_dev(s_vec)

        ys, s_norm = _sync_scalars(ys_dev, s_norm_dev, backend=backend)
        if ys > 1e-12:
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        params = params_new
        grad = grad_new
        if s_norm < tol:
            break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"lbfgs_b_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter


def _clip_to_bounds(params, lb, ub, backend):
    """Clip parameters to [lb, ub]. Works on all backends."""
    if backend == "torch":
        import torch
        return torch.clamp(params, min=lb, max=ub)
    else:
        xp = np
        return xp.maximum(xp.minimum(params, ub), lb)


def _projected_gradient(grad, params, lb, ub):
    """Projected gradient: zero out components at active bounds.

    A component is at a bound if:
    - params[i] == lb[i] and grad[i] > 0 (at lower bound, gradient points up)
    - params[i] == ub[i] and grad[i] < 0 (at upper bound, gradient points down)
    """
    backend = "torch" if hasattr(params, "device") else "numpy"
    if backend == "torch":
        import torch
        at_lower = (params <= lb) & (grad > 0)
        at_upper = (params >= ub) & (grad < 0)
        at_bound = at_lower | at_upper
        return grad * (~at_bound).to(grad.dtype)
    else:
        at_lower = (params <= lb) & (grad > 0)
        at_upper = (params >= ub) & (grad < 0)
        at_bound = at_lower | at_upper
        mask = (~at_bound).astype(grad.dtype)
        return grad * mask
