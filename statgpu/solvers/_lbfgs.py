"""Limited-memory BFGS solver for smooth penalised objectives.

Generic solver -- works with any loss that implements fused_value_and_gradient().
Keeps parameters, gradients, and curvature history on the input backend.
GPU-optimised path uses:
- loss.fused_value_and_gradient to avoid redundant X@coef
- _dot_dev / _norm2_dev to stay on device
- _sync_scalars to batch GPU-to-CPU transfers
- _device_leq for device-side line search
"""

from __future__ import annotations

__all__ = ["lbfgs_solver"]

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


def lbfgs_solver(
    loss: "GLMLoss",
    penalty: "Penalty | None",
    X,
    y,
    max_iter: int = 100,
    tol: float = 1e-4,
    init_coef=None,
    history_size: int = 10,
    sample_weight=None,
) -> tuple:
    """Limited-memory BFGS for smooth objectives.

    Works with any loss that implements ``fused_value_and_gradient(X, y, coef)``
    returning ``(value, gradient)``.  Supports numpy / cupy / torch backends
    via auto-detection of *X*.

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
        Maximum number of L-BFGS iterations.
    tol : float
        Convergence tolerance on gradient norm and step norm.
    init_coef : array-like or None
        Initial coefficient vector.  Zeros if *None*.
    history_size : int
        Number of past (s, y) pairs to store.
    sample_weight : array-like or None
        Sample weights.  Must be uniform (all equal) for this solver.

    Returns
    -------
    params : array
        Optimised coefficient vector.
    n_iter : int
        Number of iterations performed.
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "lbfgs_solver")

    if init_coef is not None:
        params = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        params = _zeros(n_features, backend, ref_tensor=X)

    s_hist = []
    y_hist = []
    rho_hist = []

    # Initial gradient (fused to avoid redundant X@coef)
    _init_val_dev, grad = loss.fused_value_and_gradient(X_proc, y_proc, params)
    grad = grad + _smooth_penalty_gradient(penalty, params)

    if backend == "torch":
        import torch
        tol_dev = torch.tensor(tol, dtype=torch.float64, device=params.device)
    else:
        tol_dev = tol
    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        grad_norm_dev = _norm2_dev(grad)

        # Two-loop recursion -- all dot products stay on device
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

        # Batch sync: grad_norm + grad_dot_dir
        gn, gdd = _sync_scalars(grad_norm_dev, gdd_dev, backend=backend)
        if gn < tol:
            break
        if gdd >= 0:
            direction = -grad
            gdd = -gn  # -||grad||^2

        # Line search -- stays on device
        old_val_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params)
        old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)

        step = 1.0
        params_new = params
        _ls_accepted = False
        for _ in range(25):
            candidate = params + step * direction
            cand_val_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, candidate)
            cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(penalty, candidate)
            # Device-side comparison -- single sync for the bool
            if _device_leq(cand_val_dev, old_val_dev + 1e-4 * step * gdd):
                params_new = candidate
                _ls_accepted = True
                break
            step *= 0.5
        if not _ls_accepted:
            warnings.warn(
                "lbfgs_solver: line search failed to find a descent step "
                f"after 25 backtracking steps (iteration {iteration}). "
                "Solver may stagnate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Update gradient (fused)
        _, grad_new = loss.fused_value_and_gradient(X_proc, y_proc, params_new)
        grad_new = grad_new + _smooth_penalty_gradient(penalty, params_new)

        s_vec = params_new - params
        y_vec = grad_new - grad
        ys_dev = _dot_dev(y_vec, s_vec)
        s_norm_dev = _norm2_dev(s_vec)

        # Batch sync: ys + s_norm
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
            f"lbfgs_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter
