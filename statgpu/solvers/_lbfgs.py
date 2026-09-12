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
    _validate_sample_weight,
    _as_backend_vector,
    _validate_smooth_penalty,
)


def _prepare_lbfgs_sample_weight(sample_weight, n_samples, backend, ref_arr, loss):
    """Validate analytic weights and align active weights to the fit backend.

    Uniform weights retain the historical L-BFGS behavior and are normalized
    away before optimization.  Genuine non-uniform weights are accepted only
    for losses that explicitly opt into the shared weighted-L-BFGS contract.

    Match Newton's established ordering exactly: validate first, align to the
    executed design backend/dtype, then apply the historical uniformity rule.
    This prevents the two explicit smooth solvers from classifying the same
    public weight vector differently merely because its input container/dtype
    differs from the numerical design.
    """
    if sample_weight is None:
        return None

    _validate_sample_weight(sample_weight, n_samples)
    values = _as_backend_vector(sample_weight, backend, ref_arr).reshape(-1)
    if backend == "torch":
        import torch

        uniform_dev = (
            torch.allclose(values, values[0])
            if torch.is_floating_point(values)
            else torch.all(values == values[0])
        )
    else:
        from statgpu.backends._utils import _get_xp

        xp = _get_xp(backend)
        uniform_dev = (
            xp.allclose(values, values[0])
            if getattr(values.dtype, "kind", "") == "f"
            else xp.all(values == values[0])
        )
    uniform = bool(
        uniform_dev.item() if hasattr(uniform_dev, "item") else uniform_dev
    )
    if uniform:
        return None

    if not bool(getattr(loss, "_supports_nonuniform_lbfgs_weights", False)):
        raise ValueError(
            "lbfgs_solver does not support non-uniform sample_weight for "
            f"loss='{getattr(loss, 'name', '?')}'."
        )

    return values


def _call_loss_with_weight(fn, *args, sample_weight=None):
    """Call one loss primitive with analytic weights when they are active."""
    if sample_weight is None:
        return fn(*args)
    return fn(*args, sample_weight=sample_weight)


def lbfgs_solver(
    loss,
    penalty,
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

    Genuine non-uniform ``sample_weight`` is supported only when the loss
    explicitly opts into the shared weighted-L-BFGS contract.  Maintained GLM
    losses do so and evaluate value, gradient, line-search candidates, and the
    accepted iterate under one normalized objective
    ``sum_i w_i * contribution_i / sum_i w_i``.  Generic non-GLM losses remain
    fail-closed unless they independently declare the same capability.

    Uniform weights are normalized away using the historical uniformity rule,
    preserving the established unweighted numerical path.

    Parameters
    ----------
    loss : object
        Loss with ``fused_value_and_gradient(X, y, coef)`` and
        ``preprocess(X, y)`` methods.
    penalty : object or None
        Smooth penalty (l2 or none).
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
        Analytic sample weights. Uniform weights are equivalent to unweighted
        fitting; genuine non-uniform weights require a capable loss contract.

    Returns
    -------
    params : array
        Optimised coefficient vector.
    n_iter : int
        Number of iterations performed.
    """
    _validate_smooth_penalty(penalty, "lbfgs_solver")
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    sample_weight = _prepare_lbfgs_sample_weight(
        sample_weight, X_proc.shape[0], backend, X_proc, loss
    )

    if init_coef is not None:
        params = _as_backend_vector(init_coef, backend, X_proc)
    else:
        params = _zeros(n_features, backend, ref_tensor=X_proc)

    s_hist = []
    y_hist = []
    rho_hist = []

    # Initial gradient (fused to avoid redundant X@coef)
    _init_val_dev, grad = _call_loss_with_weight(
        loss.fused_value_and_gradient,
        X_proc,
        y_proc,
        params,
        sample_weight=sample_weight,
    )
    grad = grad + _smooth_penalty_gradient(penalty, params)

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
            gdd = -gn * gn  # grad'(-grad) = -||grad||^2

        # Line search -- stays on device and uses the same analytic weights as
        # the gradient that generated the search direction.
        old_val_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params,
            sample_weight=sample_weight,
        )
        old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)

        step = 1.0
        params_new = params
        _ls_accepted = False
        for _ in range(25):
            candidate = params + step * direction
            cand_val_dev, _ = _call_loss_with_weight(
                loss.fused_value_and_gradient,
                X_proc,
                y_proc,
                candidate,
                sample_weight=sample_weight,
            )
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

        # Update gradient (fused) using the same weighted objective.
        _, grad_new = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params_new,
            sample_weight=sample_weight,
        )
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