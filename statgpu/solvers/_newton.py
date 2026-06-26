"""Newton-Raphson solver with Armijo backtracking line search.

Generic solver — works with any loss that implements hessian() and gradient().
"""

from __future__ import annotations

__all__ = ["newton_solver"]

import warnings
import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._array_ops import (
    _copy_arr,
    _dot_dev,
    _norm2_dev,
    _sync_scalars,
    _zeros,
    _device_leq,
)
from statgpu.backends._utils import _to_float_scalar

from ._convergence import ConvergenceWarning
from ._utils import (
    _validate_uniform_sample_weight,
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _smooth_penalty_value_dev,
)


def newton_solver(
    loss,
    penalty,
    X,
    y,
    max_iter: int = 100,
    tol: float = 1e-4,
    init_coef=None,
    sample_weight=None,
) -> tuple:
    """Newton-Raphson solver with Armijo backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    For losses with constant Hessian (e.g. Gamma log link), the Hessian
    doesn't change across iterations, so the Newton step is always valid
    and line search is skipped.

    Requires: loss has hessian() and penalty is smooth.
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        params = _zeros(n_features, backend, ref_tensor=X_proc)

    # Constant-Hessian detection via loss attribute (generic, not loss-name based)
    _const_hessian = getattr(loss, "_has_constant_hessian", False)

    _fixed_hess = None
    if _const_hessian:
        _fixed_hess = loss.hessian(X_proc, y_proc, params) + _smooth_penalty_hessian(
            penalty, params
        )

    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "newton_solver")
    iteration = -1
    line_search_failed = False

    for iteration in range(max_iter):
        params_old = _copy_arr(params)
        grad = loss.gradient(X_proc, y_proc, params) + _smooth_penalty_gradient(
            penalty, params
        )
        grad_norm_dev = _norm2_dev(grad)
        (grad_norm,) = _sync_scalars(grad_norm_dev, backend=backend)
        if grad_norm <= tol:
            break

        hess = _fixed_hess if _fixed_hess is not None else (
            loss.hessian(X_proc, y_proc, params) + _smooth_penalty_hessian(penalty, params)
        )
        hess = 0.5 * (hess + hess.T)

        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess, grad)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.solve(hess, grad)
            else:
                import torch

                direction = torch.linalg.solve(hess, grad.unsqueeze(1))
                direction = direction.squeeze(1)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            if backend == "numpy":
                direction = np.linalg.lstsq(hess, grad, rcond=None)[0]
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess, grad)[0]
            else:
                import torch

                direction = torch.linalg.lstsq(hess, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)

        # Armijo backtracking line search
        obj_old_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_old)
        obj_old_dev = obj_old_dev + _smooth_penalty_value_dev(penalty, params_old)
        gdd_dev = _dot_dev(grad, direction)
        gdd = _to_float_scalar(gdd_dev)
        if not np.isfinite(gdd) or gdd <= 0.0:
            # A Newton system may be singular or numerically indefinite.
            # Fall back to steepest descent so Armijo still has a descent step.
            direction = grad
            gdd = grad_norm * grad_norm

        step = 1.0
        accepted = False
        for _bt in range(20):
            params_try = params_old - step * direction
            try:
                obj_try_dev, _ = loss.fused_value_and_gradient(X_proc, y_proc, params_try)
                obj_try_dev = obj_try_dev + _smooth_penalty_value_dev(
                    penalty, params_try
                )
                if _device_leq(obj_try_dev, obj_old_dev - 1e-4 * step * gdd):
                    params = params_try
                    accepted = True
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                pass
            step *= 0.5
        if not accepted:
            # Never accept an unverified trial step.  A tiny rejected step
            # would also make a parameter-difference test report false
            # convergence.
            params = params_old
            line_search_failed = True
            break

    n_iter = iteration + 1
    if line_search_failed:
        warnings.warn(
            "newton_solver line search failed to find a descent step "
            f"(loss={getattr(loss, 'name', '?')}, "
            f"penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    elif n_iter >= max_iter:
        warnings.warn(
            f"newton_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, "
            f"penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter
