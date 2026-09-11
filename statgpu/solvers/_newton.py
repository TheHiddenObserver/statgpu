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
    _validate_sample_weight,
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _smooth_penalty_value_dev,
    _runtime_error_is_singular,
    _as_backend_vector,
    _validate_smooth_penalty,
    _trial_error_is_numerical,
)


def _prepare_newton_sample_weight(sample_weight, n_samples, backend, ref_arr):
    """Validate and align analytic weights to the Newton execution backend.

    Uniform weights are normalized away using the historical solver tolerance
    (``allclose`` for floating-point arrays), because that contract already
    treated them as the unweighted objective. This also preserves compatibility
    for losses that explicitly reject genuine weighting (for example Cox).
    Any genuinely non-uniform vector remains explicit throughout value,
    gradient, Hessian, and line-search evaluation.
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
    return None if uniform else values


def _call_loss_with_weight(fn, *args, sample_weight=None):
    """Call one loss primitive with analytic weights when they are active."""
    if sample_weight is None:
        return fn(*args)
    return fn(*args, sample_weight=sample_weight)


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

    Non-uniform ``sample_weight`` is treated as analytic/frequency-style
    objective weighting using the same normalized average-loss convention as
    ``LossBase``: weighted loss, gradient, Hessian, and every Armijo trial use
    ``sum_i w_i * contribution_i / sum_i w_i``. Losses that do not implement
    weighted curvature remain free to reject the request explicitly.

    For losses with constant Hessian, the Hessian is computed once and
    reused across iterations; Armijo backtracking still verifies each step.

    Requires: loss has hessian() and penalty is smooth.
    """
    _validate_smooth_penalty(penalty, "newton_solver")
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    sample_weight = _prepare_newton_sample_weight(
        sample_weight, X_proc.shape[0], backend, X_proc
    )
    n_features = X_proc.shape[1]

    if init_coef is not None:
        params = _as_backend_vector(init_coef, backend, X_proc)
    else:
        params = _zeros(n_features, backend, ref_tensor=X_proc)

    # Constant-Hessian detection via loss attribute (generic, not loss-name based)
    _const_hessian = getattr(loss, "_has_constant_hessian", False)

    _fixed_hess = None
    if _const_hessian:
        _fixed_hess = _call_loss_with_weight(
            loss.hessian,
            X_proc,
            y_proc,
            params,
            sample_weight=sample_weight,
        ) + _smooth_penalty_hessian(penalty, params)

    iteration = -1
    line_search_failed = False

    # Check if loss supports fused gradient+hessian (avoids redundant X@coef)
    _has_fused = hasattr(loss, 'fused_gradient_and_hessian')

    # Pre-allocate ridge matrix (reused every iteration, avoids O(p^2) allocation)
    # Use same dtype as params to avoid dtype mismatch with Hessian
    _n = n_features
    _dtype = getattr(params, 'dtype', np.float64)
    _device = getattr(params, 'device', None)
    if backend == "numpy":
        _ridge = 1e-10 * np.eye(_n, dtype=_dtype)
    elif backend == "cupy":
        import cupy as cp
        _ridge = 1e-10 * cp.eye(_n, dtype=_dtype)
    else:
        import torch
        _ridge = 1e-10 * torch.eye(_n, dtype=_dtype, device=_device)

    for iteration in range(max_iter):
        params_old = _copy_arr(params)

        if _has_fused and _fixed_hess is None:
            # Fused: compute gradient and hessian in one pass
            loss_grad, loss_hess = _call_loss_with_weight(
                loss.fused_gradient_and_hessian,
                X_proc,
                y_proc,
                params,
                sample_weight=sample_weight,
            )
            grad = loss_grad + _smooth_penalty_gradient(penalty, params)
            hess = loss_hess + _smooth_penalty_hessian(penalty, params)
        else:
            loss_grad = _call_loss_with_weight(
                loss.gradient,
                X_proc,
                y_proc,
                params,
                sample_weight=sample_weight,
            )
            grad = loss_grad + _smooth_penalty_gradient(penalty, params)
            hess = _fixed_hess if _fixed_hess is not None else (
                _call_loss_with_weight(
                    loss.hessian,
                    X_proc,
                    y_proc,
                    params,
                    sample_weight=sample_weight,
                )
                + _smooth_penalty_hessian(penalty, params)
            )

        grad_norm_dev = _norm2_dev(grad)
        (grad_norm,) = _sync_scalars(grad_norm_dev, backend=backend)
        if grad_norm <= tol:
            break
        hess = 0.5 * (hess + hess.T)

        # Add small ridge to ensure non-singular Hessian.
        # This is needed for losses like CoxPH where the intercept is
        # not identifiable (Hessian has zero eigenvalue for intercept).
        # The ridge is negligible for well-conditioned problems.
        hess_reg = hess + _ridge

        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess_reg, grad)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.solve(hess_reg, grad)
            else:
                import torch

                direction = torch.linalg.solve(hess_reg, grad.unsqueeze(1))
                direction = direction.squeeze(1)
        except np.linalg.LinAlgError:
            if backend == "numpy":
                direction = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess_reg, grad)[0]
            else:
                import torch

                direction = torch.linalg.lstsq(hess_reg, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)
        except RuntimeError as exc:
            if not _runtime_error_is_singular(exc):
                raise
            if backend == "torch":
                import torch

                direction = torch.linalg.lstsq(hess_reg, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess_reg, grad)[0]
            else:
                direction = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]

        # Armijo backtracking line search. The acceptance objective must use
        # the same analytic weights as the Newton system itself.
        obj_old_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params_old,
            sample_weight=sample_weight,
        )
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
                obj_try_dev, _ = _call_loss_with_weight(
                    loss.fused_value_and_gradient,
                    X_proc,
                    y_proc,
                    params_try,
                    sample_weight=sample_weight,
                )
                obj_try_dev = obj_try_dev + _smooth_penalty_value_dev(
                    penalty, params_try
                )
                if _device_leq(obj_try_dev, obj_old_dev - 1e-4 * step * gdd):
                    params = params_try
                    accepted = True
                    break
            except FloatingPointError:
                pass
            except (ValueError, RuntimeError) as exc:
                if not _trial_error_is_numerical(exc):
                    raise
            step *= 0.5
        if not accepted:
            # Never accept an unverified trial step. A tiny rejected step
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
