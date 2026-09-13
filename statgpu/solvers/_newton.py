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
    _device_leq,
)
from statgpu.backends._utils import _to_float_scalar

from ._convergence import ConvergenceWarning
from ._smooth_domain import (
    _LossDomainError,
    _domain_feasible,
    _domain_max_step,
    _initial_smooth_params,
    _prepare_analytic_sample_weight,
)
from ._utils import (
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _smooth_penalty_value_dev,
    _runtime_error_is_singular,
    _validate_smooth_penalty,
    _trial_error_is_numerical,
)


def _prepare_newton_sample_weight(sample_weight, n_samples, backend, ref_arr):
    """Compatibility alias for the shared smooth-solver weight preparation."""
    return _prepare_analytic_sample_weight(
        sample_weight, n_samples, backend, ref_arr
    )


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

    Losses may additionally expose private ``_loss_domain_*`` hooks.  When
    present, Newton obtains or validates an interior start before the first
    derivative evaluation, caps Armijo using the final post-fallback additive
    search direction, and never evaluates an infeasible trial point.

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

    params = _initial_smooth_params(
        loss,
        X_proc,
        y_proc,
        backend=backend,
        n_features=n_features,
        init_coef=init_coef,
        sample_weight=sample_weight,
    )

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

        # Armijo uses the same weighted objective as the Newton system.
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

        # Freeze the final direction *after* all fallbacks, then ask the loss
        # for an interior cap for the actual additive update Delta=-direction.
        additive_direction = -direction
        domain_cap = _domain_max_step(
            loss,
            X_proc,
            params_old,
            additive_direction,
            sample_weight=sample_weight,
        )
        step = min(1.0, domain_cap) if domain_cap is not None else 1.0

        accepted = False
        evaluated_domain_trial = False
        rejected_by_domain = False
        for _bt in range(20):
            params_try = params_old + step * additive_direction
            if not _domain_feasible(
                loss, X_proc, params_try, sample_weight=sample_weight
            ):
                rejected_by_domain = True
                step *= 0.5
                continue
            evaluated_domain_trial = True
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
            if rejected_by_domain and not evaluated_domain_trial:
                raise _LossDomainError(
                    "newton_solver could not evaluate a numerically interior "
                    "trial step for the maintained loss domain."
                )
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