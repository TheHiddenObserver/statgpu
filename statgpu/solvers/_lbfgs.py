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
)

from ._convergence import ConvergenceWarning
from ._smooth_domain import (
    _LossDomainError,
    _domain_feasible,
    _domain_max_step,
    _floating_eps,
    _initial_smooth_params,
    _prepare_analytic_sample_weight,
)
from ._utils import (
    _external_warning_stacklevel,
    _smooth_penalty_gradient,
    _smooth_penalty_value_dev,
    _validate_smooth_penalty,
)


def _prepare_lbfgs_sample_weight(sample_weight, n_samples, backend, ref_arr, loss):
    """Prepare analytic weights and enforce the L-BFGS capability gate."""
    values = _prepare_analytic_sample_weight(
        sample_weight, n_samples, backend, ref_arr
    )
    if values is None:
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


def _domain_step_or_raise(loss, X, params, direction, tol, sample_weight, backend):
    """Return the loss-owned cap and fail when no material domain step remains."""
    domain_cap = _domain_max_step(
        loss,
        X,
        params,
        direction,
        sample_weight=sample_weight,
    )
    if domain_cap is None:
        return None
    direction_norm_dev = _norm2_dev(direction)
    (direction_norm,) = _sync_scalars(direction_norm_dev, backend=backend)
    if domain_cap * direction_norm <= tol:
        raise _LossDomainError(
            f"loss='{getattr(loss, 'name', '?')}' is pinned to the "
            "smooth-domain boundary before gradient convergence."
        )
    return domain_cap


def _armijo_domain_search(
    loss,
    penalty,
    X,
    y,
    params,
    direction,
    old_val_dev,
    gdd,
    domain_cap,
    *,
    sample_weight,
    objective_roundoff,
    direction_norm,
    parameter_tol,
):
    """Run one Armijo search, with a bounded floating-point resolution rule.

    The exact Armijo condition remains authoritative whenever its requested
    decrease is numerically resolvable. A roundoff-limited trial may be accepted
    only when both the requested decrease is below the objective's floating-
    point resolution and the actual parameter displacement is below the solver
    tolerance. The candidate objective must also be non-increasing up to that
    same roundoff scale. This prevents a large bad direction with tiny
    directional derivative from being mislabeled as numerical convergence.
    """
    step = min(1.0, domain_cap) if domain_cap is not None else 1.0
    evaluated_domain_trial = False
    rejected_by_domain = False
    for _ in range(25):
        candidate = params + step * direction
        if not _domain_feasible(
            loss, X, candidate, sample_weight=sample_weight
        ):
            rejected_by_domain = True
            step *= 0.5
            continue
        evaluated_domain_trial = True
        cand_val_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X,
            y,
            candidate,
            sample_weight=sample_weight,
        )
        cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(penalty, candidate)
        if _device_leq(cand_val_dev, old_val_dev + 1e-4 * step * gdd):
            return candidate, True, step, evaluated_domain_trial, rejected_by_domain

        required_decrease = max(0.0, -1e-4 * step * gdd)
        parameter_displacement = step * direction_norm
        if (
            required_decrease <= objective_roundoff
            and parameter_displacement <= parameter_tol
            and _device_leq(cand_val_dev, old_val_dev + objective_roundoff)
        ):
            return candidate, True, step, evaluated_domain_trial, rejected_by_domain
        step *= 0.5
    return params, False, step, evaluated_domain_trial, rejected_by_domain


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
    returning ``(value, gradient)``. Supports numpy / cupy / torch backends
    via auto-detection of *X*.

    Genuine non-uniform ``sample_weight`` is supported only when the loss
    explicitly opts into the shared weighted-L-BFGS contract. Built-in GLM
    losses that advertise this capability evaluate value, gradient, line-search
    candidates, and the accepted iterate under one normalized objective
    ``sum_i w_i * contribution_i / sum_i w_i``. Generic non-GLM losses reject
    non-uniform weights unless they independently declare the same capability.

    Uniform weights are normalized away using the historical uniformity rule,
    preserving the established unweighted objective. Losses may additionally
    expose private ``_loss_domain_*`` hooks; L-BFGS then validates/generates an
    interior start and caps Armijo using the final post-fallback search
    direction without evaluating infeasible candidates. If a quasi-Newton
    direction exhausts Armijo inside a declared loss domain, the solver retries
    once with steepest descent and a freshly computed domain cap; failure of
    that recovery path is a hard domain error rather than a publishable fit.
    """
    _validate_smooth_penalty(penalty, "lbfgs_solver")
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    sample_weight = _prepare_lbfgs_sample_weight(
        sample_weight, X_proc.shape[0], backend, X_proc, loss
    )

    params = _initial_smooth_params(
        loss,
        X_proc,
        y_proc,
        backend=backend,
        n_features=n_features,
        init_coef=init_coef,
        sample_weight=sample_weight,
    )

    s_hist = []
    y_hist = []
    rho_hist = []

    _init_val_dev, grad = _call_loss_with_weight(
        loss.fused_value_and_gradient,
        X_proc,
        y_proc,
        params,
        sample_weight=sample_weight,
    )
    grad = grad + _smooth_penalty_gradient(penalty, params)

    iteration = -1

    for iteration in range(max_iter):
        grad_norm_dev = _norm2_dev(grad)

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

        gn, gdd = _sync_scalars(grad_norm_dev, gdd_dev, backend=backend)
        if gn < tol:
            break
        if gdd >= 0 or not np.isfinite(gdd):
            direction = -grad
            gdd = -gn * gn

        direction_norm_dev = _norm2_dev(direction)
        (direction_norm,) = _sync_scalars(direction_norm_dev, backend=backend)

        domain_cap = _domain_step_or_raise(
            loss,
            X_proc,
            params,
            direction,
            tol,
            sample_weight,
            backend,
        )

        old_val_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params,
            sample_weight=sample_weight,
        )
        old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)
        (old_val,) = _sync_scalars(old_val_dev, backend=backend)
        objective_roundoff = (
            64.0 * _floating_eps(old_val_dev) * max(1.0, abs(old_val))
        )

        (
            params_new,
            _ls_accepted,
            step,
            evaluated_domain_trial,
            rejected_by_domain,
        ) = _armijo_domain_search(
            loss,
            penalty,
            X_proc,
            y_proc,
            params,
            direction,
            old_val_dev,
            gdd,
            domain_cap,
            sample_weight=sample_weight,
            objective_roundoff=objective_roundoff,
            direction_norm=direction_norm,
            parameter_tol=tol,
        )

        if not _ls_accepted and domain_cap is not None:
            # A maintained loss domain is a fail-closed path. Before declaring
            # non-convergence, discard quasi-Newton history for this step and
            # retry the mathematically valid steepest-descent direction with a
            # newly computed cap. This salvages recoverable curvature-history
            # failures without publishing an unverified boundary iterate.
            direction = -grad
            gdd = -gn * gn
            direction_norm = gn
            domain_cap = _domain_step_or_raise(
                loss,
                X_proc,
                params,
                direction,
                tol,
                sample_weight,
                backend,
            )
            (
                params_new,
                _ls_accepted,
                step,
                fallback_evaluated,
                fallback_rejected,
            ) = _armijo_domain_search(
                loss,
                penalty,
                X_proc,
                y_proc,
                params,
                direction,
                old_val_dev,
                gdd,
                domain_cap,
                sample_weight=sample_weight,
                objective_roundoff=objective_roundoff,
                direction_norm=direction_norm,
                parameter_tol=tol,
            )
            evaluated_domain_trial = evaluated_domain_trial or fallback_evaluated
            rejected_by_domain = rejected_by_domain or fallback_rejected
            if not _ls_accepted:
                if rejected_by_domain and not evaluated_domain_trial:
                    raise _LossDomainError(
                        "lbfgs_solver could not evaluate a numerically interior "
                        "trial step for the declared loss domain."
                    )
                raise _LossDomainError(
                    "lbfgs_solver Armijo line search failed inside the declared "
                    "loss domain before gradient convergence."
                )

        if not _ls_accepted:
            if rejected_by_domain and not evaluated_domain_trial:
                raise _LossDomainError(
                    "lbfgs_solver could not evaluate a numerically interior "
                    "trial step for the declared loss domain."
                )

            warnings.warn(
                "lbfgs_solver: line search failed to find a descent step "
                f"after 25 backtracking steps (iteration {iteration}). "
                "Solver may stagnate.",
                RuntimeWarning,
                stacklevel=(
                    _external_warning_stacklevel()
                    if str(getattr(loss, "name", "") or "").lower() == "quantile"
                    else 2
                ),
            )
            break

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
            if domain_cap is None:
                break
            grad_new_norm_dev = _norm2_dev(grad_new)
            (grad_new_norm,) = _sync_scalars(
                grad_new_norm_dev, backend=backend
            )
            if grad_new_norm < tol:
                break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"lbfgs_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=(
                _external_warning_stacklevel()
                if str(getattr(loss, "name", "") or "").lower() == "quantile"
                else 2
            ),
        )
    return params, n_iter
