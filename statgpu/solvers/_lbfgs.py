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
    _initial_smooth_params,
    _prepare_analytic_sample_weight,
)
from ._utils import (
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
    explicitly opts into the shared weighted-L-BFGS contract. Maintained GLM
    losses do so and evaluate value, gradient, line-search candidates, and the
    accepted iterate under one normalized objective
    ``sum_i w_i * contribution_i / sum_i w_i``. Generic non-GLM losses remain
    fail-closed unless they independently declare the same capability.

    Uniform weights are normalized away using the historical uniformity rule,
    preserving the established unweighted objective. Losses may additionally
    expose private ``_loss_domain_*`` hooks; L-BFGS then validates/generates an
    interior start and caps Armijo using the final post-fallback search
    direction without evaluating infeasible candidates.
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
        if gdd >= 0 or not np.isfinite(gdd):
            direction = -grad
            gdd = -gn * gn  # grad'(-grad) = -||grad||^2

        # Freeze the final post-fallback additive direction before obtaining a
        # loss-domain cap. A cap for a discarded quasi-Newton direction is not
        # a valid feasibility certificate for the actual line search.
        domain_cap = _domain_max_step(
            loss,
            X_proc,
            params,
            direction,
            sample_weight=sample_weight,
        )

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

        step = min(1.0, domain_cap) if domain_cap is not None else 1.0
        params_new = params
        _ls_accepted = False
        evaluated_domain_trial = False
        rejected_by_domain = False
        for _ in range(25):
            candidate = params + step * direction
            if not _domain_feasible(
                loss, X_proc, candidate, sample_weight=sample_weight
            ):
                rejected_by_domain = True
                step *= 0.5
                continue
            evaluated_domain_trial = True
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
            if rejected_by_domain and not evaluated_domain_trial:
                raise _LossDomainError(
                    "lbfgs_solver could not evaluate a numerically interior "
                    "trial step for the maintained loss domain."
                )

            # The normal convergence rule below accepts an *accepted* step with
            # ||s_k|| < tol. At floating-point resolution Armijo may instead
            # reject that same tiny displacement because the requested decrease
            # is no longer representable in the objective. Keep those two cases
            # consistent: if the smallest step that was actually tried is
            # already below the configured parameter-step tolerance, terminate
            # through the existing small-step criterion rather than reporting a
            # line-search failure. Genuine exhaustion at a material step still
            # emits the warning.
            direction_norm_dev = _norm2_dev(direction)
            (direction_norm,) = _sync_scalars(
                direction_norm_dev, backend=backend
            )
            smallest_tried_step = 2.0 * step
            numerically_small_trial = (
                smallest_tried_step * direction_norm < tol
            )
            if not numerically_small_trial:
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
