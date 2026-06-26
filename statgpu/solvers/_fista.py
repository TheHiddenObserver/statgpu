"""FISTA solver with backtracking line search.

minimize: loss(X, y, w) + penalty(w)

Supports numpy / cupy / torch backends via auto-detection.
"""

from __future__ import annotations

__all__ = ["fista_solver"]

import warnings
import numpy as np
from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends._array_ops import (
    _abs_sum_dev,
    _clip_grad_on_device,
    _copy_arr,
    _dot_dev,
    _norm2_dev,
    _sum_sq_dev,
    _sync_scalars,
    _zeros,
)
from ._convergence import ConvergenceWarning
from ._constants import (
    _SLACK_TOLERANCE,
    _DIVERGE_COEF_NORM_CAP,
    _LIPSCHITZ_SAFETY_LOGISTIC_CV,
    _GRAD_CLIP_COEF_FACTOR,
    _GRAD_CLIP_ABS_FLOOR,
    _GRAD_CLIP_MAX,
)
from ._utils import (
    _validate_sample_weight,
    _as_backend_vector,
    _call_with_weight,
    _nesterov_update,
    _penalty_name,
    _smooth_penalty_lipschitz,
    _abs_mean_max,
    _tracking_penalty_value,
)


def fista_solver(
    loss,
    penalty,
    X,
    y,
    max_iter: int = 1000,
    tol: float = 1e-4,
    init_coef=None,
    sample_weight=None,
    lipschitz_L: float | None = None,
    cv_mode: bool = False,
) -> tuple:
    """General FISTA solver with backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Parameters
    ----------
    loss : LossBase
        Loss function with gradient(), lipschitz(), preprocess(), value().
    penalty : Penalty
        Penalty with proximal().
    X : array
        Design matrix (numpy/cupy/torch).
    y : array
        Target (numpy/cupy/torch).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    init_coef : array, optional
        Initial coefficient vector.
    sample_weight : array, optional
        Per-sample weights. Non-uniform weights are currently rejected in this
        solver path to avoid silently running an incorrect unweighted update.
    cv_mode : bool, default=False
        Private CV fast path: keeps the same update rule but checks objective
        and convergence less often on GPU non-smooth GLM paths.

    Returns
    -------
    coef : array
        Fitted coefficients (same backend as X).
    n_iter : int
        Number of iterations.
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    _is_quadratic = getattr(loss, '_is_quadratic', False)
    # Momentum control via loss class attributes:
    #   _momentum_beta_cap: if set, cap Nesterov beta at this value
    #   _skip_momentum: if True, disable momentum entirely
    # Conservative momentum (cap beta at 0.5) for exp-link families and
    # for logistic/gamma with non-smooth penalties.  Logistic/gamma with
    # smooth penalties (none, l2) benefit from full Nesterov acceleration.
    _momentum_beta_cap = getattr(loss, '_momentum_beta_cap', None)
    _skip_momentum = getattr(loss, '_skip_momentum', False)

    n_features = X_proc.shape[1]
    if init_coef is not None:
        coef = _as_backend_vector(init_coef, backend, X)
    else:
        coef = _zeros(n_features, backend, ref_tensor=X)

    y_k = _copy_arr(coef)
    t_k = 1.0

    # Divergence detection: track best objective for recovery
    _obj_best_fista = float('inf')
    _coef_best_fista = None

    # Objective stability tracking for adaptive penalty convergence
    _obj_prev_f = float('inf')
    _obj_stable_count = 0

    # Initial Lipschitz: default to zero (safe for exp-link warm starts),
    # but allow losses to request evaluation at the provided init to avoid
    # degenerate curvature from eta=0 clipping.
    _cached_XtWX_weighted = None  # populated in Lipschitz block, used in GPU loop
    if lipschitz_L is not None and lipschitz_L > 0:
        L = lipschitz_L
    else:
        if getattr(loss, '_lipschitz_at_init', False):
            _lip_coef = _copy_arr(coef)
        else:
            _lip_coef = _zeros(n_features, backend, ref_tensor=X)
        if sample_weight is not None:
            # Weighted Lipschitz: eigenvalue of X' diag(w) X / sum(w)
            _xp_mod = _get_xp(backend)
            # Ensure sample_weight is on same backend as X_proc
            _sw_np = _to_numpy(sample_weight)
            _sw = _xp_mod.asarray(_sw_np, dtype=X_proc.dtype)
            sw_sum = _to_float_scalar(_xp_mod.sum(_sw))
            sw_col = _sw[:, None] if _sw.ndim == 1 else _sw
            XtWX = X_proc.T @ (X_proc * sw_col) / sw_sum
            L = _to_float_scalar(_xp_mod.max(_xp_mod.diag(XtWX)))  # conservative bound
            if L <= 0:
                L = 1.0
            # Cache for periodic recomputation in the loop (X and weights are constant)
            _cached_XtWX_weighted = XtWX
        else:
            L = loss.lipschitz(X_proc, _lip_coef, y=y_proc)
            _cached_XtWX_weighted = None
    if L <= 0:
        L = 1.0
    # Add smooth penalty Lipschitz contribution (e.g. l2 penalty gradient
    # alpha*coef has Lipschitz constant alpha).  Without this, the step
    # size 1/L is too large, causing oscillation near the optimum.
    _smooth_lip = _smooth_penalty_lipschitz(penalty)
    if _smooth_lip > 0:
        L = L + _smooth_lip
    # For GLM losses with exp link (Poisson, etc.), mu at coef=0
    # is ~1, but mu near the optimum ≈ y.  Scale Lipschitz up by a
    # geometric-mean factor to avoid oversized first steps that cause
    # divergence on non-smooth penalties (scad, mcp, etc.).
    # Logistic now uses iterate-dependent Lipschitz, so y-scaling applies.
    # Gamma's expected Fisher Hessian X'X/n underestimates
    # true curvature by ~mean(y), so y-scaling IS needed.
    _skip_y_scaling = getattr(loss, '_lipschitz_uses_y', False)
    _y_scale = 1.0  # default; overridden below for families that need it
    if not _is_quadratic and not _skip_y_scaling:
        _y_mean, _y_max = _abs_mean_max(y_proc, backend)
        _y_scale = max(1.0, _y_mean, np.sqrt(_y_mean * _y_max))
        if _y_scale > 1.0:
            L = L * _y_scale

    # Loss-specific Lipschitz safety factors (from loss class attributes)
    _lip_safety = getattr(loss, '_lipschitz_safety', 1.0)
    if _lip_safety > 1.0:
        L = L * _lip_safety
    # Additional safety for CV mode (from loss class attribute)
    _lip_safety_cv = getattr(loss, '_lipschitz_safety_cv', _LIPSCHITZ_SAFETY_LOGISTIC_CV if cv_mode else 1.0)
    if cv_mode and _lip_safety_cv > 1.0:
        L = L * _lip_safety_cv
    # Async GPU loop: skip backtracking, deferred checks.
    # For non-smooth penalties (l1, elasticnet, scad, mcp, adaptive, group):
    #   - Quadratic losses (squared_error): Lipschitz is exact, fixed step is optimal
    #   - GLM losses: use 3x safety factor on Lipschitz, no backtracking
    # Smooth penalties (l2, none) need backtracking for GLM losses.
    n_samples = X_proc.shape[0]
    _pen_name_lower = _penalty_name(penalty)
    _non_smooth = _pen_name_lower not in ("none", "null", "l2", "")
    _gpu_excluded = getattr(loss, '_gpu_loop_excluded', False) and not cv_mode
    # Async GPU loop: skip backtracking, use fixed step size.
    # For squared_error + non-smooth penalties, Lipschitz is exact → no backtracking needed.
    # For GLM losses, only enabled in CV mode (backtracking needed for safety).
    _use_gpu_loop = (
        backend in ("torch", "cupy")
        and _non_smooth
        and (cv_mode or _is_quadratic)
        and not _gpu_excluded
    )
    _is_gpu = backend in ("torch", "cupy")
    _conv_interval = 3
    _div_interval = 5
    _lip_interval = 5
    if _use_gpu_loop:
        _conv_interval = 10
        _div_interval = 25
        _lip_interval = 25
    _validate_sample_weight(sample_weight, X_proc.shape[0])

    # Gram matrix optimization for squared_error on async GPU path only.
    # Precompute X'X/n and X'y/n to avoid redundant X@coef per iteration.
    _use_xtx = _is_quadratic and sample_weight is None and _use_gpu_loop
    if _use_xtx:
        _xp_mod = _get_xp(backend)
        XtX = X_proc.T @ X_proc / n_samples
        Xty = X_proc.T @ y_proc / n_samples
    else:
        XtX = None
        Xty = None

    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Compute gradient
        if _use_xtx and XtX is not None:
            # Gram matrix path: single matmul instead of X@coef + X.T@resid
            # XtX = X'X/n, Xty = X'y/n, so grad = XtX @ w - Xty = X'(Xw-y)/n
            grad = XtX @ y_k - Xty
            q_yk_dev = loss.value(X_proc, y_proc, y_k)
        elif sample_weight is not None:
            q_yk_dev, grad = loss.fused_value_and_gradient(
                X_proc, y_proc, y_k, sample_weight=sample_weight
            )
        else:
            q_yk_dev, grad = loss.fused_value_and_gradient(X_proc, y_proc, y_k)

        if _use_gpu_loop:
            # -- GPU async path: all ops stay on device --
            grad = _clip_grad_on_device(grad, coef_old, backend)

            step = 1.0 / L

            # Single proximal step with one Armijo retry on device.
            # First try step = 1/L; if objective increases, halve and retry.
            # This keeps GPU↔CPU syncs minimal (only one extra proximal+loss
            # call in the worst case) while matching CPU path behavior.
            w_tilde = y_k - step * grad
            coef = penalty.proximal(w_tilde, step, backend=backend)

            # GPU path: single proximal step (no backtracking).
            # Backtracking on GPU requires loss.value() + GPU→CPU sync per
            # retry, which defeats the purpose of the async GPU path.
            # The Lipschitz constant L is conservative enough that step=1/L
            # almost always satisfies the Armijo condition on the first try.

            # ALL safety checks deferred -- no per-iteration GPU->CPU sync.
            # Finiteness + divergence + objective tracking batched together.
            if iteration > 0 and (iteration < 20 or iteration % _div_interval == 0):
                _obj_dev = loss.value(X_proc, y_proc, coef)
                # Single D2H transfer: extract float, then check finiteness.
                _obj_val_f = float(_to_numpy(_obj_dev))
                _all_finite = np.isfinite(_obj_val_f)
                if not _all_finite:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    else:
                        coef = _zeros(n_features, backend, ref_tensor=X_proc)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    L = L * 2.0
                    continue
                # Track best objective (reuse _obj_val_f from finiteness check above)
                _obj_val_f += _tracking_penalty_value(penalty, coef)
                if _obj_val_f < _obj_best_fista:
                    _obj_best_fista = _obj_val_f
                    _coef_best_fista = _copy_arr(coef)
                # Convergence check for quadratic losses on GPU:
                # objective stability (adaptive penalties oscillate in coef space)
                if _is_quadratic and iteration > 20:
                    if abs(_obj_val_f - _obj_prev_f) < tol * max(abs(_obj_val_f), 1.0):
                        _obj_stable_count += 1
                        if _obj_stable_count >= 5:
                            break
                    else:
                        _obj_stable_count = 0
                    _obj_prev_f = _obj_val_f
                # Periodic Lipschitz recomputation (piggyback on same sync)
                # Skip for quadratic losses -- Lipschitz is constant (spectral norm of X^T X).
                # Interval matches CPU path for trajectory consistency.
                if not _is_quadratic and iteration % _lip_interval == 0:
                    if sample_weight is not None and _cached_XtWX_weighted is not None:
                        # Use cached weighted Gram matrix (X and weights are constant)
                        _xp_lip = _get_xp(backend)
                        L_new = _to_float_scalar(_xp_lip.max(_xp_lip.diag(_cached_XtWX_weighted)))
                    else:
                        L_new = loss.lipschitz(X_proc, coef, y=y_proc)
                    if L_new > 0:
                        # Re-apply y-scaling (Lipschitz at current coef may not
                        # capture the y-dependent curvature scaling applied at init)
                        if _y_scale > 1.0:
                            L_new = L_new * _y_scale
                        _safety = getattr(loss, '_lipschitz_safety', 1.0)
                        L_new *= _safety
                        if _smooth_lip > 0:
                            L_new = L_new + _smooth_lip
                        if L_new > L:
                            L = L_new
                        else:
                            L = max(L * 0.8, L_new)


        else:
            # -- CPU/GPU path with backtracking (smooth penalties) --
            # GPU optimization: batch ALL checks into ONE sync per iteration.

            step = 1.0 / L
            _q_new_dev_last = None

            # Backtracking line search
            _bt_accepted = False
            for _bt in range(20):
                w_tilde = y_k - step * grad
                coef_new = penalty.proximal(w_tilde, step, backend=backend)

                diff = coef_new - y_k
                if sample_weight is not None:
                    q_new_dev, _ = loss.fused_value_and_gradient(
                        X_proc, y_proc, coef_new, sample_weight=sample_weight
                    )
                else:
                    q_new_dev = loss.value(X_proc, y_proc, coef_new)
                _q_new_dev_last = q_new_dev
                bound_dev = q_yk_dev + _dot_dev(grad, diff) + 0.5 * L * _sum_sq_dev(diff)
                slack_dev = bound_dev + _SLACK_TOLERANCE - q_new_dev

                if _to_float_scalar(slack_dev) >= 0:
                    _bt_accepted = True
                    break
                L *= 1.5
                step = 1.0 / L

            coef = coef_new

            # ── CPU convergence for quadratic losses ──
            # Quadratic losses (squared_error) on CPU don't enter the GPU
            # loop or the non-quadratic CPU block.  Check objective stability
            # here so adaptive penalties (adaptive_l1) can converge.
            if not _is_gpu and _is_quadratic:
                _obj_val_f = float(loss.value(X_proc, y_proc, coef) if sample_weight is None
                                   else loss.fused_value_and_gradient(X_proc, y_proc, coef, sample_weight=sample_weight)[0])
                _obj_val_f += _tracking_penalty_value(penalty, coef)
                _conv_dev = _abs_sum_dev(coef - coef_old)
                _conv_f = float(_conv_dev)
                if _conv_f < tol:
                    break
                if iteration > 20 and abs(_obj_val_f - _obj_prev_f) < tol * max(abs(_obj_val_f), 1.0):
                    _obj_stable_count += 1
                    if _obj_stable_count >= 5:
                        break
                else:
                    _obj_stable_count = 0
                _obj_prev_f = _obj_val_f

            # ═══ GPU: batch ALL checks into ONE sync every _check_interval iterations ═══
            # Skip checks entirely for early iterations (save syncs)
            if _is_gpu and not _is_quadratic:
                # Always compute convergence metric on device (no sync)
                _conv_dev = _abs_sum_dev(coef - coef_old)

                # Full check (objective + divergence): every _conv_interval
                _do_full_check = (iteration < 20) or (iteration % _conv_interval == 0)
                # Quick convergence check: every 5 iterations (just convergence metric)
                _do_conv_check = (iteration < 20) or (iteration % 5 == 0)
                # Lipschitz recompute: every 5 iterations (same as CPU)
                _do_lip_check = (iteration > 0 and iteration % 5 == 0)

                if _do_full_check:
                    # Compute ALL check values on device first, then ONE sync
                    _obj_dev = _q_new_dev_last if _q_new_dev_last is not None else (
                        loss.fused_value_and_gradient(X_proc, y_proc, coef, sample_weight=sample_weight)[0]
                        if sample_weight is not None else loss.value(X_proc, y_proc, coef)
                    )
                    _q_new_dev_last = None

                    # ONE sync: (objective, coef_norm, coef_change_norm)
                    _need_norm = (iteration > 10)
                    if _need_norm:
                        _obj_val_f, _coef_norm_f, _conv_f = _sync_scalars(
                            _obj_dev, _norm2_dev(coef), _conv_dev, backend=backend
                        )
                    else:
                        _obj_val_f, _conv_f = _sync_scalars(
                            _obj_dev, _conv_dev, backend=backend
                        )
                        _coef_norm_f = 0.0

                    _finite_ok = np.isfinite(_obj_val_f)

                    # Finiteness
                    if not _finite_ok:
                        if _coef_best_fista is not None:
                            coef = _copy_arr(_coef_best_fista)
                            y_k = _copy_arr(coef); t_k = 1.0; L *= 2.0
                            continue

                    # Divergence
                    _obj_val_f += _tracking_penalty_value(penalty, coef)
                    _diverged_f = False
                    if not np.isfinite(_obj_val_f):
                        _diverged_f = True
                    elif _obj_best_fista > 1e-8:
                        _diverged_f = _obj_val_f > _obj_best_fista * 10.0 + 1e-8
                    else:
                        _diverged_f = _obj_val_f > _obj_best_fista + max(abs(_obj_best_fista) * 10.0, 1.0)
                    if not _diverged_f and _need_norm and _coef_norm_f > _DIVERGE_COEF_NORM_CAP:
                        _diverged_f = True
                    if _diverged_f:
                        if _coef_best_fista is not None:
                            coef = _copy_arr(_coef_best_fista)
                        else:
                            coef = _zeros(n_features, backend, ref_tensor=X_proc)
                        y_k = _copy_arr(coef); t_k = 1.0; L *= 2.0
                        continue
                    elif _obj_val_f < _obj_best_fista:
                        _obj_best_fista = _obj_val_f
                        _coef_best_fista = _copy_arr(coef)

                    # Convergence: coefficient change OR objective stability
                    # Adaptive penalties (adaptive_l1, group_scad, etc.) cause
                    # coefficients to oscillate near the optimum, so coef_diff
                    # never reaches tol.  Check objective stability as fallback.
                    if _conv_f < tol:
                        break
                    if iteration > 20 and abs(_obj_val_f - _obj_prev_f) < tol * max(abs(_obj_val_f), 1.0):
                        _obj_stable_count += 1
                        if _obj_stable_count >= 5:
                            break
                    else:
                        _obj_stable_count = 0
                    _obj_prev_f = _obj_val_f

                    # Periodic Lipschitz recompute
                    if _do_lip_check:
                        _relative_change = _conv_f / max(_coef_norm_f, 1e-10) if _need_norm else 1.0
                        if _relative_change > 1e-3:
                            L_new = _call_with_weight(loss.lipschitz, X_proc, coef, y=y_proc, sample_weight=sample_weight)
                            _lip_safety_recomp = getattr(loss, '_lipschitz_safety', 1.0)
                            if _lip_safety_recomp > 1.0:
                                L_new *= _lip_safety_recomp
                            if _smooth_lip > 0:
                                L_new += _smooth_lip
                            L = max(L, L_new) if L_new > L else max(L * 0.8, L_new)

                # Quick convergence check (just sync convergence metric)
                elif _do_conv_check:
                    _conv_f = _to_float_scalar(_conv_dev)
                    if _conv_f < tol:
                        break

                    # Lipschitz recompute (reuse convergence sync)
                    if _do_lip_check:
                        _coef_norm_f = _to_float_scalar(_norm2_dev(coef))
                        _relative_change = _conv_f / max(_coef_norm_f, 1e-10)
                        if _relative_change > 1e-3:
                            L_new = _call_with_weight(loss.lipschitz, X_proc, coef, y=y_proc, sample_weight=sample_weight)
                            _lip_s = getattr(loss, '_lipschitz_safety', 1.0)
                            if _lip_s > 1.0: L_new *= _lip_s
                            if _smooth_lip > 0: L_new += _smooth_lip
                            L = max(L, L_new) if L_new > L else max(L * 0.8, L_new)

            elif not _is_quadratic:
                # CPU path: check every iteration
                if _q_new_dev_last is not None:
                    _obj_dev = _q_new_dev_last; _q_new_dev_last = None
                else:
                    _obj_dev = loss.value(X_proc, y_proc, coef) if sample_weight is None else \
                        loss.fused_value_and_gradient(X_proc, y_proc, coef, sample_weight=sample_weight)[0]
                _obj_val_f = float(_obj_dev) + _tracking_penalty_value(penalty, coef)
                _conv_dev = _abs_sum_dev(coef - coef_old)
                _conv_f = float(_conv_dev)

                if not np.isfinite(_obj_val_f):
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    y_k = _copy_arr(coef); t_k = 1.0; L *= 2.0; continue

                _diverged_f = False
                if _obj_best_fista > 1e-8:
                    _diverged_f = _obj_val_f > _obj_best_fista * 10.0 + 1e-8
                else:
                    _diverged_f = _obj_val_f > _obj_best_fista + max(abs(_obj_best_fista) * 10.0, 1.0)
                if _diverged_f:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    else:
                        coef = _zeros(n_features, backend, ref_tensor=X_proc)
                    y_k = _copy_arr(coef); t_k = 1.0; L *= 2.0; continue
                elif _obj_val_f < _obj_best_fista:
                    _obj_best_fista = _obj_val_f
                    _coef_best_fista = _copy_arr(coef)

                # Convergence: coefficient change OR objective stability
                if _conv_f < tol:
                    break
                if iteration > 20 and abs(_obj_val_f - _obj_prev_f) < tol * max(abs(_obj_val_f), 1.0):
                    _obj_stable_count += 1
                    if _obj_stable_count >= 5:
                        break
                else:
                    _obj_stable_count = 0
                _obj_prev_f = _obj_val_f

                # Lipschitz recompute
                if iteration > 0 and iteration % 5 == 0:
                    _coef_change_f = float(_abs_sum_dev(coef - coef_old))
                    _coef_norm_f = float(_norm2_dev(coef))
                    if _coef_change_f / max(_coef_norm_f, 1e-10) > 1e-3:
                        L_new = _call_with_weight(loss.lipschitz, X_proc, coef, y=y_proc, sample_weight=sample_weight)
                        _lip_s = getattr(loss, '_lipschitz_safety', 1.0)
                        if _lip_s > 1.0: L_new *= _lip_s
                        if _smooth_lip > 0: L_new += _smooth_lip
                        L = max(L, L_new) if L_new > L else max(L * 0.8, L_new)

        # Momentum update -- all backends
        if _skip_momentum:
            # No momentum (e.g. inverse_gaussian): just copy coef
            y_k = _copy_arr(coef)
        elif _momentum_beta_cap is not None:
            # Conservative momentum with capped beta
            y_k, t_k = _nesterov_update(coef, coef_old, t_k, beta_cap=_momentum_beta_cap)
        else:
            y_k, t_k = _nesterov_update(coef, coef_old, t_k)

        # Note: convergence check is now inside the batched sync above (GPU)
        # or inside the CPU path (CPU). No duplicate check needed.

    # Return best iterate if available
    if _coef_best_fista is not None:
        coef = _copy_arr(_coef_best_fista)

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"fista_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}). "
            f"Consider increasing max_iter or using a different solver (newton, lbfgs, irls).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return coef, n_iter
