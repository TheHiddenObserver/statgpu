"""
Unified solvers for GLMLoss + Penalty optimization.

minimize: loss(X, y, w) + penalty(w)

Supports numpy / cupy / torch backends via auto-detection.
"""

__all__ = ["ConvergenceWarning", "fista_solver", "fista_lla_path", "fista_bb_solver", "newton_solver", "lbfgs_solver", "admm_solver"]


import copy
import warnings
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends._array_ops import (
    _abs_sum,
    _abs_sum_dev,
    _clip_grad_on_device,
    _copy_arr,
    _device_gt,
    _device_leq,
    _dot,
    _dot_dev,
    _eye_like,
    _norm2,
    _norm2_dev,
    _sum_sq,
    _sum_sq_dev,
    _sync_scalars,
    _zeros,
    _zeros_like,
)

# Import shared helpers from _solver_utils
from statgpu.glm_core._solver_utils import (
    ConvergenceWarning,
    _LIPSCHITZ_SAFETY_LOGISTIC_CV,
    _SLACK_TOLERANCE,
    _DIVERGE_COEF_NORM_CAP,
    _DIVERGE_OBJ_RATIO,
    _DIVERGE_OBJ_ABS,
    _BB_RESTART_DOT_TOL,
    _LIPSCHITZ_FLOOR,
    _get_fista_step_compiled,
    _fista_step_call,
    _get_newton_step_compiled,
    _newton_step_call,
    _validate_uniform_sample_weight,
    _validate_sample_weight,
    _as_backend_vector,
    _penalty_name,
    _smooth_penalty_value,
    _tracking_penalty_value,
    _abs_mean_max,
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _objective_value,
    _objective_gradient,
    _smooth_penalty_lipschitz,
    _smooth_penalty_value_dev,
    _objective_value_dev,
    _fused_logistic,
    _fused_poisson,
    _fused_gamma,
    _fused_negative_binomial,
    _fused_tweedie,
    _fused_inverse_gaussian,
    _fused_glm_value_and_gradient,
    _weighted_loss_and_grad,
)


# =============================================================================
# torch.compile for FISTA/Newton elementwise ops
# Falls back to eager mode on GPUs with CUDA capability < 7.0
# =============================================================================


def fista_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=1000,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    lipschitz_L=None,
    cv_mode=False,
):
    """General FISTA solver with backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Parameters
    ----------
    loss : GLMLoss
        GLM loss function with gradient(), lipschitz(), preprocess(), value().
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
    _loss_name = getattr(loss, 'name', '')
    _is_quadratic = (_loss_name == "squared_error")
    # Exp-link families where Nesterov extrapolation can cause mu = exp(X@w)
    # to explode (Poisson: grad ~ mu) or vanish into extreme oscillation
    # (Inverse Gaussian: grad ~ 1/mu^3).  Gamma and Tweedie have self-
    # stabilizing 1/mu-type gradient scaling and are safe with momentum.
    # Disable momentum entirely for inverse_gaussian (1/mu^3 scaling).
    # Use conservative momentum for Poisson and negative_binomial
    # (exp-link families where Nesterov can cause mu explosion).
    # Losses that disable momentum entirely (empty = none currently).
    # See _conservative_momentum below for the active momentum control.
    # _NO_MOMENTUM_LOSSES removed — momentum control via _conservative_momentum
    # Conservative momentum (cap beta at 0.5) for exp-link families and
    # for logistic/gamma with non-smooth penalties.  Logistic/gamma with
    # smooth penalties (none, l2) benefit from full Nesterov acceleration.
    _non_smooth_pen = getattr(penalty, 'name', '') in (
        "l1", "elasticnet", "en", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    )
    _conservative_momentum = (
        _loss_name in ("poisson", "negative_binomial", "tweedie", "inverse_gaussian")
        or (_loss_name in ("logistic", "gamma") and _non_smooth_pen)
    )

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

    # Initial Lipschitz: default to zero (safe for exp-link warm starts),
    # but allow losses to request evaluation at the provided init to avoid
    # degenerate curvature from eta=0 clipping.
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
    # Additional safety for CV mode with logistic
    if _loss_name == "logistic" and cv_mode:
        L = L * _LIPSCHITZ_SAFETY_LOGISTIC_CV
    # Async GPU loop: skip backtracking, deferred checks.
    # For non-smooth penalties (l1, elasticnet, scad, mcp, adaptive, group):
    #   - Quadratic losses (squared_error): Lipschitz is exact, fixed step is optimal
    #   - GLM losses: use 3x safety factor on Lipschitz, no backtracking
    # Smooth penalties (l2, none) need backtracking for GLM losses.
    _pen_name_lower = _penalty_name(penalty)
    _non_smooth = _pen_name_lower not in ("none", "null", "l2", "")
    # Logistic: keep Armijo for non-CV mode (CPU/GPU path parity).
    # In CV mode, allow async GPU loop with conservative Lipschitz.
    _logistic_excluded = _loss_name == "logistic" and not cv_mode
    _use_gpu_loop = (
        backend in ("torch", "cupy")
        and cv_mode
        and _non_smooth
        and not _logistic_excluded
    )
    _is_gpu = backend in ("torch", "cupy")
    _conv_interval = 1 if _loss_name == "logistic" and not _use_gpu_loop else 3
    _div_interval = 5   # check divergence every N iterations (GPU path)
    _lip_interval = 5
    if cv_mode and _use_gpu_loop:
        _conv_interval = max(_conv_interval, 10)
        _div_interval = 25
        _lip_interval = 25
    _validate_sample_weight(sample_weight, X_proc.shape[0])

    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Compute gradient (fused value+gradient for GLM losses)
        if sample_weight is not None:
            q_yk_dev, grad = _weighted_loss_and_grad(loss, X_proc, y_proc, y_k, sample_weight)
        else:
            q_yk_dev, grad = _fused_glm_value_and_gradient(loss, X_proc, y_proc, y_k)

        if _use_gpu_loop:
            # ── GPU async path: all ops stay on device ──
            grad = _clip_grad_on_device(grad, coef_old, backend)

            step = 1.0 / L

            # Single proximal step — no backtracking (L is conservative enough)
            w_tilde = y_k - step * grad
            coef = penalty.proximal(w_tilde, step, backend=backend)

            # ALL safety checks deferred — no per-iteration GPU→CPU sync.
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
                # Periodic Lipschitz recomputation (piggyback on same sync)
                # Skip for quadratic losses — Lipschitz is constant (spectral norm of X^T X).
                # Interval matches CPU path (line 929) for trajectory consistency.
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
            # ── CPU/GPU path with backtracking (smooth penalties) ──
            # Use identical sync-based clipping for both CPU and GPU.
            # (Backtracking already syncs every iteration for slack check,
            #  so on-device clipping has no performance benefit here.)
            _gn_f, _coef_abs_f = _sync_scalars(
                _norm2_dev(grad), _abs_sum_dev(coef_old), backend=backend)
            _gmax = max(_coef_abs_f * 10.0 + 1e3, 1e4)
            if _gn_f > _gmax:
                grad = grad * (_gmax / _gn_f)

            step = 1.0 / L
            _q_new_dev_last = None
            for _bt in range(20):
                w_tilde = y_k - step * grad
                coef_new = penalty.proximal(w_tilde, step, backend=backend)

                diff = coef_new - y_k
                if sample_weight is not None:
                    q_new_dev, _ = _weighted_loss_and_grad(loss, X_proc, y_proc, coef_new, sample_weight)
                else:
                    q_new_dev = loss.value(X_proc, y_proc, coef_new)
                _q_new_dev_last = q_new_dev
                bound_dev = q_yk_dev + _dot_dev(grad, diff) + 0.5 * L * _sum_sq_dev(diff)
                slack_dev = bound_dev + _SLACK_TOLERANCE - q_new_dev
                _armijo_ok = _to_float_scalar(slack_dev) >= 0
                if _armijo_ok:
                    break
                L *= 1.5
                step = 1.0 / L

            coef = coef_new

            # Finiteness check
            if not _is_quadratic:
                _coef_norm_dev = _norm2_dev(coef)
                _finite_ok = np.isfinite(float(_coef_norm_dev))
                if not _finite_ok:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                        y_k = _copy_arr(coef)
                        t_k = 1.0
                        L = L * 2.0
                        continue

            # Divergence detection
            if not _is_quadratic and iteration > 0:
                _need_norm_check = (iteration > 10)
                if _q_new_dev_last is not None:
                    _obj_dev = _q_new_dev_last
                    _q_new_dev_last = None
                else:
                    if sample_weight is not None:
                        _obj_dev, _ = _weighted_loss_and_grad(loss, X_proc, y_proc, coef, sample_weight)
                    else:
                        _obj_dev = loss.value(X_proc, y_proc, coef)
                _obj_val_f = float(_to_numpy(_obj_dev))
                _obj_val_f += _tracking_penalty_value(penalty, coef)
                _diverged_f = False
                if not np.isfinite(_obj_val_f):
                    _diverged_f = True
                elif _obj_best_fista > 1e-8:
                    _diverged_f = _obj_val_f > _obj_best_fista * 10.0 + 1e-8
                else:
                    _diverged_f = _obj_val_f > _obj_best_fista + max(abs(_obj_best_fista) * 10.0, 1.0)
                if not _diverged_f and _need_norm_check:
                    if float(_to_numpy(_norm2_dev(coef))) > _DIVERGE_COEF_NORM_CAP:
                        _diverged_f = True
                if _diverged_f:
                    if _coef_best_fista is not None:
                        coef = _copy_arr(_coef_best_fista)
                    else:
                        coef = _zeros(n_features, backend, ref_tensor=X_proc)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    L = L * 2.0
                    continue
                elif _obj_val_f < _obj_best_fista:
                    _obj_best_fista = _obj_val_f
                    _coef_best_fista = _copy_arr(coef)

            # Periodic Lipschitz recomputation
            # Skip if coefficients haven't changed much (Lipschitz is stable)
            if not _is_quadratic and iteration > 0 and iteration % 5 == 0:
                # Batch both norms into a single GPU→CPU transfer
                _coef_change, _coef_norm = _sync_scalars(
                    _norm2_dev(coef - coef_old), _norm2_dev(coef), backend=backend)
                _relative_change = _coef_change / max(_coef_norm, 1e-10)
                if _relative_change > 1e-3:  # Only recompute if coefficients changed significantly
                    try:
                        L_new = loss.lipschitz(X_proc, coef, y=y_proc, sample_weight=sample_weight)
                    except TypeError:
                        L_new = loss.lipschitz(X_proc, coef, y=y_proc)
                    # Safety factors from loss class
                    _lip_safety_recomp = getattr(loss, '_lipschitz_safety', 1.0)
                    if _lip_safety_recomp > 1.0:
                        L_new = L_new * _lip_safety_recomp
                    if _smooth_lip > 0:
                        L_new = L_new + _smooth_lip
                    if L_new > L:
                        L = L_new
                    else:
                        L = max(L * 0.8, L_new)

        # Momentum update — all backends
        if _conservative_momentum:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = min((t_k - 1.0) / t_new, 0.5)
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

        # Convergence check — deferred for GPU, every iteration for CPU
        if _is_gpu:
            if iteration < 20 or iteration % _conv_interval == 0:
                _conv_dev = _abs_sum_dev(coef - coef_old)
                if _to_float_scalar(_conv_dev) < tol:
                    break
        else:
            _conv_dev = _abs_sum_dev(coef - coef_old)
            if float(_conv_dev) < tol:
                break

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


def fista_lla_path(
    loss,
    scad_penalty,
    X,
    y,
    alpha_path,
    max_lla_per_step=6,
    lla_tol=1e-6,
    max_iter=1000,
    tol=1e-4,
    fit_intercept=True,
    sample_weight=None,
    lla_penalty_factory=None,
    init_coef=None,
    init_intercept=None,
    return_path=False,
):
    """Fused LLA+FISTA solver for SCAD/MCP over a continuation path.

    Runs the entire continuation → LLA → FISTA loop in one tight function,
    eliminating per-call overhead (backend detect, preprocess, Lipschitz
    recompute, array allocation) that accumulates over 300+ fista_solver calls.

    Parameters
    ----------
    loss : GLMLoss
    scad_penalty : SCADPenalty or MCPPenalty
        Penalty object; its .alpha will be set along the path.
    X, y : array (pre-centered if fit_intercept=True)
    alpha_path : array of alpha values (descending, geomspace)
    max_lla_per_step : int
    lla_tol : float
    max_iter : int or list[int]
        FISTA iteration limit. If a list, one value per continuation step.
    tol : float
    fit_intercept : bool
    sample_weight : array or None
    init_coef : array or None
        Warm-start coefficients (without intercept). If provided, they are
        injected only at the final target-alpha continuation step.
    init_intercept : float or None
        Warm-start intercept value.
    return_path : bool, default=False
        When True, also return coefficients/intercepts after each continuation
        alpha. The default keeps the historical 3-tuple return value.

    Returns
    -------
    coef : array (p,)
    intercept : float
    total_iter : int
    """
    from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty
    from statgpu.backends import _to_numpy

    backend = _resolve_backend("auto", X)
    if backend == "torch":
        import torch as xp
        torch = xp
        x_dtype = X.dtype if getattr(X, "is_floating_point", lambda: False)() else torch.float64
        y_dtype = y.dtype if getattr(y, "is_floating_point", lambda: False)() else torch.float64
        common_dtype = torch.promote_types(x_dtype, y_dtype)
        X = X.to(dtype=common_dtype)
        y = torch.as_tensor(y, device=X.device, dtype=common_dtype)
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np
    X_proc, y_proc = loss.preprocess(X, y)
    _loss_name = getattr(loss, 'name', '')
    _is_quadratic = (_loss_name == "squared_error")
    _no_momentum = _loss_name in ("poisson",)
    _non_smooth_pen_lla = getattr(scad_penalty, 'name', '') in (
        "l1", "elasticnet", "en", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    )
    _conservative_momentum_lla = (
        _loss_name in ("poisson", "negative_binomial", "tweedie", "inverse_gaussian")
        or (_loss_name in ("logistic", "gamma") and _non_smooth_pen_lla)
    )

    n_samples, n_features = X_proc.shape
    _validate_sample_weight(sample_weight, n_samples)

    # --- Intercept handling ---
    # For squared_error (identity link): centering X, y is exact.
    # For GLM losses (log/logit link): centering is WRONG — it changes
    # the objective.  Instead, augment X with a ones column so the
    # intercept is part of the coefficient vector.
    _augment_intercept = fit_intercept and not _is_quadratic
    if _augment_intercept:
        # Augment X with a column of ones
        from statgpu.backends._utils import xp_ones
        ones_col = xp_ones((X.shape[0], 1), dtype=X.dtype, xp=xp, ref_arr=X)
        X_c = xp.concatenate([X, ones_col], axis=1)
        y_c = y
        n_aug = n_features + 1
    elif fit_intercept:
        # squared_error: centering is exact for identity link
        X_mean = xp.mean(X, axis=0)
        y_mean = xp.mean(y)
        X_c = X - X_mean
        y_c = y - y_mean
        n_aug = n_features
    else:
        X_c = X
        y_c = y
        n_aug = n_features

    # Precompute Lipschitz using loss-specific method.
    # Pass zero coef (global bound) — not all losses handle coef=None.
    _zero_coef_lla = _zeros(n_aug, backend, ref_tensor=X_c)
    L_base = loss.lipschitz(X_c, _zero_coef_lla, y=y_c)
    # Precompute XtX only for squared_error fast path (skip for GLM losses)
    XtX = X_c.T @ X_c if _is_quadratic else None
    if L_base <= 0:
        L_base = 1.0

    # Apply loss-specific Lipschitz safety factor (e.g. NB=2x, gamma=3x)
    _lipschitz_safety = getattr(loss, '_lipschitz_safety', 1.0)
    if _lipschitz_safety > 1.0:
        L_base = L_base * _lipschitz_safety

    # Y-scaling for exp-link families (Poisson, Gamma, etc.).
    # At coef=0, mu≈1, but near the optimum mu≈y.  The Hessian scales
    # with mu, so L_base underestimates by up to max(y).
    # Cap at 10x — periodic Lipschitz recomputation corrects any remaining
    # underestimate during the FISTA inner loop.
    _skip_y_scaling = getattr(loss, '_lipschitz_uses_y', False)
    _y_lipschitz_scale = 1.0
    if not _is_quadratic and not _skip_y_scaling:
        _y_arr = _to_numpy(y_c)
        _y_abs = np.abs(_y_arr)
        _y_mean = float(np.mean(_y_abs))
        _y_max = float(np.max(_y_abs))
        _y_lipschitz_scale = min(10.0, max(1.0, np.sqrt(_y_mean * _y_max)))
        if _y_lipschitz_scale > 1.0:
            L_base = L_base * _y_lipschitz_scale

    def _zeros_coef():
        return _zeros(n_aug, backend, ref_tensor=X_c)

    def _warm_start_coef():
        if init_coef is None:
            return None
        if backend == "torch":
            import torch
            _init = torch.as_tensor(init_coef, device=X_c.device, dtype=X_c.dtype)
            if _augment_intercept and _init.shape[0] == n_features:
                return torch.cat([
                    _init,
                    torch.tensor(
                        [0.0 if init_intercept is None else init_intercept],
                        device=X_c.device,
                        dtype=X_c.dtype,
                    ),
                ])
            return _init.clone()
        if backend == "cupy":
            import cupy as cp
            _init = cp.asarray(init_coef, dtype=X_c.dtype)
            if _augment_intercept and _init.shape[0] == n_features:
                return cp.concatenate([
                    _init,
                    cp.array([0.0 if init_intercept is None else init_intercept], dtype=X_c.dtype),
                ])
            return _init.copy()
        _init = np.asarray(init_coef, dtype=np.float64)
        if _augment_intercept and _init.shape[0] == n_features:
            return np.concatenate([
                _init,
                [0.0 if init_intercept is None else float(init_intercept)],
            ])
        return _init.copy()

    # Keep the continuation path deterministic from zero. CV warm-starts are
    # injected only at the target-alpha step, otherwise SCAD/MCP LLA weights can
    # follow a different local trajectory for NB/Tweedie-like losses.
    coef = _zeros_coef()
    warm_coef = _warm_start_coef()

    total_iter = 0
    inner_pen = AdaptiveL1Penalty(alpha=1.0)
    path_records = [] if return_path else None

    def _split_current_coef(current_coef):
        coef_all = np.asarray(_to_numpy(current_coef), dtype=np.float64).ravel()
        if _augment_intercept:
            return coef_all[:n_features].copy(), float(coef_all[n_features])
        if fit_intercept:
            X_mean_np = np.asarray(_to_numpy(X_mean), dtype=np.float64).ravel()
            y_mean_np = float(_to_numpy(y_mean))
            return coef_all.copy(), float(y_mean_np - X_mean_np @ coef_all)
        return coef_all.copy(), 0.0

    def _record_path_alpha(alpha_value):
        if path_records is None:
            return
        coef_rec, intercept_rec = _split_current_coef(coef)
        path_records.append({
            "alpha": float(alpha_value),
            "coef": coef_rec,
            "intercept": float(intercept_rec),
            "n_iter": int(total_iter),
        })

    # For squared_error + GPU: fully inlined fused loop.
    # Uses torch.compile for torch, ElementwiseKernel for cupy.
    if _is_quadratic and backend in ("torch", "cupy"):
        Xty = X_c.T @ y_c

        # Get fused proximal kernel
        if backend == "torch":
            _fused = _get_sqerr_proximal_torch()
            coef_old = coef.clone()
            y_k = coef.clone()
        else:
            _fused = _get_sqerr_proximal_cupy()
            coef_old = coef.copy()
            y_k = coef.copy()

        step = 1.0 / L_base
        t_k = 1.0

        for _cont_i, cont_alpha in enumerate(alpha_path):
            # Create a copy with the continuation alpha to avoid mutating
            # the shared penalty object (thread-safety for future parallel CV).
            _pen_step = copy.copy(scad_penalty)
            _pen_step.alpha = float(cont_alpha)
            _mi = max_iter[_cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
            if warm_coef is not None and _cont_i == len(alpha_path) - 1:
                coef = _copy_arr(warm_coef)
            for _lla_i in range(max_lla_per_step):
                # lla_weights() is now backend-aware — stays on device
                lla_w = _pen_step.lla_weights(coef)
                thresh = lla_w * step  # stays on device

                # Save coef for LLA convergence check (on device)
                coef_before_lla = _copy_arr(coef)

                # Reset momentum for new LLA step
                t_k = 1.0
                coef_old = _copy_arr(coef)
                y_k = _copy_arr(coef)

                # FISTA inner solve (inlined, fused proximal+momentum)
                _conv_interval = 20  # check convergence every N iters (reduced GPU sync)
                iteration = -1  # guard against _mi=0 causing UnboundLocalError
                for iteration in range(_mi):
                    coef_old = _copy_arr(coef)

                    # Gradient: grad = (XtX @ y_k - Xty) / n
                    grad = (XtX @ y_k - Xty) / n_samples

                    # Clip gradients
                    if iteration % 10 == 0:
                        grad = _clip_grad_on_device(grad, coef_old, backend)

                    # Compute momentum beta BEFORE proximal so fused kernel does both
                    if _no_momentum:
                        beta_mom = 0.0
                    else:
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta_mom = (t_k - 1.0) / t_new
                        t_k = t_new

                    # Fused proximal + momentum in one kernel call. The gradient
                    # is evaluated at y_k, so y_k is the proximal center.
                    coef, y_k = _fused(y_k, grad, step, thresh, coef_old, beta_mom)

                    # Convergence check (device-side, minimal sync)
                    if iteration < 20 or iteration % _conv_interval == 0:
                        coef_diff_dev = _abs_sum_dev(coef - coef_old)
                        _cdf = _to_float_scalar(coef_diff_dev)
                        converged = _cdf < tol
                        diverged = (not np.isfinite(_cdf))
                        if converged:
                            break
                        if diverged:
                            coef = _copy_arr(coef_old)
                            break

                total_iter += iteration + 1

                # LLA convergence check (device-side, minimal sync)
                delta_dev = _abs_sum_dev(coef - coef_before_lla)
                if _to_float_scalar(delta_dev) < lla_tol:
                    break
            _record_path_alpha(cont_alpha)
    else:
        # Pre-compute XtX and Xty for squared_error (avoids redundant matmuls)
        _use_xtx = _is_quadratic and backend == "numpy"
        if _use_xtx:
            Xty = X_c.T @ y_c

        for _cont_i, cont_alpha in enumerate(alpha_path):
            # Create a copy with the continuation alpha to avoid mutating
            # the shared penalty object (thread-safety for future parallel CV).
            _pen_step = copy.copy(scad_penalty)
            _pen_step.alpha = float(cont_alpha)
            _mi = max_iter[_cont_i] if isinstance(max_iter, (list, tuple)) else max_iter
            if warm_coef is not None and _cont_i == len(alpha_path) - 1:
                coef = _copy_arr(warm_coef)

            for _lla_i in range(max_lla_per_step):
                # lla_weights() is now backend-aware — stays on device
                if _augment_intercept:
                    lla_w_feat = _pen_step.lla_weights(coef[:n_features])
                    # Append 0.0 for intercept on device
                    _zero_append = _zeros(1, backend, ref_tensor=coef)
                    lla_w = xp.concatenate([lla_w_feat, _zero_append])
                else:
                    lla_w = _pen_step.lla_weights(coef)
                if lla_penalty_factory is not None:
                    # lla_penalty_factory expects numpy; convert only if needed
                    lla_w_np = _to_numpy(lla_w) if type(lla_w).__module__ != "numpy" else lla_w
                    inner_pen = lla_penalty_factory(lla_w_np)
                else:
                    inner_pen._weights = lla_w

                # Save coef for LLA convergence check (on device)
                coef_before_lla = _copy_arr(coef)

                # --- FISTA inner solve (inlined, no function call overhead) ---
                _obj_best_lla_inner = None
                _coef_best_lla_inner = None
                y_k = _copy_arr(coef)
                t_k = 1.0
                # Use L_base which includes y-scaling for exp-link families.
                # loss.lipschitz(X, zeros) returns the local Hessian at mu=1,
                # but y-scaling approximates the Hessian at mu≈y — much tighter.
                # Backtracking will adapt L within the inner loop.
                L = L_base
                step = 1.0 / L
                _L_recompute_interval = 20

                _fista_step = _get_fista_step_compiled() if backend == "torch" else None

                # Pre-compute device-side tolerance for convergence check
                if backend != "numpy":
                    _tol_dev = xp.asarray(tol)

                for iteration in range(_mi):
                    coef_old = _copy_arr(coef)

                    if _use_xtx:
                        # Fast path: pre-computed XtX for squared_error
                        q_yk_dev = float(_sum_sq_dev(y_c - X_c @ y_k)) * 0.5 / n_samples
                        grad = (XtX @ y_k - Xty) / n_samples
                    else:
                        if sample_weight is not None:
                            q_yk_dev, grad = _weighted_loss_and_grad(loss, X_c, y_c, y_k, sample_weight)
                        else:
                            q_yk_dev, grad = _fused_glm_value_and_gradient(loss, X_c, y_c, y_k)

                    # Clip gradients (device-side, every 10 iterations)
                    if backend == "numpy" or iteration % 10 == 0:
                        _gn_dev = _norm2_dev(grad)
                        _gsum = _abs_sum_dev(coef_old) * 10.0 + 1e3
                        if backend == "torch":
                            _gmax_dev = xp.clamp(_gsum, min=1e4)
                        else:
                            _gmax_dev = xp.maximum(_gsum, 1e4)
                        # Batch both norms into a single GPU→CPU transfer
                        _gn_f, _gmax_f = _sync_scalars(_gn_dev, _gmax_dev, backend=backend)
                        _clip_needed = _gn_f > _gmax_f
                        if _clip_needed:
                            grad = grad * (_gmax_dev / _gn_dev)

                    # Backtracking (device-side Armijo check)
                    # Optimization: compute quadratic bound first; skip expensive
                    # loss.value() when bound is clearly below current best (NB/matmul cost).
                    for _bt in range(20):
                        if _fista_step is not None:
                            w_tilde, _ = _fista_step_call(_fista_step, y_k, grad, step, coef_old, coef, 0.0)
                        else:
                            w_tilde = y_k - step * grad
                        coef_new = inner_pen.proximal(w_tilde, step, backend=backend)

                        diff = coef_new - y_k
                        bound_dev = q_yk_dev + _dot_dev(grad, diff) + 0.5 * L * _sum_sq_dev(diff)

                        try:
                            q_new_dev = loss.value(X_c, y_c, coef_new, sample_weight=sample_weight)
                        except TypeError:
                            q_new_dev = loss.value(X_c, y_c, coef_new)
                        slack_dev = bound_dev + _SLACK_TOLERANCE - q_new_dev
                        _armijo_ok = _to_float_scalar(slack_dev) >= 0
                        if _armijo_ok:
                            break
                        L *= 1.5
                        step = 1.0 / L

                    # If all backtracking steps failed, fall back to best known
                    # iterate instead of accepting a potentially worse point.
                    if not _armijo_ok and _coef_best_lla_inner is not None:
                        coef = _copy_arr(_coef_best_lla_inner)
                    else:
                        coef = coef_new

                    # Batched safety checks: coef norm capping + finiteness + divergence
                    # All comparisons done on device, single D2H transfer for booleans
                    if not _is_quadratic:
                        # Compute coef norm once (shared by cap + finiteness + divergence)
                        if _augment_intercept:
                            _cn_dev = _norm2_dev(coef[:n_features])
                        else:
                            _cn_dev = _norm2_dev(coef)

                        # On-device checks
                        if backend == "torch":
                            _finite_dev = xp.isfinite(_cn_dev)
                            _cap_needed_dev = _cn_dev > 5.0
                            _diverge_norm_dev = _cn_dev > _DIVERGE_COEF_NORM_CAP if iteration > 10 else xp.tensor(False, device=coef.device)
                            _obj_finite_dev = xp.isfinite(q_new_dev) if iteration > 0 else xp.tensor(True, device=coef.device)
                        elif backend == "cupy":
                            _finite_dev = xp.isfinite(_cn_dev)
                            _cap_needed_dev = _cn_dev > 5.0
                            _diverge_norm_dev = _cn_dev > _DIVERGE_COEF_NORM_CAP if iteration > 10 else xp.asarray(False)
                            _obj_finite_dev = xp.isfinite(q_new_dev) if iteration > 0 else xp.asarray(True)
                        else:
                            # Batch GPU→CPU sync: transfer 2 scalars instead of 4
                            _cn_f, _obj_f = _sync_scalars(_cn_dev, q_new_dev, backend=backend)
                            _finite_dev = np.isfinite(_cn_f)
                            _cap_needed_dev = _cn_f > 5.0
                            _diverge_norm_dev = _cn_f > _DIVERGE_COEF_NORM_CAP if iteration > 10 else False
                            _obj_finite_dev = np.isfinite(_obj_f) if iteration > 0 else True

                        # Single D2H transfer: pack booleans
                        if backend == "numpy":
                            _finite = _finite_dev
                            _cap_needed = _cap_needed_dev
                            _obj_finite = _obj_finite_dev
                            _diverge_norm = _diverge_norm_dev
                        else:
                            _checks = _to_numpy(xp.stack([
                                _finite_dev,
                                _cap_needed_dev,
                                _obj_finite_dev,
                                _diverge_norm_dev,
                            ]))
                            _finite = bool(_checks[0])
                            _cap_needed = bool(_checks[1])
                            _obj_finite = bool(_checks[2])
                            _diverge_norm = bool(_checks[3])

                        # Finiteness reset
                        if not _finite:
                            if _coef_best_lla_inner is not None:
                                coef = _copy_arr(_coef_best_lla_inner)
                            else:
                                coef = _copy_arr(coef_old)
                            y_k = _copy_arr(coef)
                            t_k = 1.0
                            L = L * 2.0
                            step = 1.0 / L
                            continue

                        # Batch sync: coef norm + objective (1 GPU→CPU transfer)
                        if iteration > 0 or _cap_needed:
                            _cn_f, _obj_val_f = _sync_scalars(_cn_dev, q_new_dev, backend=backend)

                        # Coef norm capping
                        if _cap_needed:
                            _scale = 5.0 / _cn_f
                            if _augment_intercept:
                                coef = _copy_arr(coef)
                                coef[:n_features] = coef[:n_features] * _scale
                            else:
                                coef = coef * _scale
                            y_k = _copy_arr(coef)
                            t_k = 1.0

                        # Divergence detection
                        if iteration > 0:
                            _diverged_f = not _obj_finite
                            if not _diverged_f and _obj_best_lla_inner is not None:
                                if _obj_best_lla_inner > 1e-8:
                                    _diverged_f = _obj_val_f > _obj_best_lla_inner * 10.0 + 1e-8
                                else:
                                    _diverged_f = _obj_val_f > _obj_best_lla_inner + max(abs(_obj_best_lla_inner) * 10.0, 1.0)
                            if not _diverged_f and _diverge_norm:
                                _diverged_f = True
                            if _diverged_f:
                                if _coef_best_lla_inner is not None:
                                    coef = _copy_arr(_coef_best_lla_inner)
                                else:
                                    coef = _copy_arr(coef_old)
                                y_k = _copy_arr(coef)
                                t_k = 1.0
                                L = L * 2.0
                                step = 1.0 / L
                                continue
                            if _obj_best_lla_inner is None or _obj_val_f < _obj_best_lla_inner:
                                _obj_best_lla_inner = _obj_val_f
                                _coef_best_lla_inner = _copy_arr(coef)

                    # Momentum
                    if _no_momentum:
                        t_k = 1.0
                        y_k = _copy_arr(coef)
                    elif _conservative_momentum_lla:
                        # Conservative Nesterov for exp-link families: cap beta to avoid explosion
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta_raw = (t_k - 1.0) / t_new
                        beta = min(beta_raw, 0.5)  # uniform cap matching fista_solver
                        y_k = coef + beta * (coef - coef_old)
                        t_k = t_new
                    else:
                        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                        beta = (t_k - 1.0) / t_new
                        y_k = coef + beta * (coef - coef_old)
                        t_k = t_new

                    # Convergence (device-side comparison, only D2H 1 bool)
                    if backend == "numpy" or iteration < 20 or iteration % 5 == 0:
                        _conv_dev = _abs_sum_dev(coef - coef_old)
                        if backend != "numpy":
                            if bool(_to_numpy(_conv_dev < _tol_dev)):
                                break
                        else:
                            if float(_to_numpy(_conv_dev)) < tol:
                                break

                    # Periodic Lipschitz recomputation — corrects stale L
                    # as coef moves away from zero. Piggybacked on existing
                    # sync (convergence check already syncs every 5 iters).
                    if not _is_quadratic and iteration > 0 and iteration % _L_recompute_interval == 0:
                        L_new = loss.lipschitz(X_c, coef, y=y_c)
                        if _y_lipschitz_scale > 1.0:
                            L_new = L_new * _y_lipschitz_scale
                        if L_new > L * 1.5 or L_new < L / 1.5:
                            L = max(L_new, L_base * 0.1)
                            step = 1.0 / L

                    total_iter += 1
                # --- end FISTA ---

                # LLA convergence (on device, single sync for scalar)
                delta = float(_to_numpy(_abs_sum_dev(coef - coef_before_lla)))
                if delta < lla_tol:
                    break
            _record_path_alpha(cont_alpha)

    # Extract coef and intercept
    coef_np, intercept = _split_current_coef(coef)

    if return_path:
        if path_records:
            path = {
                "alpha": np.asarray([r["alpha"] for r in path_records], dtype=np.float64),
                "coef": np.vstack([r["coef"] for r in path_records]).astype(np.float64, copy=False),
                "intercept": np.asarray([r["intercept"] for r in path_records], dtype=np.float64),
                "n_iter": np.asarray([r["n_iter"] for r in path_records], dtype=np.int64),
            }
        else:
            path = {
                "alpha": np.empty(0, dtype=np.float64),
                "coef": np.empty((0, n_features), dtype=np.float64),
                "intercept": np.empty(0, dtype=np.float64),
                "n_iter": np.empty(0, dtype=np.int64),
            }
        return coef_np, intercept, total_iter, path
    return coef_np, intercept, total_iter


# ---------------------------------------------------------------------------
# Fused FISTA for squared_error + AdaptiveL1 (SCAD/MCP via LLA)
# ---------------------------------------------------------------------------
# Pre-computes XtX, Xty to avoid redundant matmul; fuses element-wise ops;
# defers GPU→CPU syncs for convergence.

_SQERR_PROXIMAL_TORCH = None
_SQERR_PROXIMAL_CUPY = None


def _get_sqerr_proximal_torch():
    global _SQERR_PROXIMAL_TORCH
    if _SQERR_PROXIMAL_TORCH is None:
        import torch
        # torch.compile requires CUDA capability >= 7.0 (Triton).
        # Fall back to JIT script for older GPUs (P100 = 6.0).
        _cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if _cap >= 7:
            try:
                @torch.compile(mode='reduce-overhead', backend='inductor')
                def _fused_update(y_current, grad, step, thresh, coef_old, beta):
                    w = y_current - step * grad
                    abs_w = w.abs()
                    sign_w = w.sign()
                    coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                    y_k = coef_new + beta * (coef_new - coef_old)
                    return coef_new, y_k
                _SQERR_PROXIMAL_TORCH = _fused_update
            except (RuntimeError, TypeError):
                pass
        if _SQERR_PROXIMAL_TORCH is None:
            def _fused_update_eager(y_current, grad, step, thresh, coef_old, beta):
                w = y_current - step * grad
                abs_w = w.abs()
                sign_w = w.sign()
                coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                y_k = coef_new + beta * (coef_new - coef_old)
                return coef_new, y_k
            _SQERR_PROXIMAL_TORCH = _fused_update_eager
    return _SQERR_PROXIMAL_TORCH


def _get_sqerr_proximal_cupy():
    global _SQERR_PROXIMAL_CUPY
    if _SQERR_PROXIMAL_CUPY is None:
        import cupy as cp
        _SQERR_PROXIMAL_CUPY = cp.ElementwiseKernel(
            'T y_current, T grad, T step, T thresh, T coef_old, T beta',
            'T coef_new, T y_k',
            '''
            T w = y_current - step * grad;
            T abs_w = abs(w);
            T sign_w = (w > 0) ? 1 : ((w < 0) ? -1 : 0);
            coef_new = (abs_w > thresh) ? sign_w * (abs_w - thresh) : 0;
            y_k = coef_new + beta * (coef_new - coef_old);
            ''',
            'sqerr_proximal_fused',
        )
    return _SQERR_PROXIMAL_CUPY


def newton_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
):
    """Newton-Raphson solver with Armijo backtracking line search.

    Supports numpy / cupy / torch backends via auto-detection of X.

    For losses with constant Hessian (Gamma, Tweedie with power≈2), the
    Hessian doesn't change across iterations, so the Newton direction is
    always valid and line search is skipped — this avoids the O(n*p*20)
    overhead of repeated objective evaluations in backtracking.

    Requires: loss has hessian() and penalty is smooth.
    """
    backend = _resolve_backend("auto", X)
    n_features = X.shape[1]

    if init_coef is not None:
        params = _copy_arr(init_coef) if hasattr(init_coef, 'copy') or hasattr(init_coef, 'clone') else np.array(init_coef).copy()
    else:
        params = _zeros(n_features, backend, ref_tensor=X)

    # Detect constant-Hessian losses (Gamma log link: H=X'X/n, Tweedie power≈2).
    # For these, the Newton step is always valid — skip line search.
    _loss_name = getattr(loss, 'name', '')
    _gamma_link = getattr(loss, 'link_name', getattr(loss, 'link', 'log'))
    _const_hessian = (_loss_name == "gamma" and _gamma_link == "log")
    if not _const_hessian and _loss_name == "tweedie":
        pw = getattr(loss, 'power', 1.5)
        if abs(pw - 2.0) < 0.01:
            _const_hessian = True

    # Precompute constant Hessian if applicable (saves O(p^2) per iter)
    _fixed_hess = None
    if _const_hessian:
        _fixed_hess = loss.hessian(X, y, params) + _smooth_penalty_hessian(penalty, params)

    _newton_step = _get_newton_step_compiled() if backend == "torch" else None
    _use_fused = _loss_name in ('logistic', 'poisson', 'gamma',
                                'negative_binomial', 'tweedie', 'inverse_gaussian')

    _validate_uniform_sample_weight(sample_weight, X.shape[0], "newton_solver")
    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        params_old = _copy_arr(params)
        grad = _objective_gradient(loss, penalty, X, y, params)
        hess = _fixed_hess if _fixed_hess is not None else (
            loss.hessian(X, y, params) + _smooth_penalty_hessian(penalty, params)
        )

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

        # Armijo backtracking line search — device-side.
        if _use_fused:
            obj_old_dev, _ = _fused_glm_value_and_gradient(loss, X, y, params_old)
            obj_old_dev = obj_old_dev + _smooth_penalty_value_dev(penalty, params_old)
        else:
            obj_old_dev = _objective_value_dev(loss, penalty, X, y, params_old)
        gdd_dev = _dot_dev(grad, direction)
        # Only sync gdd (needed for Armijo threshold); obj_old stays device-side
        gdd = _to_float_scalar(gdd_dev)

        step = 1.0
        for _bt in range(20):
            params_try = params_old - step * direction
            try:
                if _use_fused:
                    obj_try_dev, _ = _fused_glm_value_and_gradient(loss, X, y, params_try)
                    obj_try_dev = obj_try_dev + _smooth_penalty_value_dev(penalty, params_try)
                else:
                    obj_try_dev = _objective_value_dev(loss, penalty, X, y, params_try)
                if _device_leq(obj_try_dev, obj_old_dev + 1e-4 * step * gdd):
                    params = params_try
                    break
            except (ValueError, RuntimeError, FloatingPointError):
                pass
            step *= 0.5
        else:
            params = params_old - step * direction

        norm_diff_dev = _norm2_dev(params - params_old)
        nd, = _sync_scalars(norm_diff_dev, backend=backend)
        if nd < tol:
            break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"newton_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter


def fista_bb_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=1000,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    use_restart=True,
    step_max_factor=1e3,
    step_min_factor=1e-3,
    bb_burn_in=20,
    cv_mode=False,
    lipschitz_L=None,
):
    """FISTA with Barzilai-Borwein step sizes and adaptive restart.

    Uses alternating BB1/BB2 steps (Barzilai & Borwein 1988) that adapt to
    local curvature, eliminating the backtracking line search while preserving
    sparsity.  BB1 = <dw,dw>/<dw,dg> (long step), BB2 = <dw,dg>/<dg,dg>
    (short step).  Adaptive restart (O'Donoghue & Candes 2015) resets
    momentum when it opposes the descent direction.

    Supports numpy / cupy / torch backends via auto-detection of X.
    """
    backend = _resolve_backend("auto", X)
    _is_gpu = backend in ("torch", "cupy")
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    _loss_name = getattr(loss, 'name', '')
    _pen_name = _penalty_name(penalty)

    # Smooth logistic objectives are better handled by the Armijo-backed FISTA
    # path.  This keeps explicit fista_bb numerically aligned across CPU/CuPy/
    # Torch for logistic+none/l2 Section A checks.
    if _loss_name == "logistic" and _pen_name in ("l2", "none", "null", ""):
        return fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=max_iter,
            tol=tol,
            init_coef=init_coef,
            sample_weight=sample_weight,
            cv_mode=cv_mode,
        )

    # --- Initialize coefficients ---
    if init_coef is not None:
        coef = _as_backend_vector(init_coef, backend, X)
    else:
        coef = _zeros(n_features, backend, ref_tensor=X)

    y_k = _copy_arr(coef)
    t_k = 1.0

    # Divergence detection: track best objective for recovery
    _obj_best = float('inf')
    _coef_best = None
    _diverge_count = 0

    _bb_use_long = True     # alternate BB1 / BB2
    dot_dw_dg = 0.0         # BB step numerator (initialized for bb_burn_in=0)
    dot_dw_dw = 1.0         # BB step denominator
    _div_check_interval = 25 if cv_mode and _is_gpu else 5
    _lip_check_interval = 25 if cv_mode and _is_gpu else 5
    _conv_check_interval = 10 if cv_mode and _is_gpu else 3
    # For quadratic losses (squared_error) the gradient is linear in coef,
    # so dg = H @ dw and BB1 = BB2 = 1 / Rayleigh_quotient(H, dw).  The BB
    # step gives zero adaptation and the algorithm degenerates to ISTA
    # (O(1/k) convergence), too slow to reach the true sparse solution
    # within max_iter.  Use standard FISTA (fixed Lipschitz step + Nesterov
    # momentum, O(1/k^2)) instead.
    _is_quadratic = (_loss_name == "squared_error")

    # BB steps estimate local curvature from smooth-gradient differences.
    # For non-smooth penalties the proximal operator introduces a
    # discontinuity that makes the gradient differences noisy.
    #
    # On quadratic losses (squared_error) BB adds nothing — BB1 = BB2 =
    # 1/R_H and the method degenerates to ISTA (O(1/k)).  _is_quadratic
    # already disables BB above.
    #
    # For GLM losses with convex non-smooth penalties (L1, elasticnet,
    # adaptive_l1) the subgradient is bounded and BB differences are valid
    # after a burn-in that lets the iterates stabilise.  This gives 2-3×
    # faster convergence for logistic+L1, poisson+L1, etc.
    #
    # For non-convex non-smooth penalties (SCAD, MCP, group_*) the
    # subgradient can change abruptly (reweighting, folding points),
    # amplifying noise through the non-linear link and causing catastrophic
    # divergence.  Disable BB entirely for these.
    _pen_name = getattr(penalty, "name", _pen_name).lower() if hasattr(getattr(penalty, "name", _pen_name), 'lower') else _pen_name
    _bb_disabled = {
        "scad", "mcp",
        "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
    }
    if _pen_name in _bb_disabled:
        bb_burn_in = max_iter + 1  # never switch to BB
    elif _pen_name in {"l1", "elasticnet", "en", "adaptive_l1", "adaptive_lasso"}:
        bb_burn_in = max(bb_burn_in, 50)  # longer burn-in for non-smooth

    # Initial Lipschitz at zero (safe for all losses).  Computing L at
    # init_coef can produce enormous values for exp-link families (mu =
    # exp(X@coef) explodes for warm-start coefs from OLS).
    _zero_coef_bb = _zeros(n_features, backend, ref_tensor=X)
    _cached_lipschitz_L = None
    if lipschitz_L is not None:
        try:
            _cached_lipschitz_L = float(_to_numpy(lipschitz_L))
        except (ValueError, TypeError):
            _cached_lipschitz_L = None
    if _cached_lipschitz_L is not None and _cached_lipschitz_L > 0:
        L = _cached_lipschitz_L
    else:
        _cached_lipschitz_L = None
        try:
            L = loss.lipschitz(X_proc, _zero_coef_bb, y=y_proc, sample_weight=sample_weight)
        except TypeError:
            L = loss.lipschitz(X_proc, _zero_coef_bb, y=y_proc)
    if L <= 0:
        L = 1.0
    # For GLM losses with exp link (Poisson, etc.), mu at coef=0
    # is ~1, but mu near the optimum ≈ y.  The Hessian X'@diag(mu)@X
    # scales linearly with mu, so Lipschitz at init can underestimate the
    # true curvature by orders of magnitude (e.g. max(y)=2865 vs init mu=1).
    # Use geometric-mean heuristic: robust against extreme outliers while
    # still scaling up enough to avoid oversized first steps.
    # Logistic: BB step handles adaptation, y-scaling causes divergence.
    # Gamma's expected Fisher Hessian X'X/n underestimates
    # true curvature by ~mean(y), so y-scaling IS needed.
    _loss_global_lip_bb = _loss_name in ("logistic",)
    _skip_y_scaling_bb = getattr(loss, '_lipschitz_uses_y', False)
    _y_scale = 1.0  # default; overridden below for families that need it
    if not _is_quadratic and not _loss_global_lip_bb and not _skip_y_scaling_bb:
        _y_mean, _y_max = _abs_mean_max(y_proc, backend)
        _y_scale = max(1.0, _y_mean, np.sqrt(_y_mean * _y_max))
        if _y_scale > 1.0:
            L = L * _y_scale
    # Inverse Gaussian: gradient scales as 1/mu^3, causing extreme
    # sensitivity to step size.  Use a much more conservative Lipschitz
    # to prevent catastrophic divergence.
    _invgauss_like = _loss_name in ("inverse_gaussian",)
    _tweedie_like = _loss_name == "tweedie"
    _lip_safety_bb = getattr(loss, '_lipschitz_safety', 1.0)
    if _lip_safety_bb > 1.0:
        L = L * _lip_safety_bb
    # Add smooth penalty Lipschitz contribution (e.g. l2 gradient alpha*coef
    # has Lipschitz alpha).  Without this the step 1/L is too large.
    _smooth_lip_bb = _smooth_penalty_lipschitz(penalty)
    if _smooth_lip_bb > 0:
        L = L + _smooth_lip_bb
    step_L = 1.0 / L
    step_k = step_L
    step_max = step_L * step_max_factor
    step_min = step_L * step_min_factor
    _validate_sample_weight(sample_weight, X_proc.shape[0])

    # Gradient at initial point for first BB difference
    try:
        grad_old = loss.gradient(X_proc, y_proc, coef, sample_weight=sample_weight)
    except TypeError:
        grad_old = loss.gradient(X_proc, y_proc, coef)
    # Initialize dg for BB step selection (used before first assignment in loop)
    dg = _zeros(n_features, backend, ref_tensor=X_proc)
    iteration = -1  # default if max_iter=0

    # Loop-invariant constants for momentum/BB decisions
    _poisson_like = _loss_name in ("poisson",)
    _gamma_like = _loss_name in ("gamma",)

    # --- Pre-compute loop-invariant burn-in and momentum parameters ---
    # These depend only on loss/penalty type, not on iterates.
    if _invgauss_like:
        bb_burn_in = max_iter + 1   # never switch to BB
    elif _tweedie_like:
        bb_burn_in = max(200, max_iter // 2)
    elif _gamma_like:
        bb_burn_in = max(50, max_iter // 8)

    if _poisson_like or _invgauss_like:
        _momentum_burn_in = max_iter + 1   # never use momentum
    elif _tweedie_like:
        _momentum_burn_in = max(100, max_iter // 4)
    elif _gamma_like:
        _momentum_burn_in = max(30, max_iter // 10)
    else:
        _momentum_burn_in = 0  # momentum from the start

    # Conservative momentum for specific loss+penalty combos
    _conservative_bb = False
    if _poisson_like and not _invgauss_like:
        _pen_name_bb = getattr(penalty, 'name', '')
        if _pen_name_bb in ("l2", "none", "", None):
            _momentum_burn_in = min(100, max_iter)
            _conservative_bb = True
    if _tweedie_like or _gamma_like:
        _conservative_bb = True

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Gradient at extrapolated point
        try:
            grad = loss.gradient(X_proc, y_proc, y_k, sample_weight=sample_weight)
        except TypeError:
            grad = loss.gradient(X_proc, y_proc, y_k)

        # Clip extreme gradients — every iteration, all backends.
        # Skip for inverse_gaussian: 1/mu^3 gradient scaling produces large but
        # valid gradients; clipping prevents convergence to the true optimum.
        # Use identical sync-based clipping for both CPU and GPU to ensure
        # consistent trajectories (backtracking already syncs for non-quadratic).
        if not _invgauss_like:
            if cv_mode and _is_gpu:
                grad = _clip_grad_on_device(grad, coef_old, backend)
            else:
                _gn_f, _coef_abs_f = _sync_scalars(
                    _norm2_dev(grad), _abs_sum_dev(coef_old), backend=backend)
                _gmax = max(_coef_abs_f * 10.0 + 1e3, 1e4)
                if _gn_f > _gmax:
                    grad = grad * (_gmax / _gn_f)

        # --- Divergence detection ---
        # Full objective check every 5 iterations (GPU optimization: reduces
        # expensive loss.value() calls). Coefficient norm check every iteration
        # (cheap) catches catastrophic explosion early.
        # Batch obj + coef-norm into a single sync when both are needed.
        _do_full_div_check = (
            iteration % _div_check_interval == 0 or iteration <= 5
        )
        # GPU: defer ALL divergence checks to every 5 iterations (no per-iter sync)
        _do_div_check = (not _is_quadratic and iteration > 0 and
                         (not _is_gpu or _do_full_div_check))
        if _do_div_check:
            _diverged = False
            # Coef norm divergence check (works for both CPU and GPU)
            if iteration > 10 and not _diverged:
                _coef_norm_dev = _norm2_dev(coef)
                if _to_float_scalar(_coef_norm_dev) > _DIVERGE_COEF_NORM_CAP:
                    _diverged = True
            # Full objective check every 5 iterations
            if not _diverged:
                try:
                    _obj_val = float(_to_numpy(loss.value(X_proc, y_proc, coef, sample_weight=sample_weight)))
                except TypeError:
                    _obj_val = float(_to_numpy(loss.value(X_proc, y_proc, coef)))
                _pen_val = _tracking_penalty_value(penalty, coef)
                _obj_total = _obj_val + _pen_val
                if not np.isfinite(_obj_total):
                    _diverged = True
                elif not np.isfinite(_obj_best):
                    # _obj_best is inf/-inf (first valid iter or degenerate loss):
                    # skip ratio-based check, rely on norm check above.
                    pass
                elif _obj_best > 1e-8:
                    _diverge_threshold = _obj_best * 10.0 + 1e-8
                    if _invgauss_like or _tweedie_like:
                        _diverge_threshold = _obj_best * _DIVERGE_OBJ_RATIO + _DIVERGE_OBJ_ABS
                    _diverged = _obj_total > _diverge_threshold
                else:
                    _diverge_threshold = _obj_best + max(abs(_obj_best) * 10.0, 1.0)
                    if _invgauss_like or _tweedie_like:
                        _diverge_threshold = _obj_best + max(abs(_obj_best) * _DIVERGE_OBJ_RATIO, _DIVERGE_OBJ_ABS)
                    _diverged = _obj_total > _diverge_threshold
            if _diverged:
                # Diverged: reset to best known iterate (or zeros) and halve step
                _diverge_count += 1
                if _coef_best is not None:
                    coef = _copy_arr(_coef_best)
                else:
                    # No valid iterate yet — reset to zeros
                    coef = _zeros(n_features, backend, ref_tensor=X_proc)
                y_k = _copy_arr(coef)
                t_k = 1.0
                try:
                    grad_old = loss.gradient(X_proc, y_proc, coef, sample_weight=sample_weight)
                except TypeError:
                    grad_old = loss.gradient(X_proc, y_proc, coef)
                # Halve step size bounds
                step_L = step_L * 0.5
                step_k = step_L
                step_max = step_max * 0.5
                step_min = step_min * 0.5
                L = L * 2.0
                # Reset BB state
                dot_dw_dg = 0.0
                dot_dw_dw = 1.0
                continue
            elif _obj_total < _obj_best:
                _obj_best = _obj_total
                _coef_best = _copy_arr(coef)

        # --- Step size selection ---
        if _is_quadratic or iteration < bb_burn_in:
            # Quadratic loss or burn-in phase: use fixed Lipschitz step.
            # During burn-in for GLM losses, BB steps are delayed because
            # early gradient differences (dw, dg) are dominated by the
            # coef trajectory from zero toward the optimum rather than by
            # local curvature; using BB too early amplifies oscillations.
            step_k = step_L
            # Recompute Lipschitz periodically during burn-in since mu
            # (and therefore the Hessian scale) changes rapidly.
            if (
                not _is_quadratic
                and iteration > 0
                and iteration % _lip_check_interval == 0
            ):
                # Use global Lipschitz (coef=zero) during burn-in to prevent
                # iterate-dependent Lipschitz from shrinking too fast.
                # BB steps handle adaptation after burn-in.
                # Pass zero coef — not all losses handle coef=None.
                if _cached_lipschitz_L is not None:
                    L_new = _cached_lipschitz_L
                else:
                    L_new = loss.lipschitz(X_proc, _zero_coef_bb, y=y_proc)
                if L_new > 0:
                    # Re-apply y-scaling and per-family safety factor
                    if _y_scale > 1.0:
                        L_new = L_new * _y_scale
                    _lip_safety_bt = getattr(loss, '_lipschitz_safety', 1.0)
                    if _lip_safety_bt > 1.0:
                        L_new = L_new * _lip_safety_bt
                    # Allow L to move toward L_new: full increase, gradual decrease
                    if L_new > L:
                        L = L_new
                    else:
                        L = max(L * 0.8, L_new)
                    step_L = 1.0 / L
                    step_k = step_L
                    step_max = step_L * step_max_factor
                    step_min = step_L * step_min_factor
        else:
            # Nonlinear GLM loss, post-burn-in: use BB step when valid,
            # fall back to Lipschitz step otherwise.
            if dot_dw_dg > _BB_RESTART_DOT_TOL:
                if _bb_use_long:
                    step_k = dot_dw_dw / dot_dw_dg       # BB1: long
                else:
                    dot_dg_dg = float(_to_numpy(_dot_dev(dg, dg)))
                    step_k = dot_dw_dg / max(dot_dg_dg, 1e-14)  # BB2: short
                _bb_use_long = not _bb_use_long
                # Tweedie: cap BB step more aggressively to prevent overshoot
                if _tweedie_like:
                    step_k = min(step_k, step_L * 2.0)
                step_k = min(max(step_k, step_min), step_max)
            # else: keep previous step_k (step_L or last valid BB step)

        # Gradient step + proximal
        w_tilde = y_k - step_k * grad
        coef_new = penalty.proximal(w_tilde, step_k, backend=backend)
        coef = coef_new

        # Safeguarded backtracking for GLM losses:
        # After proximal, verify the objective didn't explode.  If it did,
        # halve step and recompute.  This catches cases where the BB step
        # or Lipschitz estimate was too optimistic for the new coef region.
        # Interval-based: full objective check every 5 iterations (expensive
        # loss.value() call), cheap norm check every iteration.
        _last_coef_norm_f = None
        if not _is_quadratic:
            _steep_loss = _loss_name in ("tweedie", "negative_binomial")
            # Interval-based: only run expensive objective check every 5 iters
            # (divergence detection above also checks every 5 iters)
            _do_bt_check = (iteration % 5 == 0 or iteration <= 5)
            if _do_bt_check:
                for _bt in range(15):
                    # Batch obj + coef-norm into a single sync.
                    _new_obj, _new_norm = _sync_scalars(
                        loss.value(X_proc, y_proc, coef), _norm2_dev(coef), backend=backend)
                    _new_pen = _tracking_penalty_value(penalty, coef)
                    _new_total = _new_obj + _new_pen
                    # Accept if: finite, reasonable norm, and objective not exploded.
                    # Use relative threshold (10x initial objective) instead of
                    # absolute 1e6 — NB/Tweedie with large counts can have
                    # legitimate loss > 1e6.
                    _obj_cap = max(_obj_best * 10.0, 1e6) if np.isfinite(_obj_best) else 1e6
                    if _steep_loss:
                        _obj_acceptable = (np.isfinite(_new_total) and _new_norm < _DIVERGE_COEF_NORM_CAP and
                                           _new_total < _obj_cap)
                    else:
                        # For logistic/gamma/poisson: accept if finite, reasonable
                        # norm, and objective not significantly worse than best known.
                        _obj_acceptable = (np.isfinite(_new_total) and _new_norm < _DIVERGE_COEF_NORM_CAP and
                                           _new_total < max(_obj_best * 1.5 + 1.0, 1e3))
                    if _obj_acceptable:
                        _last_coef_norm_f = _new_norm
                        break
                    # Step too large — halve and retry
                    step_k = step_k * 0.5
                    L = L * 2.0
                    w_tilde = y_k - step_k * grad
                    coef = penalty.proximal(w_tilde, step_k, backend=backend)
                    _last_coef_norm_f = None

        # Finiteness check: if coef is non-finite after proximal, reset.
        # Reuse the norm already synchronized by safeguarded backtracking.
        if not _is_quadratic:
            if _last_coef_norm_f is not None:
                _finite_ok2 = np.isfinite(_last_coef_norm_f)
            else:
                _coef_norm_dev2 = _norm2_dev(coef)
                _finite_ok2 = np.isfinite(_to_float_scalar(_coef_norm_dev2))
            if not _finite_ok2:
                _diverge_count += 1
                if _coef_best is not None:
                    coef = _copy_arr(_coef_best)
                    y_k = _copy_arr(coef)
                    t_k = 1.0
                    try:
                        grad_old = loss.gradient(X_proc, y_proc, coef, sample_weight=sample_weight)
                    except TypeError:
                        grad_old = loss.gradient(X_proc, y_proc, coef)
                    step_L = step_L * 0.5
                    step_k = step_L
                    step_max = step_max * 0.5
                    step_min = step_min * 0.5
                    L = L * 2.0
                    dot_dw_dg = 0.0
                    dot_dw_dw = 1.0
                    continue

        # --- Store BB step info for next iteration (non-quadratic only) ---
        # Use accepted iterate (coef) not pre-backtracking (coef_new)
        if not _is_quadratic:
            try:
                grad_new = loss.gradient(X_proc, y_proc, coef, sample_weight=sample_weight)
            except TypeError:
                grad_new = loss.gradient(X_proc, y_proc, coef)

            dw = coef - coef_old
            dg = grad_new - grad_old
            # Batch two dot products into a single GPU→CPU sync.
            dot_dw_dw, dot_dw_dg = _sync_scalars(
                _dot_dev(dw, dw), _dot_dev(dw, dg), backend=backend)
            grad_old = grad_new

        # --- Nesterov momentum with adaptive restart ---
        # bb_burn_in, _momentum_burn_in, _conservative_bb are loop-invariant
        # and computed once before the loop.
        if iteration < _momentum_burn_in:
            t_k = 1.0
            beta = 0.0
            y_k = _copy_arr(coef)   # next gradient at current point, not extrapolated
        elif _conservative_bb:
            # Conservative momentum: fixed small beta to avoid explosion
            beta = 0.2
            y_k = coef + beta * (coef - coef_old)
            t_k = 1.0
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta = (t_k - 1.0) / t_new

            if use_restart and iteration > 0:
                # GPU-side comparison, only sync bool.
                # Use `coef` (always current) not `coef_new` (stale after reset).
                _mc_dev = _dot_dev(y_k - coef, coef - coef_old)
                if _to_float_scalar(_mc_dev) > 0:
                    t_k = 1.0
                    t_new = 1.0
                    beta = 0.0

            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

        # --- Convergence check — deferred for GPU, every iteration for CPU. ---
        if _is_gpu:
            if iteration < 20 or iteration % _conv_check_interval == 0:
                _conv_dev2 = _abs_sum_dev(coef - coef_old)
                if _to_float_scalar(_conv_dev2) < tol:
                    break
        else:
            _conv_dev2 = _abs_sum_dev(coef - coef_old)
            if _to_float_scalar(_conv_dev2) < tol:
                break

    # Return best iterate if divergence was detected
    if _diverge_count > 0 and _coef_best is not None:
        coef = _coef_best

    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"fista_bb_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}). "
            f"Consider increasing max_iter or using a different solver (newton, lbfgs, irls).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return coef, n_iter


def lbfgs_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    history_size=10,
    sample_weight=None,
):
    """Limited-memory BFGS for smooth GLM objectives.

    The implementation keeps parameters, gradients, and curvature history on
    the input backend.  GPU-optimised path uses:
    - _fused_glm_value_and_gradient to avoid redundant X@coef
    - _dot_dev / _norm2_dev to stay on device
    - _sync_scalars to batch GPU→CPU transfers
    - _objective_value_dev + _device_leq for device-side line search
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

    _loss_name = getattr(loss, 'name', '')
    _use_fused = _loss_name in ('logistic', 'poisson', 'gamma',
                                'negative_binomial', 'tweedie', 'inverse_gaussian')

    # Initial gradient (fused to avoid redundant X@coef)
    if _use_fused:
        _init_val_dev, grad = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params)
        grad = grad + _smooth_penalty_gradient(penalty, params)
    else:
        grad = _objective_gradient(loss, penalty, X_proc, y_proc, params)

    if backend == "torch":
        import torch
        tol_dev = torch.tensor(tol, dtype=torch.float64, device=params.device)
    else:
        tol_dev = tol
    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        grad_norm_dev = _norm2_dev(grad)

        # Two-loop recursion — all dot products stay on device
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

        # Line search — stays on device
        if _use_fused:
            old_val_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params)
            old_val_dev = old_val_dev + _smooth_penalty_value_dev(penalty, params)
        else:
            old_val_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, params)

        step = 1.0
        params_new = params
        _ls_accepted = False
        for _ in range(25):
            candidate = params + step * direction
            if _use_fused:
                cand_val_dev, _ = _fused_glm_value_and_gradient(loss, X_proc, y_proc, candidate)
                cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(penalty, candidate)
            else:
                cand_val_dev = _objective_value_dev(loss, penalty, X_proc, y_proc, candidate)
            # Device-side comparison — single sync for the bool
            if _device_leq(cand_val_dev, old_val_dev + 1e-4 * step * gdd):
                params_new = candidate
                _ls_accepted = True
                break
            step *= 0.5
        if not _ls_accepted:
            import warnings
            warnings.warn(
                "lbfgs_solver: line search failed to find a descent step "
                f"after 25 backtracking steps (iteration {iteration}). "
                "Solver may stagnate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Update gradient (fused)
        if _use_fused:
            _, grad_new = _fused_glm_value_and_gradient(loss, X_proc, y_proc, params_new)
            grad_new = grad_new + _smooth_penalty_gradient(penalty, params_new)
        else:
            grad_new = _objective_gradient(loss, penalty, X_proc, y_proc, params_new)

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


# =============================================================================
# ADMM Solver (three-backend: numpy/cupy/torch)
# =============================================================================



def admm_solver(
    loss,
    penalty,
    X,
    y,
    max_iter=200,
    tol=1e-4,
    rho=1.0,
    adaptive_rho=True,
    cg_max_iter=30,
    cg_tol=1e-6,
    init_coef=None,
    sample_weight=None,
):
    """ADMM solver for penalized GLM optimization.

    Reformulates min_w f(Xw; y) + p(w) as:
        min_{w,z} f(Xw; y) + p(z)  s.t. w = z

    and solves via the alternating direction method of multipliers:
        w^{k+1} = argmin_w f(Xw; y) + (rho/2)||w - z^k + u^k||^2
        z^{k+1} = prox_{p/rho}(w^{k+1} + u^k)
        u^{k+1} = u^k + w^{k+1} - z^{k+1}

    The w-update is a smooth, strongly convex problem solved via conjugate
    gradient. The z-update reuses penalty.proximal(). Both are GPU-friendly:
    w-update uses dense matmuls (cuBLAS), z-update is element-wise.

    Supports numpy / cupy / torch backends via auto-detection of X.

    Parameters
    ----------
    loss : GLMLoss
    penalty : Penalty
    X, y : arrays
    max_iter : int
        Maximum ADMM outer iterations.
    tol : float
        Convergence tolerance for primal/dual residuals.
    rho : float
        Augmented Lagrangian penalty parameter.
    adaptive_rho : bool
        Adapt rho based on primal/dual residual balance.
    cg_max_iter : int
        Maximum CG iterations for w-update subproblem.
    cg_tol : float
        CG convergence tolerance.
    init_coef : array, optional
        Initial coefficients.
    sample_weight : array, optional

    Returns
    -------
    coef : array, n_iter : int
    """
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    # Initialize
    if init_coef is not None:
        w = (
            _copy_arr(init_coef)
            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")
            else np.array(init_coef).copy()
        )
    else:
        w = _zeros(n_features, backend, ref_tensor=X)

    z = _copy_arr(w)
    u = _zeros_like(w)

    if sample_weight is not None:
        _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "admm_solver")

    def _grad_w(w_vec, z_cur, u_cur):
        """Gradient of f(w) + (rho/2)||w - z_cur + u_cur||^2 w.r.t. w."""
        g = loss.gradient(X_proc, y_proc, w_vec)
        g = g + rho * (w_vec - z_cur + u_cur)
        return g

    # Detect if loss has a constant Hessian (squared_error) — use Cholesky.
    # For GLM losses, use Nesterov-accelerated gradient descent.
    # When using Cholesky we pin rho (disable adaptive_rho) because the
    # precomputed _A_mat = XtX/n + rho*I would become stale if rho changed.
    loss_name = getattr(loss, 'name', '')
    use_cholesky = loss_name == "squared_error" and n_features <= 2000
    if use_cholesky:
        adaptive_rho = False

    if use_cholesky:
        _hess_const = loss.hessian(X_proc, y_proc, w)          # XtX / n
        _A_mat = _hess_const
        _cholesky_ok = False
        if hasattr(_hess_const, 'shape'):
            try:
                if backend == "numpy":
                    _A_mat = _hess_const + rho * np.eye(n_features, dtype=_hess_const.dtype)
                    _L = np.linalg.cholesky(_A_mat)
                elif backend == "cupy":
                    import cupy as cp
                    _A_mat = _hess_const + rho * cp.eye(n_features, dtype=_hess_const.dtype)
                    _L = cp.linalg.cholesky(_A_mat)
                else:
                    import torch
                    _A_mat = _hess_const + rho * torch.eye(n_features, dtype=_hess_const.dtype, device=_hess_const.device)
                    _L = torch.linalg.cholesky(_A_mat)
                _cholesky_ok = True
            except (np.linalg.LinAlgError, Exception):
                # Matrix not positive-definite (numerical issues, collinear features)
                # Fall back to CG solver below
                _cholesky_ok = False
        if not _cholesky_ok:
            use_cholesky = False

        # Precompute -grad_f(0) = Xty/n for squared_error (the constant part)
        _zero_coef = _zeros_like(w)
        _neg_grad_zero = -loss.gradient(X_proc, y_proc, _zero_coef)  # Xty/n

    else:
        # Gradient descent step: 1/(L_f + rho)
        L_f = loss.lipschitz(X_proc, w, y=y_proc)
        if L_f <= 0:
            L_f = 1.0
        lr_sub = 1.0 / (L_f + rho + 1e-8)
    iteration = -1  # default if max_iter=0

    for iteration in range(max_iter):
        z_old = _copy_arr(z)

        # --- w-update ---
        if use_cholesky:
            # Closed-form: (XtX/n + rho*I) w = Xty/n + rho*(z - u)
            # Use precomputed Cholesky factor for forward/back substitution
            rhs = _neg_grad_zero + rho * (z - u)
            if backend == "numpy":
                from scipy.linalg import solve_triangular
                tmp = solve_triangular(_L, rhs, lower=True)
                w = solve_triangular(_L.T, tmp, lower=False)
            elif backend == "cupy":
                # Use triangular solve when available (O(n³/6) vs O(n³/3) for LU)
                try:
                    from cupyx.scipy.linalg import solve_triangular
                    tmp = solve_triangular(_L, rhs, lower=True)
                    w = solve_triangular(_L.T, tmp, lower=False)
                except ImportError:
                    tmp = cp.linalg.solve(_L, rhs)
                    w = cp.linalg.solve(_L.T, tmp)
            else:
                tmp = torch.linalg.solve_triangular(_L, rhs.unsqueeze(1), upper=False)
                w = torch.linalg.solve_triangular(_L.T, tmp, upper=True).squeeze(1)
        else:
            # Nesterov-accelerated gradient descent on the w-subproblem
            w_new = _copy_arr(w)
            w_mom = _copy_arr(w)
            t_mom = 1.0
            for _ in range(cg_max_iter):
                w_old_mom = _copy_arr(w_new)
                g_sub = _grad_w(w_mom, z, u)
                w_next = w_mom - lr_sub * g_sub
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
                w_mom = w_next + ((t_mom - 1.0) / t_new) * (w_next - w_new)
                w_new = w_next
                t_mom = t_new
                diff_dev = _abs_sum_dev(w_next - w_old_mom)
                if backend != "numpy":
                    if _device_leq(diff_dev, cg_tol * n_features):
                        break
                elif diff_dev < cg_tol * n_features:
                    break
            w = w_new

        # --- z-update: proximal operator ---
        # Contract: proximal(z, step) = argmin_x step*P(x) + (1/2)||x - z||²
        # ADMM z-update needs argmin_z P(z)/rho + (1/2)||z - (w+u)||²
        #   = proximal(w + u, 1/rho)  with step = 1/rho
        z = penalty.proximal(w + u, 1.0 / rho, backend=backend)

        # --- u-update: dual ascent ---
        u = u + w - z

        # --- Adaptive rho + Convergence check (batched sync) ---
        rp_dev = _norm2_dev(w - z)
        rd_dev = _norm2_dev(z - z_old)
        rp, rd_raw = _sync_scalars(rp_dev, rd_dev, backend=backend)
        r_dual = rho * rd_raw

        if adaptive_rho:
            if rp > 10.0 * r_dual:
                rho = min(rho * 2.0, 1e4)
            elif r_dual > 10.0 * rp:
                rho = max(rho * 0.5, 1e-4)
            # Recompute step size to match updated rho
            lr_sub = 1.0 / (L_f + rho + 1e-8)

        if rp < tol and r_dual < tol:
            break

    # Return z (penalized/feasible variable), not w (unconstrained).
    # At convergence w ≈ z, but z always satisfies the penalty structure.
    n_iter = iteration + 1
    if n_iter >= max_iter:
        warnings.warn(
            f"admm_solver did not converge within {max_iter} iterations "
            f"(loss={getattr(loss, 'name', '?')}, penalty={getattr(penalty, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return z, n_iter
