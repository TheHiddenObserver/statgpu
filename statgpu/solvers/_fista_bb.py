"""FISTA with Barzilai-Borwein step sizes and adaptive restart.

Uses alternating BB1/BB2 steps (Barzilai & Borwein 1988) that adapt to
local curvature, eliminating the backtracking line search while preserving
sparsity.  BB1 = <dw,dw>/<dw,dg> (long step), BB2 = <dw,dg>/<dg,dg>
(short step).  Adaptive restart (O'Donoghue & Candes 2015) resets
momentum when it opposes the descent direction.

Supports numpy / cupy / torch backends via auto-detection of X.
"""

from __future__ import annotations

__all__ = ["fista_bb_solver"]

import warnings
import numpy as np
from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _get_xp, _to_float_scalar
from statgpu.backends._array_ops import (
    _abs_sum_dev, _clip_grad_on_device, _copy_arr, _dot_dev,
    _norm2_dev, _sync_scalars, _zeros,
)
from statgpu.penalties._categories import BB_DISABLED as _BB_DISABLED
from ._convergence import ConvergenceWarning
from ._constants import (
    _DIVERGE_COEF_NORM_CAP,
    _BB_RESTART_DOT_TOL,
    _DIVERGE_OBJ_RATIO,
    _DIVERGE_OBJ_ABS,
    _GRAD_CLIP_COEF_FACTOR,
    _GRAD_CLIP_ABS_FLOOR,
    _GRAD_CLIP_MAX,
)
from ._fista import fista_solver
from ._utils import (
    _validate_sample_weight,
    _as_backend_vector,
    _call_with_weight,
    _nesterov_update,
    _penalty_name,
    _smooth_penalty_lipschitz,
    _tracking_penalty_value,
    _abs_mean_max,
)


def fista_bb_solver(
    loss,
    penalty: "Penalty | None",
    X,
    y,
    max_iter: int = 1000,
    tol: float = 1e-4,
    init_coef=None,
    sample_weight=None,
    use_restart: bool = True,
    step_max_factor: float = 1e3,
    step_min_factor: float = 1e-3,
    bb_burn_in: int = 20,
    cv_mode: bool = False,
    lipschitz_L: float | None = None,
) -> tuple:
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
    _pen_name = _penalty_name(penalty)

    # Convert sample_weight to backend-native (prevent CPU/CUDA mismatch)
    _sw_arr = None
    if sample_weight is not None:
        _xp_mod = _get_xp(backend)
        _sw_arr = _xp_mod.asarray(sample_weight, dtype=X_proc.dtype)
        if hasattr(X_proc, "device") and hasattr(_sw_arr, "to"):
            _sw_arr = _sw_arr.to(device=X_proc.device)

    # Smooth logistic objectives are better handled by the Armijo-backed FISTA
    # path.  This keeps explicit fista_bb numerically aligned across CPU/CuPy/
    # Torch for logistic+none/l2 Section A checks.
    if getattr(loss, '_prefer_fista_over_bb', False) and _pen_name in ("l2", "none", "null", ""):
        return fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=max_iter,
            tol=tol,
            init_coef=init_coef,
            sample_weight=_sw_arr,
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
    _is_quadratic = getattr(loss, '_is_quadratic', False)

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
    # after a burn-in that lets the iterates stabilise.  This gives 2-3x
    # faster convergence for logistic+L1, poisson+L1, etc.
    #
    # For non-convex non-smooth penalties (SCAD, MCP, group_*) the
    # subgradient can change abruptly (reweighting, folding points),
    # amplifying noise through the non-linear link and causing catastrophic
    # divergence.  Disable BB entirely for these.
    _pen_name = getattr(penalty, "name", _pen_name).lower() if hasattr(getattr(penalty, "name", _pen_name), 'lower') else _pen_name
    if _pen_name in _BB_DISABLED:
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
        L = _call_with_weight(loss.lipschitz, X_proc, _zero_coef_bb, y=y_proc, sample_weight=_sw_arr)
    if L <= 0:
        L = 1.0
    # For GLM losses with exp link (Poisson, etc.), mu at coef=0
    # is ~1, but mu near the optimum ~ y.  The Hessian X'@diag(mu)@X
    # scales linearly with mu, so Lipschitz at init can underestimate the
    # true curvature by orders of magnitude (e.g. max(y)=2865 vs init mu=1).
    # Use geometric-mean heuristic: robust against extreme outliers while
    # still scaling up enough to avoid oversized first steps.
    # Logistic: BB step handles adaptation, y-scaling causes divergence.
    # Gamma's expected Fisher Hessian X'X/n underestimates
    # true curvature by ~mean(y), so y-scaling IS needed.
    _skip_y_scaling_bb = getattr(loss, '_lipschitz_uses_y', False)
    _y_scale = 1.0  # default; overridden below for families that need it
    if not _is_quadratic and not _skip_y_scaling_bb:
        _y_mean, _y_max = _abs_mean_max(y_proc, backend)
        _y_scale = max(1.0, _y_mean, np.sqrt(_y_mean * _y_max))
        if _y_scale > 1.0:
            L = L * _y_scale
    # Inverse Gaussian: gradient scales as 1/mu^3, causing extreme
    # sensitivity to step size.  Use a much more conservative Lipschitz
    # to prevent catastrophic divergence.
    _invgauss_like = getattr(loss, '_inverse_gaussian', False)
    _tweedie_like = getattr(loss, '_tweedie', False)
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
    grad_old = _call_with_weight(loss.gradient, X_proc, y_proc, coef, sample_weight=_sw_arr)
    # Initialize dg for BB step selection (used before first assignment in loop)
    dg = _zeros(n_features, backend, ref_tensor=X_proc)
    iteration = -1  # default if max_iter=0

    # Loop-invariant constants for momentum/BB decisions
    _poisson_like = getattr(loss, '_poisson_like', False)
    _gamma_like = getattr(loss, '_gamma_like', False)

    # --- Pre-compute loop-invariant burn-in and momentum parameters ---
    # These depend only on loss/penalty type, not on iterates.
    if _invgauss_like:
        bb_burn_in = max_iter + 1   # never switch to BB
    elif _tweedie_like:
        bb_burn_in = max(200, max_iter // 2)
    elif _gamma_like:
        bb_burn_in = max(50, max_iter // 8)

    _momentum_disabled = getattr(loss, '_momentum_disabled', False)
    if _momentum_disabled:
        _momentum_burn_in = max_iter + 1   # never use momentum
    elif _tweedie_like:
        _momentum_burn_in = max(100, max_iter // 4)
    elif _gamma_like:
        _momentum_burn_in = max(30, max_iter // 10)
    else:
        _momentum_burn_in = 0  # momentum from the start

    # Conservative momentum for specific loss+penalty combos
    _momentum_beta_cap = getattr(loss, '_momentum_beta_cap', None)
    if _momentum_beta_cap is not None and _poisson_like and not _invgauss_like:
        _pen_name_bb = getattr(penalty, 'name', '')
        if _pen_name_bb in ("l2", "none", "", None):
            _momentum_burn_in = min(100, max_iter)
    if _tweedie_like or _gamma_like:
        if _momentum_beta_cap is None:
            _momentum_beta_cap = 0.2

    for iteration in range(max_iter):
        coef_old = _copy_arr(coef)

        # Gradient at extrapolated point
        grad = _call_with_weight(loss.gradient, X_proc, y_proc, y_k, sample_weight=_sw_arr)

        # Clip extreme gradients -- every iteration, all backends.
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
                _gmax = max(_coef_abs_f * _GRAD_CLIP_COEF_FACTOR + _GRAD_CLIP_ABS_FLOOR, _GRAD_CLIP_MAX)
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
                _obj_val = float(_to_numpy(_call_with_weight(loss.value, X_proc, y_proc, coef, sample_weight=_sw_arr)))
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
                    # No valid iterate yet -- reset to zeros
                    coef = _zeros(n_features, backend, ref_tensor=X_proc)
                y_k = _copy_arr(coef)
                t_k = 1.0
                grad_old = _call_with_weight(loss.gradient, X_proc, y_proc, coef, sample_weight=_sw_arr)
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
                # Pass zero coef -- not all losses handle coef=None.
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
            _steep_loss = getattr(loss, '_steep_loss', False)
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
                    # absolute 1e6 -- NB/Tweedie with large counts can have
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
                    # Step too large -- halve and retry
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
                    grad_old = _call_with_weight(loss.gradient, X_proc, y_proc, coef, sample_weight=_sw_arr)
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
            grad_new = _call_with_weight(loss.gradient, X_proc, y_proc, coef, sample_weight=_sw_arr)

            dw = coef - coef_old
            dg = grad_new - grad_old
            # Batch two dot products into a single GPU->CPU sync.
            dot_dw_dw, dot_dw_dg = _sync_scalars(
                _dot_dev(dw, dw), _dot_dev(dw, dg), backend=backend)
            grad_old = grad_new

        # --- Nesterov momentum with adaptive restart ---
        # bb_burn_in, _momentum_burn_in, _momentum_beta_cap are loop-invariant
        # and computed once before the loop.
        if iteration < _momentum_burn_in:
            t_k = 1.0
            beta = 0.0
            y_k = _copy_arr(coef)   # next gradient at current point, not extrapolated
        elif _momentum_beta_cap is not None:
            # Conservative momentum: fixed small beta to avoid explosion
            beta = _momentum_beta_cap
            y_k = coef + beta * (coef - coef_old)
            t_k = 1.0
        else:
            y_k, t_new = _nesterov_update(coef, coef_old, t_k)
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

        # --- Convergence check -- deferred for GPU, every iteration for CPU. ---
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
