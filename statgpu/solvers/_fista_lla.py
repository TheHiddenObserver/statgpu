"""Fused LLA+FISTA solver for SCAD/MCP over a continuation path.

Runs the entire continuation -> LLA -> FISTA loop in one tight function,
eliminating per-call overhead (backend detect, preprocess, Lipschitz
recompute, array allocation) that accumulates over 300+ fista_solver calls.
"""

__all__ = ["fista_lla_path"]

import copy
import warnings
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, _get_xp, xp_ones
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
from statgpu.penalties._categories import NONSMOOTH as _NONSMOOTH_ALL
from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty
from ._constants import _SLACK_TOLERANCE, _DIVERGE_COEF_NORM_CAP
from ._linesearch import _get_fista_step_compiled, _fista_step_call
from ._utils import (
    _validate_sample_weight,
)

# ---------------------------------------------------------------------------
# Fused proximal kernels for squared_error + AdaptiveL1 (SCAD/MCP via LLA)
# ---------------------------------------------------------------------------
# Pre-computes XtX, Xty to avoid redundant matmul; fuses element-wise ops;
# defers GPU->CPU syncs for convergence.

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


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


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

    Runs the entire continuation -> LLA -> FISTA loop in one tight function,
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
    _is_quadratic = getattr(loss, 'name', '') == 'squared_error'
    _no_momentum = getattr(loss, '_skip_momentum', False)
    _non_smooth_pen_lla = getattr(scad_penalty, 'name', '') in _NONSMOOTH_ALL
    _momentum_beta_cap = getattr(loss, '_momentum_beta_cap', None)
    _conservative_momentum_lla = (
        _momentum_beta_cap is not None
        or (_is_quadratic is False and _non_smooth_pen_lla
            and getattr(loss, 'name', '') in ("logistic", "gamma"))
    )

    n_samples, n_features = X_proc.shape
    _validate_sample_weight(sample_weight, n_samples)

    # --- Intercept handling ---
    # For squared_error (identity link): centering X, y is exact.
    # For GLM losses (log/logit link): centering is WRONG -- it changes
    # the objective.  Instead, augment X with a ones column so the
    # intercept is part of the coefficient vector.
    _augment_intercept = fit_intercept and not _is_quadratic
    if _augment_intercept:
        # Augment X with a column of ones
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
    # Pass zero coef (global bound) -- not all losses handle coef=None.
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
    # At coef=0, mu~1, but near the optimum mu~y.  The Hessian scales
    # with mu, so L_base underestimates by up to max(y).
    # Cap at 10x -- periodic Lipschitz recomputation corrects any remaining
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
                # lla_weights() is now backend-aware -- stays on device
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
                # lla_weights() is now backend-aware -- stays on device
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
                # but y-scaling approximates the Hessian at mu~y -- much tighter.
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
                            q_yk_dev, grad = loss.fused_value_and_gradient(
                                X_c, y_c, y_k, sample_weight=sample_weight,
                            )
                        else:
                            q_yk_dev, grad = loss.fused_value_and_gradient(X_c, y_c, y_k)

                    # Clip gradients (device-side, every 10 iterations)
                    if backend == "numpy" or iteration % 10 == 0:
                        _gn_dev = _norm2_dev(grad)
                        _gsum = _abs_sum_dev(coef_old) * 10.0 + 1e3
                        if backend == "torch":
                            _gmax_dev = xp.clamp(_gsum, min=1e4)
                        else:
                            _gmax_dev = xp.maximum(_gsum, 1e4)
                        # Batch both norms into a single GPU->CPU transfer
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
                            coef = coef_new
                            break
                        L *= 1.5
                        step = 1.0 / L

                    # If all backtracking steps failed, fall back to best known
                    # iterate instead of accepting a potentially worse point.
                    if not _armijo_ok:
                        if _coef_best_lla_inner is not None:
                            coef = _copy_arr(_coef_best_lla_inner)
                        else:
                            coef = _copy_arr(coef_old)  # reject the failed step

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
                            # Wrap scalar as tensor for torch stack compatibility
                            _q_dev = xp.tensor(float(q_new_dev), device=coef.device) if iteration > 0 else xp.tensor(0.0, device=coef.device)
                            _obj_finite_dev = xp.isfinite(_q_dev) if iteration > 0 else xp.tensor(True, device=coef.device)
                        elif backend == "cupy":
                            _finite_dev = xp.isfinite(_cn_dev)
                            _cap_needed_dev = _cn_dev > 5.0
                            _diverge_norm_dev = _cn_dev > _DIVERGE_COEF_NORM_CAP if iteration > 10 else xp.asarray(False)
                            _obj_finite_dev = xp.isfinite(q_new_dev) if iteration > 0 else xp.asarray(True)
                        else:
                            # Batch GPU->CPU sync: transfer 2 scalars instead of 4
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

                        # Batch sync: coef norm + objective (1 GPU->CPU transfer)
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

                    # Periodic Lipschitz recomputation -- corrects stale L
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
