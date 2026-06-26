"""Fused LLA+FISTA solver for SCAD/MCP over a continuation path.

Runs the entire continuation -> LLA -> FISTA loop in one tight function,
eliminating per-call overhead (backend detect, preprocess, Lipschitz
recompute, array allocation) that accumulates over 300+ fista_solver calls.
"""

__all__ = ["fista_lla_path"]

import copy
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, xp_ones
from statgpu.backends._array_ops import (
    _abs_sum_dev,
    _clip_grad_on_device,
    _copy_arr,
    _norm2_dev,
    _sync_scalars,
    _zeros,
)
from statgpu.penalties._categories import NONSMOOTH as _NONSMOOTH_ALL
from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty
from ._constants import (
    _DIVERGE_COEF_NORM_CAP,
    _GRAD_CLIP_COEF_FACTOR,
    _GRAD_CLIP_ABS_FLOOR,
    _GRAD_CLIP_MAX,
)
from ._utils import (
    _nesterov_momentum,
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
    _is_quadratic = getattr(loss, '_is_quadratic', False)
    _no_momentum = getattr(loss, '_skip_momentum', False)
    _non_smooth_pen_lla = getattr(scad_penalty, 'name', '') in _NONSMOOTH_ALL
    _momentum_beta_cap = getattr(loss, '_momentum_beta_cap', None)
    _conservative_momentum_lla = (
        _momentum_beta_cap is not None
        or (getattr(loss, '_conservative_momentum_with_nonsmooth', False)
            and _non_smooth_pen_lla)
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
    # Must gate on sample_weight is None because the fused path uses
    # unweighted Gram matrix (XtX, Xty) which is incorrect for weighted data.
    if _is_quadratic and backend in ("torch", "cupy") and sample_weight is None:
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
                        beta_mom, t_k = _nesterov_momentum(t_k)

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
        # Pre-compute XtX and Xty for squared_error (avoids redundant matmuls).
        # Must gate on sample_weight is None because XtX/Xty are unweighted.
        _use_xtx = _is_quadratic and backend == "numpy" and sample_weight is None
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

                # --- FISTA inner solve (fixed-step, no backtracking) ---
                y_k = _copy_arr(coef)
                t_k = 1.0
                L = L_base

                # Get fused proximal+momentum kernel for GPU paths
                if backend == "torch":
                    _fused_update = _get_sqerr_proximal_torch()
                elif backend == "cupy":
                    _fused_update = _get_sqerr_proximal_cupy()
                else:
                    _fused_update = None
                step = 1.0 / L

                # Pre-compute device-side tolerance for convergence check
                if backend != "numpy":
                    _tol_dev = xp.asarray(tol)

                # --- Async inner loop: skip backtracking, use fixed step ---
                # For LLA, the Lipschitz constant L is pre-computed and stable.
                # Backtracking is unnecessary — use fixed step 1/L.
                # This eliminates per-iteration GPU→CPU syncs.
                for iteration in range(_mi):
                    coef_old = _copy_arr(coef)

                    if _use_xtx:
                        grad = (XtX @ y_k - Xty) / n_samples
                    else:
                        if sample_weight is not None:
                            _, grad = loss.fused_value_and_gradient(
                                X_c, y_c, y_k, sample_weight=sample_weight,
                            )
                        else:
                            _, grad = loss.fused_value_and_gradient(X_c, y_c, y_k)

                    # Clip gradients (device-side, every 10 iterations)
                    if backend == "numpy" or iteration % 10 == 0:
                        _gn_dev = _norm2_dev(grad)
                        _gsum = _abs_sum_dev(coef_old) * _GRAD_CLIP_COEF_FACTOR + _GRAD_CLIP_ABS_FLOOR
                        if backend == "torch":
                            _gmax_dev = xp.clamp(_gsum, min=_GRAD_CLIP_MAX)
                        else:
                            _gmax_dev = xp.maximum(_gsum, _GRAD_CLIP_MAX)
                        _gn_f, _gmax_f = _sync_scalars(_gn_dev, _gmax_dev, backend=backend)
                        if _gn_f > _gmax_f:
                            grad = grad * (_gmax_dev / _gn_dev)


                    # Compute momentum beta before fused update
                    if _no_momentum:
                        beta_mom = 0.0
                    elif _conservative_momentum_lla:
                        beta_mom, t_k = _nesterov_momentum(t_k, beta_cap=0.5)
                    else:
                        beta_mom, t_k = _nesterov_momentum(t_k)

                    # Fused proximal + momentum: single kernel launch on GPU
                    # Combines: w_tilde = y_k - step*grad
                    #           coef = proximal(w_tilde, step)  [weighted soft-threshold]
                    #           y_k = coef + beta * (coef - coef_old)
                    # Reduces 3 kernel launches to 1.
                    # NOTE: fused update only works for coordinate-wise penalties
                    # (AdaptiveL1Penalty with _weights), NOT group penalties.
                    if (_fused_update is not None and backend != "numpy"
                            and hasattr(inner_pen, '_weights')):
                        # Ensure thresh is on the correct device
                        _w = inner_pen._weights
                        if isinstance(_w, np.ndarray):
                            _w = xp.asarray(_w, dtype=coef.dtype)
                        thresh = _w * inner_pen.alpha * step
                        coef, y_k = _fused_update(y_k, grad, step, thresh, coef_old, beta_mom)
                    else:
                        w_tilde = y_k - step * grad
                        coef = inner_pen.proximal(w_tilde, step, backend=backend)
                        y_k = coef + beta_mom * (coef - coef_old)

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
                    # as coef moves away from zero.
                    if not _is_quadratic and iteration > 0 and iteration % 20 == 0:
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
