"""Proximal IRLS-CD solver for Quantile + SCAD/MCP.

Combines IRLS quadratic majorization of the quantile loss with LLA
(Local Linear Approximation) for non-convex penalties (SCAD/MCP),
solved via coordinate descent for exact sparsity.

Three-backend support: numpy (CPU), cupy (CUDA), torch (CUDA/CPU).

Algorithm:
  For each continuation alpha:
    For each LLA iteration:
      1. Compute LLA weights from SCAD/MCP derivative at current beta
      2. For each IRLS iteration:
         a. Compute IRLS weights: w_i = tau_i / max(|r_i|, eps)
         b. CD sweep on Q(beta) + penalty (batch on all backends)
         c. Check convergence
      3. Check LLA convergence

References:
- Frasso & Bohning (2025): cirls package (Constrained IRLS for GLMs)
- Wu & Liu (2009): Variable selection in quantile regression
- Hunter & Li (2005): MM algorithms for nonconvex penalized estimation
"""

__all__ = ["proximal_irls_quantile_solver"]

import copy
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy


def proximal_irls_quantile_solver(
    loss,
    penalty,
    X,
    y,
    alpha_path,
    max_lla_per_step=2,
    lla_tol=1e-6,
    max_iter=None,
    tol=1e-6,
    fit_intercept=True,
    sample_weight=None,
):
    """Proximal IRLS-CD solver for quantile regression with nonconvex penalty.

    Supports numpy / cupy / torch backends.

    Parameters
    ----------
    loss : QuantileLoss
        Quantile loss object.
    penalty : Penalty
        Nonconvex penalty (SCAD, MCP).
    X : array (n, p)
        Design matrix (NO intercept column -- intercept is handled by centering).
    y : array (n,)
        Response variable.
    alpha_path : array
        Continuation path from lambda_max to target alpha.
    max_lla_per_step : int
        Maximum LLA iterations per continuation step.
    lla_tol : float
        LLA convergence tolerance.
    max_iter : int or list
        Maximum IRLS iterations per continuation step.
    tol : float
        IRLS convergence tolerance.
    fit_intercept : bool
        Whether to fit an intercept (via centering).
    sample_weight : array (n,), optional
        Sample weights. If provided, the IRLS weights are scaled accordingly.

    Returns
    -------
    coef : array (p,)
        Optimized coefficients (without intercept).
    intercept : float
        Intercept value (0 if fit_intercept=False).
    total_iter : int
        Total number of IRLS iterations.
    """
    backend = _resolve_backend("auto", X)
    n, p = X.shape
    tau = loss._tau
    eps = 1e-8

    # Get backend module
    if backend == "torch":
        import torch as xp
    elif backend == "cupy":
        import cupy as xp
    else:
        xp = np

    # Ensure float64 for numerical stability
    X_dev = xp.asarray(X, dtype=xp.float64)
    y_dev = xp.asarray(y, dtype=xp.float64)

    # Handle sample_weight
    if sample_weight is not None:
        sw = xp.asarray(sample_weight, dtype=xp.float64)
        # Ensure sw is on the same device as X_dev (for torch CUDA)
        if hasattr(X_dev, 'device') and hasattr(sw, 'to'):
            sw = sw.to(device=X_dev.device)
        sw_sum = float(_to_numpy(xp.sum(sw)))
        # Normalize so sum(sw) = n (keeps penalty scale consistent)
        sw = sw * (n / sw_sum)
    else:
        sw = None

    # Center X and y for intercept
    if fit_intercept:
        if sw is not None:
            X_mean = xp.sum(X_dev * sw[:, None], axis=0) / n
            y_mean_dev = xp.sum(y_dev * sw) / n
        else:
            X_mean = xp.mean(X_dev, axis=0)
            y_mean_dev = xp.mean(y_dev)
        y_mean = float(_to_numpy(y_mean_dev))
        X_work = X_dev - X_mean
        y_work = y_dev - y_mean_dev
    else:
        X_mean = None
        y_mean = 0.0
        X_work = X_dev
        y_work = y_dev

    n_features = p

    # Initialize coefficients with OLS on centered data
    if backend == "torch":
        beta = xp.linalg.lstsq(X_work, y_work).solution
    else:
        beta = xp.linalg.lstsq(X_work, y_work, rcond=None)[0]

    total_iter = 0

    # Precompute X^2 for weighted Hessian diagonal (reused each IRLS step)
    X_sq = X_work * X_work  # (n, p)

    for cont_i, cont_alpha in enumerate(alpha_path):
        pen_step = copy.copy(penalty)
        pen_step.alpha = float(cont_alpha)
        _mi = max_iter[cont_i] if isinstance(max_iter, (list, tuple)) else max_iter

        for lla_i in range(max_lla_per_step):
            # LLA weights = P'(|beta_j|) — penalty derivative at current coef.
            # For SCAD near-zero: alpha; middle: (a*alpha-|beta|)/(a-1); large: 0
            # For MCP: max(0, alpha - |beta|/gamma)
            #
            # WLS uses un-normalized g = X'@wr and h = sum(X^2 * w),
            # so threshold must be scaled by n to match:
            #   min (1/2n) * sum(w * (y - X@beta)^2) + sum(P'(|beta_j|) * |beta_j|)
            #   => g/h are un-normalized, thresh = n * P'(|beta_j|)
            lla_w = _compute_lla_weights(pen_step, beta, n_features, xp, backend)
            thresh = n * lla_w  # (p,) per-coordinate threshold

            beta_before_lla = _copy(beta)

            # IRLS-CD inner loop
            for irls_iter in range(_mi):
                beta_old = _copy(beta)

                # Residuals and IRLS weights
                # w_i = tau_i / max(|r_i|, eps) where tau_i = tau if r_i>=0 else 1-tau
                r = y_work - X_work @ beta
                abs_r = xp.abs(r)
                abs_r_safe = xp.maximum(abs_r, xp.asarray(eps, dtype=abs_r.dtype))
                pos_mask = (r >= 0).to(dtype=abs_r.dtype) if backend == "torch" else (r >= 0).astype(abs_r.dtype)
                tau_vec = tau * pos_mask + (1.0 - tau) * (1.0 - pos_mask)
                w = tau_vec / abs_r_safe  # (n,), always positive

                # Apply sample weights
                if sw is not None:
                    w = w * sw

                # Clamp IRLS weights to prevent numerical overflow
                # Max weight = 100 / eps ensures h_j doesn't exceed ~100 * X'X_diag
                w_max = 100.0 / eps
                w = xp.minimum(w, xp.asarray(w_max, dtype=w.dtype))

                # Parallel diagonal majorization step (Jacobi-style)
                beta = _parallel_majorization_step(
                    X_work, X_sq, y_work, w, beta, thresh,
                    n_features, eps, xp, backend)

                # Convergence check
                delta_dev = xp.abs(beta - beta_old)
                delta = float(_to_numpy(xp.max(delta_dev)))
                total_iter += 1

                if delta < tol:
                    break

            # LLA convergence check
            lla_delta_dev = xp.abs(beta - beta_before_lla)
            lla_delta = float(_to_numpy(xp.max(lla_delta_dev)))
            if lla_delta < lla_tol:
                break

    # Reconstruct intercept
    coef_np = _to_numpy(beta).astype(np.float64)
    if fit_intercept:
        X_mean_np = _to_numpy(X_mean).astype(np.float64)
        intercept = float(y_mean - X_mean_np @ coef_np)
    else:
        intercept = 0.0

    return coef_np, intercept, total_iter


# ── Parallel diagonal majorization step (all backends) ─────────────

def _parallel_majorization_step(X, X_sq, y, w, beta, thresh, p, eps, xp, backend):
    """One parallel diagonal majorization step (Jacobi-style update).

    Computes:
      g = X' @ diag(w) @ (y - X @ beta)    -- weighted gradient vector
      h = diag(X' @ diag(w) @ X)            -- weighted Hessian diagonal
      beta = S(g + h * beta, thresh) / h    -- soft-threshold update

    Note: This is a Jacobi-style parallel update, not cyclic coordinate descent.
    All coordinates are updated simultaneously using "old" beta values.
    For strongly correlated designs, convergence may differ from true CD.
    GPU-friendly: only matrix operations, no per-coordinate kernel launches.
    """
    r = y - X @ beta  # (n,)
    wr = w * r         # (n,)

    # Weighted gradient: g = X' @ (w * r)  -- O(np)
    g = X.T @ wr       # (p,)

    # Weighted Hessian diagonal: h = sum(X^2 * w, axis=0)  -- O(np)
    w_col = w[:, None] if w.ndim == 1 else w  # (n,1) for broadcast
    h = xp.sum(X_sq * w_col, axis=0)           # (p,)
    h = xp.maximum(h, eps)

    # Soft-threshold update: beta = S(g + h*beta, thresh) / h
    # S(x, t) = sign(x) * max(|x| - t, 0)
    u = g + h * beta
    abs_u = xp.abs(u)
    sign_u = xp.sign(u)
    beta_new = sign_u * xp.maximum(abs_u - thresh, xp.asarray(0.0, dtype=abs_u.dtype)) / h

    return beta_new


# ── Helpers ─────────────────────────────────────────────────────────

def _copy(arr):
    """Backend-aware copy."""
    if hasattr(arr, 'clone'):
        return arr.clone()
    return arr.copy()


def _compute_lla_weights(penalty, coef, p, xp, backend):
    """Compute LLA weights from current coefficients (stays on GPU).

    For SCAD: w_j = alpha * min(1, max(0, (a*alpha - |beta_j|) / (a*alpha - alpha)))
    For MCP: w_j = max(0, alpha - |beta_j| / gamma)

    All computation done on-device to avoid GPU->CPU->GPU round-trip.
    """
    alpha = penalty.alpha
    abs_coef = xp.abs(coef[:p])

    pen_name = getattr(penalty, 'name', '').lower()

    if 'scad' in pen_name:
        a = getattr(penalty, 'a', 3.7)
        denom = a * alpha - alpha
        if abs(denom) < 1e-15:
            # Degenerate case: a ~ 1, fall back to L1
            w = xp.full(p, alpha, dtype=xp.float64)
        else:
            v = (a * alpha - abs_coef) / denom
            v = xp.clip(v, 0.0, 1.0)
            w = alpha * v
    elif 'mcp' in pen_name:
        gamma = getattr(penalty, 'gamma', 3.0)
        w = xp.maximum(xp.asarray(0.0, dtype=xp.float64), alpha - abs_coef / gamma)
    else:
        w = xp.full(p, alpha, dtype=xp.float64)

    return w
