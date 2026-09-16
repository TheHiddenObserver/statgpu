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
        Design matrix without an intercept column. When ``fit_intercept=True``
        the solver augments the numerical design with an unpenalized ones
        column so the intercept is optimized in the pinball objective.
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
        Whether to fit an unpenalized intercept coordinate.
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

    # Quantile regression is not translation-equivalent under mean centering:
    # the intercept must minimize the asymmetric pinball objective itself.
    # Treat it as one additional, unpenalized coordinate. This matches the
    # non-quadratic intercept contract used by the maintained FISTA-LLA path.
    if fit_intercept:
        if backend == "torch":
            ones = xp.ones((n, 1), dtype=X_dev.dtype, device=X_dev.device)
        else:
            ones = xp.ones((n, 1), dtype=X_dev.dtype)
        X_work = xp.concatenate([X_dev, ones], axis=1)
        y_work = y_dev
        n_work_features = p + 1
    else:
        X_work = X_dev
        y_work = y_dev
        n_work_features = p

    n_features = p  # Penalized feature coordinates only; intercept is excluded.

    # Initialize coefficients with OLS on the numerical design. For the
    # intercept-bearing route this initializes the augmented intercept too.
    if backend == "torch":
        beta = xp.linalg.lstsq(X_work, y_work).solution
    else:
        beta = xp.linalg.lstsq(X_work, y_work, rcond=None)[0]

    total_iter = 0

    # Precompute X^2 for weighted Hessian diagonal (reused each IRLS step)
    X_sq = X_work * X_work

    for cont_i, cont_alpha in enumerate(alpha_path):
        pen_step = copy.copy(penalty)
        pen_step.alpha = float(cont_alpha)
        _mi = max_iter[cont_i] if isinstance(max_iter, (list, tuple)) else max_iter

        for lla_i in range(max_lla_per_step):
            # LLA weights = P'(|beta_j|) for feature coefficients only.
            lla_w = _compute_lla_weights(pen_step, beta, n_features, xp, backend)
            feature_thresh = n * lla_w
            if fit_intercept:
                # The final augmented coordinate is the intercept and must not
                # receive SCAD/MCP shrinkage.
                intercept_thresh = xp.zeros(1, dtype=feature_thresh.dtype)
                if backend == "torch":
                    intercept_thresh = intercept_thresh.to(device=feature_thresh.device)
                thresh = xp.concatenate([feature_thresh, intercept_thresh])
            else:
                thresh = feature_thresh

            beta_before_lla = _copy(beta)

            # If every SCAD/MCP derivative is exactly zero, this LLA surrogate
            # has no active penalty at all. Reuse the loss-owned full weighted
            # Quantile IRLS implementation rather than continuing the diagonal
            # Jacobi approximation. This is the same maintained numerical
            # kernel used by ordinary smooth Quantile IRLS; only genuinely
            # penalized LLA steps remain on the Proximal IRLS-CD inner loop.
            zero_penalty = bool(_to_numpy(xp.all(lla_w == 0)))
            if zero_penalty:
                beta, used_iter = loss.irls(
                    X_work,
                    y_work,
                    penalty=None,
                    max_iter=_mi,
                    tol=min(tol, 1e-8),
                    init_coef=beta,
                    eps=eps,
                    sample_weight=sw,
                    fit_intercept=fit_intercept,
                )
                total_iter += used_iter
            else:
                # IRLS-CD inner loop
                for irls_iter in range(_mi):
                    beta_old = _copy(beta)

                    # Residuals and IRLS weights
                    r = y_work - X_work @ beta
                    abs_r = xp.abs(r)
                    abs_r_safe = xp.maximum(
                        abs_r, _scalar_like(eps, abs_r, xp, backend)
                    )
                    pos_mask = (r >= 0).to(dtype=abs_r.dtype) if backend == "torch" else (r >= 0).astype(abs_r.dtype)
                    tau_vec = tau * pos_mask + (1.0 - tau) * (1.0 - pos_mask)
                    w = tau_vec / abs_r_safe  # (n,), always positive

                    # Apply sample weights
                    if sw is not None:
                        w = w * sw

                    # Clamp IRLS weights to prevent numerical overflow
                    w_max = 100.0 / eps
                    w = xp.minimum(w, _scalar_like(w_max, w, xp, backend))

                    # Parallel diagonal majorization step (Jacobi-style)
                    beta = _parallel_majorization_step(
                        X_work, X_sq, y_work, w, beta, thresh,
                        n_work_features, eps, xp, backend)

                    total_iter += 1

                    # Stopping semantics are part of the non-convex algorithm, not
                    # a backend performance knob. Keep the comparison backend-native
                    # but apply the same criterion every iteration on every backend.
                    delta_dev = xp.abs(beta - beta_old)
                    if backend in ("torch", "cupy"):
                        if bool(_to_numpy(
                            xp.max(delta_dev)
                            < _scalar_like(tol, delta_dev, xp, backend)
                        )):
                            break
                    else:
                        if float(_to_numpy(xp.max(delta_dev))) < tol:
                            break

            # LLA convergence check — GPU comparison stays on device
            lla_delta_dev = xp.abs(beta - beta_before_lla)
            if backend in ("torch", "cupy"):
                if bool(_to_numpy(
                    xp.max(lla_delta_dev)
                    < _scalar_like(lla_tol, lla_delta_dev, xp, backend)
                )):
                    break
            else:
                if float(_to_numpy(xp.max(lla_delta_dev))) < lla_tol:
                    break

    beta_np = _to_numpy(beta).astype(np.float64)
    if fit_intercept:
        coef_np = beta_np[:p]
        intercept = float(beta_np[p])
    else:
        coef_np = beta_np
        intercept = 0.0

    return coef_np, intercept, total_iter


# ── Parallel diagonal majorization step (all backends) ─────────────

def _parallel_majorization_step(X, X_sq, y, w, beta, thresh, p, eps, xp, backend):
    """One parallel diagonal majorization step (Jacobi-style update).

    Computes:
      g = X' @ diag(w) @ (y - X @ beta)    -- weighted gradient vector
      h = diag(X' @ diag(w) @ X)            -- weighted Hessian diagonal
      beta = S(g + h * beta, thresh) / h    -- soft-threshold update

    ``thresh`` is zero for any unpenalized coordinate such as an augmented
    intercept. This is a Jacobi-style parallel update, not cyclic coordinate
    descent: all coordinates are updated simultaneously using old beta values.
    GPU-friendly: only matrix operations, no per-coordinate kernel launches.
    """
    r = y - X @ beta  # (n,)
    wr = w * r         # (n,)

    # Weighted gradient: g = X' @ (w * r)  -- O(np)
    g = X.T @ wr

    # Weighted Hessian diagonal: h = sum(X^2 * w, axis=0)  -- O(np)
    w_col = w[:, None] if w.ndim == 1 else w
    h = xp.sum(X_sq * w_col, axis=0)
    h = xp.maximum(h, _scalar_like(eps, h, xp, backend))

    # Soft-threshold update: beta = S(g + h*beta, thresh) / h
    # S(x, t) = sign(x) * max(|x| - t, 0)
    u = g + h * beta
    abs_u = xp.abs(u)
    sign_u = xp.sign(u)
    beta_new = sign_u * xp.maximum(
        abs_u - thresh, _scalar_like(0.0, abs_u, xp, backend)
    ) / h

    return beta_new


# ── Helpers ─────────────────────────────────────────────────────────

def _copy(arr):
    """Backend-aware copy."""
    if hasattr(arr, 'clone'):
        return arr.clone()
    return arr.copy()


def _scalar_like(value, ref, xp, backend):
    """Create a scalar on the same dtype/device as a backend-native array."""
    if backend == "torch":
        return xp.tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


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
            w = xp.full_like(abs_coef, alpha)
        else:
            v = (a * alpha - abs_coef) / denom
            v = xp.clip(v, 0.0, 1.0)
            w = alpha * v
    elif 'mcp' in pen_name:
        gamma = getattr(penalty, 'gamma', 3.0)
        w = xp.maximum(
            _scalar_like(0.0, abs_coef, xp, backend),
            alpha - abs_coef / gamma,
        )
    else:
        w = xp.full_like(abs_coef, alpha)

    return w
