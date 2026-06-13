"""Legacy solver methods from _solver.py.

DO NOT import in production code."""

from __future__ import annotations

import numpy as np

def fista_sqerr_adaptive_l1_fused(
    X, y, penalty_weights, alpha,
    XtX, Xty, yty, n_samples,
    L_init, max_iter, tol,
    backend, no_momentum=False,
):
    """Fused FISTA for squared_error + AdaptiveL1 with pre-computed XtX/Xty.

    Eliminates:
    - Redundant X@coef matmul (uses XtX instead)
    - GPU→CPU syncs (convergence check deferred)
    - Element-wise kernel overhead (fused update+proximal+momentum)

    Parameters
    ----------
    X, y : array (centered)
    penalty_weights : array (p,) — LLA weights
    alpha : float — penalty alpha
    XtX, Xty, yty : pre-computed
    n_samples : int
    L_init : float — initial Lipschitz
    max_iter, tol : FISTA params
    backend : 'torch' or 'cupy'
    no_momentum : bool

    Returns
    -------
    coef : array (p,)
    n_iter : int
    """
    p = XtX.shape[0]
    step = 1.0 / L_init
    L = L_init

    if backend == "torch":
        import torch
        thresh = torch.tensor(
            alpha * penalty_weights * step,
            device=XtX.device, dtype=XtX.dtype,
        )
        coef = torch.zeros(p, device=XtX.device, dtype=XtX.dtype)
        coef_old = coef.clone()
        y_k = coef.clone()
        _fused = _get_sqerr_proximal_torch()
        # Pre-allocate for momentum-free case
        _zero_beta = 0.0
    else:
        import cupy as cp
        thresh = cp.asarray(alpha * penalty_weights * step, dtype=cp.float64)
        coef = cp.zeros(p, dtype=cp.float64)
        coef_old = coef.copy()
        y_k = coef.copy()
        _fused = _get_sqerr_proximal_cupy()
        _zero_beta = 0.0

    t_k = 1.0
    _sync_interval = 10  # Only check convergence every N iterations

    iteration = -1  # default if max_iter=0
    for iteration in range(max_iter):
        # Gradient: grad = (XtX @ y_k - Xty) / n
        grad = (XtX @ y_k - Xty) / n_samples

        # Clip gradients (avoid sync — do it on GPU)
        if iteration % 10 == 0:
            grad = _clip_grad_on_device(grad, coef_old, backend)

        # Proximal gradient step (no backtracking — Lipschitz is exact for squared_error)
        # Pre-compute momentum coefficient so the fused kernel can apply it in one pass.
        if no_momentum:
            beta_mom = 0.0
        else:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            beta_mom = (t_k - 1.0) / t_new
        coef_new, y_k = _fused(y_k, grad, step, thresh, coef_old, beta_mom)
        coef = coef_new

        # Momentum state update
        if not no_momentum:
            t_k = t_new

        # Convergence check (device-side, minimal sync)
        if iteration < 20 or iteration % _sync_interval == 0:
            coef_diff_dev = _abs_sum_dev(coef - coef_old)
            if _to_float_scalar(coef_diff_dev) < tol:
                break

        coef_old = _copy_arr(coef)

    return _to_numpy(coef), iteration + 1


