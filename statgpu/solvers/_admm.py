"""ADMM solver for penalized GLM optimization.

Reformulates min_w f(Xw; y) + p(w) as a consensus ADMM problem and solves
via alternating direction method of multipliers. The w-update (smooth
subproblem) uses either a direct Cholesky solve (for squared-error loss with
moderate dimensionality) or Nesterov-accelerated gradient descent. The z-update
reuses the penalty proximal operator and is element-wise / GPU-friendly.
"""

import warnings

import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._array_ops import (
    _abs_sum_dev,
    _copy_arr,
    _device_leq,
    _norm2_dev,
    _sync_scalars,
    _zeros,
    _zeros_like,
)
from ._convergence import ConvergenceWarning
from ._utils import (
    _nesterov_momentum,
    _validate_uniform_sample_weight,
)

__all__ = ["admm_solver"]


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
        g = loss.gradient(X_proc, y_proc, w_vec, sample_weight=sample_weight)
        g = g + rho * (w_vec - z_cur + u_cur)
        return g

    # Detect if loss supports Cholesky (constant Hessian, e.g. squared_error).
    # For GLM losses, use Nesterov-accelerated gradient descent.
    # When using Cholesky we pin rho (disable adaptive_rho) because the
    # precomputed _A_mat = XtX/n + rho*I would become stale if rho changed.
    use_cholesky = getattr(loss, '_supports_cholesky', False) and n_features <= 2000
    if use_cholesky:
        adaptive_rho = False

    if use_cholesky:
        _hess_const = loss.hessian(X_proc, y_proc, w)  # XtX / n
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
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                # Matrix not positive-definite (numerical issues, collinear features)
                # Fall back to CG solver below
                _cholesky_ok = False
        if not _cholesky_ok:
            use_cholesky = False

        # Precompute -grad_f(0) = Xty/n for squared_error (the constant part)
        _zero_coef = _zeros_like(w)
        _neg_grad_zero = -loss.gradient(X_proc, y_proc, _zero_coef, sample_weight=sample_weight)  # Xty/n

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
                beta_mom, t_mom = _nesterov_momentum(t_mom)
                w_mom = w_next + beta_mom * (w_next - w_new)
                w_new = w_next
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
