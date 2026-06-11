"""
Unified IRLS solver for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch.
Single implementation works on numpy/cupy/torch backends via auto detection.
"""

import warnings
from typing import Optional

import numpy as np

from statgpu.backends._array_ops import (
    _clip,
    _copy_arr,
    _diag,
    _norm2,
    _solve_linear_system,
    _to_backend,
    _zeros,
)

# IRLS deviance tolerance constants
_IRLS_DEV_TOL_REL = 1e-10      # Relative deviance tolerance
_IRLS_DEV_TOL_ABS = 1e-6       # Absolute deviance tolerance floor


def _to_backend_module(backend):
    """Return the array module (numpy/cupy/torch) for the given backend string."""
    if backend == "torch":
        import torch
        return torch
    elif backend == "cupy":
        import cupy as cp
        return cp
    else:
        return np


def _infer_backend(X):
    """Detect backend from array type."""
    mod = type(X).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _solve(A, b, backend="auto"):
    """Solve linear system, fallback to lstsq if singular."""
    return _solve_linear_system(A, b, backend=backend)


def _norm(x, backend):
    """Compute L2 norm of array."""
    return float(_norm2(x))


# =============================================================================
# Torch.compile for IRLS elementwise chain fusion
# =============================================================================
# When backend is torch on CUDA, the per-iteration elementwise ops
# (link inverse, weight computation, working response, weighted matmul)
# can be fused via torch.compile to reduce kernel launch overhead.

_IRLS_STEP_COMPILED = None


def _torch_compile_supported():
    """Check if torch.compile is safe (CUDA Capability >= 7.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 7
    except Exception:
        pass
    return True


def _get_irls_step_compiled():
    """Lazily create a torch.compile'd IRLS step function."""
    global _IRLS_STEP_COMPILED
    if _IRLS_STEP_COMPILED is not None:
        return _IRLS_STEP_COMPILED

    import torch

    def _irls_weighted_gemm(X, W, z):
        """Weighted X'WX and X'Wz — elementwise ops fused by torch.compile."""
        W_col = W.unsqueeze(1)
        XtWX = X.T @ (X * W_col)
        Xtz = X.T @ (W * z)
        return XtWX, Xtz

    if _torch_compile_supported():
        try:
            _IRLS_STEP_COMPILED = torch.compile(_irls_weighted_gemm, dynamic=True, fullgraph=False)
        except Exception:
            _IRLS_STEP_COMPILED = _irls_weighted_gemm
    else:
        _IRLS_STEP_COMPILED = _irls_weighted_gemm

    return _IRLS_STEP_COMPILED


def _irls_step_call(compiled_fn, *args):
    """Call compiled IRLS step, falling back to eager on GPU arch mismatch."""
    try:
        return compiled_fn(*args)
    except Exception:
        def _irls_gemm_eager(X, W, z):
            W_col = W.unsqueeze(1)
            XtWX = X.T @ (X * W_col)
            Xtz = X.T @ (W * z)
            return XtWX, Xtz
        return _irls_gemm_eager(*args)


def irls_solver(
    family,
    X,
    y,
    max_iter=100,
    tol=1e-4,
    init_coef=None,
    sample_weight=None,
    ridge_alpha=0.0,
    ridge_penalize_intercept=False,
    backend="auto",
    penalty_matrix=None,
):
    """IRLS: solve GLM by iteratively weighted least squares.

    Parameters
    ----------
    family : Family
        GLM family with link/variance/irls_* methods.
    X : array
        Design matrix (n_samples, n_features).
    y : array
        Target (n_samples,).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on parameter change.
    init_coef : array, optional
        Initial coefficient vector.
    sample_weight : array, optional
        Sample weights.
    ridge_alpha : float
        L2 regularization (lambda = 1/(2*C) format).
    ridge_penalize_intercept : bool
        Whether to penalize the intercept.
    backend : str
        'numpy', 'cupy', 'torch', or 'auto'.
    penalty_matrix : array, optional
        Additional penalty matrix to add to the normal equations.
        Shape must be (n_features, n_features). When provided, the
        normal equations become: X'WX + ridge_alpha*I + penalty_matrix.

    Returns
    -------
    params : array
        Fitted parameters.
    n_iter : int
        Number of iterations.
    """
    if backend == "auto":
        backend = _infer_backend(X)

    # Ensure X and y have the same dtype — promote to higher precision
    from statgpu.backends._utils import xp_astype, _to_float_scalar
    _xp_mod = _to_backend_module(backend)
    if hasattr(X, 'dtype') and hasattr(y, 'dtype') and X.dtype != y.dtype:
        # Convert to numpy-compatible dtype strings for result_type
        _xd = str(X.dtype).replace('torch.', '') if 'torch' in str(type(X.dtype)) else X.dtype
        _yd = str(y.dtype).replace('torch.', '') if 'torch' in str(type(y.dtype)) else y.dtype
        target_dtype = np.result_type(_xd, _yd)
        X = xp_astype(X, target_dtype, _xp_mod)
        y = xp_astype(y, target_dtype, _xp_mod)

    if init_coef is None:
        n_features = X.shape[1]
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = init_coef

    # Ensure params dtype matches X to avoid torch matmul dtype errors
    if hasattr(X, 'dtype') and hasattr(params, 'dtype') and params.dtype != X.dtype:
        params = xp_astype(params, X.dtype, _xp_mod)

    if max_iter <= 0:
        return params, 0

    # Pre-compute family constants (outside iteration loop)
    _fname = getattr(family, 'name', '')
    _tweedie_power = float(getattr(family, 'power', 1.5)) if _fname == "tweedie" else 0.0
    _nb_alpha = float(getattr(family, 'alpha', 1.0)) if _fname == "negative_binomial" else 0.0
    _is_constant_W = _fname in ("gamma", "gaussian", "squared_error")
    _y_backend = _to_backend(y, backend, X)

    def _dev_val(mu_arr):
        """Compute family-specific deviance (lower is better).

        Returns device-side value (no GPU→CPU sync) for torch/cupy.
        Uses backend-agnostic operations to avoid triplicated code.
        """
        xp = _to_backend_module(backend)
        y = _y_backend

        # Clip mu for families that require positive mu
        if _fname not in ("gaussian", "squared_error"):
            mu_arr = _clip(mu_arr, 1e-10, None)

        if _fname in ("gaussian", "squared_error"):
            return xp.sum((y - mu_arr) ** 2)
        elif _fname == "gamma":
            return xp.sum(y / mu_arr - xp.log(y / mu_arr) - 1.0)
        elif _fname == "inverse_gaussian":
            return xp.sum((y - mu_arr) ** 2 / (y * mu_arr ** 2))
        elif _fname == "negative_binomial":
            _mu_c = _clip(mu_arr, 1e-10, None)
            _y_c = _clip(y, 1e-10, None)
            _a = _nb_alpha
            return xp.sum(
                2.0 * (_y_c * xp.log(_y_c / _mu_c)
                       - (_y_c + 1.0 / _a) * xp.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c)))
            )
        elif _fname == "tweedie":
            p = _tweedie_power
            if abs(p - 1.0) < 0.01:
                return xp.sum(mu_arr - y * xp.log(mu_arr))
            elif abs(p - 2.0) < 0.01:
                return xp.sum(y / mu_arr - xp.log(y / mu_arr) - 1.0)
            else:
                return xp.sum(
                    y * (xp.power(y, 1.0 - p) - xp.power(mu_arr, 1.0 - p)) / (1.0 - p)
                    - (xp.power(y, 2.0 - p) - xp.power(mu_arr, 2.0 - p)) / (2.0 - p)
                )
        else:
            return xp.sum(mu_arr - y * xp.log(mu_arr))

    for iteration in range(max_iter):
        params_old = _copy_arr(params)

        # Step 1: linear predictor (clip eta to prevent exp overflow)
        # For identity link (squared_error), skip clipping — mu = eta = X@params
        # and clipping distorts the OLS solution.
        eta_raw = X @ params
        _link_name = getattr(family.link, 'name', '')
        if _link_name in ('identity', 'Identity'):
            eta = eta_raw
        else:
            eta = _clip(eta_raw, -30, 30)

        # Step 2: inverse link -> mean (clip mu to prevent extreme weights)
        # For identity link (squared_error), skip clipping — mu = eta.
        mu = family.link.inverse(eta)
        if _link_name not in ('identity', 'Identity'):
            mu = _clip(mu, 1e-10, 1e6)

        # Step 3: IRLS weights
        W = family.irls_weights(mu, y)
        W = _clip(W, 1e-10, None)

        if sample_weight is not None:
            sw = _to_backend(sample_weight, backend, X)
            W = W * sw

        # Step 4: working response
        z = family.irls_working_response(mu, y, eta)

        # Ensure W and z match X dtype (e.g. X=float32, y=float64 -> cast)
        if hasattr(X, 'dtype'):
            if hasattr(W, 'dtype') and W.dtype != X.dtype:
                W = xp_astype(W, X.dtype, _xp_mod)
            if hasattr(z, 'dtype') and z.dtype != X.dtype:
                z = xp_astype(z, X.dtype, _xp_mod)

        # Step 5: weighted least squares (X'WX + lambda*I) params = X'Wz
        if backend == "torch":
            import torch
            W_col = W.unsqueeze(1)
            _compiled_step = _get_irls_step_compiled()
            XtWX, Xtz = _irls_step_call(_compiled_step, X, W, z)
        else:
            if backend == "cupy":
                import cupy as cp
                W_col = W[:, cp.newaxis]
            else:
                W_col = W[:, np.newaxis]
            XtWX = X.T @ (X * W_col)
            Xtz = X.T @ (W * z)

        if ridge_alpha > 0:
            reg = np.full(XtWX.shape[0], ridge_alpha)
            if not ridge_penalize_intercept:
                reg[0] = 0.0
            XtWX = XtWX + _diag(reg, backend, ref_tensor=X)

        # Add penalty matrix if provided (e.g., for spline smoothing)
        if penalty_matrix is not None:
            XtWX = XtWX + _to_backend(penalty_matrix, backend, X)

        params_new = _solve(XtWX, Xtz, backend)

        # Armijo backtracking line search: find step in (0, 1] that
        # gives sufficient decrease in the loss (deviance).

        # Current loss — use only eta clipping (prevent exp overflow),
        # NOT mu clipping (which distorts the deviance landscape).
        # Reuse eta_raw from the current iteration (same as X @ params_old).
        eta_cur = _clip(eta_raw, -30, 30) if _link_name not in ('identity', 'Identity') else eta_raw
        mu_cur = family.link.inverse(eta_cur)
        try:
            dev_old_dev = _dev_val(mu_cur)
        except Exception:
            dev_old_dev = float('inf')

        # Line search: for families with constant IRLS weights (Gaussian,
        # Gamma, InverseGaussian), the IRLS step IS the Newton step on the
        # GLM loss, and the Hessian is constant X'X/n.  Accept full step.
        # For variable-weight families (Poisson, Logistic, Tweedie),
        # use Armijo backtracking on the deviance.
        _direction = params_new - params_old

        # Convert dev_old to Python float for tolerance computation
        # (single sync per iteration, not per line-search step)
        dev_old_f = _to_float_scalar(dev_old_dev)
        _dev_tol = max(abs(dev_old_f) * _IRLS_DEV_TOL_REL, _IRLS_DEV_TOL_ABS)

        def _dev_accept(dev_try_dev):
            """Check if trial deviance is acceptable (device-side NaN + comparison)."""
            _dev_f = _to_float_scalar(dev_try_dev)  # single GPU→CPU sync
            if _dev_f != _dev_f:  # NaN check
                return False
            return _dev_f <= dev_old_f + _dev_tol

        if _is_constant_W:
            # Constant weights: IRLS = Newton.  Try full step first;
            # if deviance increases significantly, fall back to Armijo.
            eta_new = _clip(X @ params_new, -30, 30)
            mu_new = family.link.inverse(eta_new)
            try:
                dev_new_dev = _dev_val(mu_new)
            except Exception:
                dev_new_dev = float('inf')
            if _dev_accept(dev_new_dev):
                params = params_new
            else:
                step = 1.0
                _accepted = False
                for _bt in range(30):
                    params_try = params_old + step * _direction
                    eta_try = _clip(X @ params_try, -30, 30)
                    mu_try = family.link.inverse(eta_try)
                    try:
                        dev_try_dev = _dev_val(mu_try)
                    except Exception:
                        step *= 0.5
                        continue
                    if _dev_accept(dev_try_dev):
                        _accepted = True
                        break
                    step *= 0.5
                params = params_try if _accepted else params_old
        else:
            # Variable weights: Armijo backtracking on deviance
            step = 1.0
            _accepted = False
            for _bt in range(30):
                params_try = params_old + step * _direction
                eta_try = _clip(X @ params_try, -30, 30)
                mu_try = family.link.inverse(eta_try)
                try:
                    dev_try_dev = _dev_val(mu_try)
                except Exception:
                    step *= 0.5
                    continue
                if _dev_accept(dev_try_dev):
                    _accepted = True
                    break
                step *= 0.5

            if _accepted:
                params = params_try
            else:
                params = params_old

        # Convergence: gradient norm check (most reliable for all families)
        if iteration % 5 == 4 or iteration == max_iter - 1:
            try:
                grad_f = family.gradient(X, y, params)
                if ridge_alpha > 0:
                    # Match normal equations: XtWX + ridge_alpha*I, so penalty
                    # gradient is ridge_alpha * params (not ridge_alpha/n)
                    grad_f[1:] = grad_f[1:] + ridge_alpha * params[1:]
                grad_norm = float(_norm(grad_f, backend))
            except Exception:
                # No gradient method available — fall back to param change
                _param_change = float(_norm(params - params_old, backend))
                _param_norm = max(float(_norm(params, backend)), 1.0)
                grad_norm = _param_change / _param_norm  # relative change
            if grad_norm < tol:
                break

    n_iter = iteration + 1
    if n_iter >= max_iter:
        from statgpu.glm_core._solver import ConvergenceWarning
        warnings.warn(
            f"irls did not converge within {max_iter} iterations "
            f"(family={getattr(family, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    return params, n_iter


class IRLSSolver:
    """Unified IRLS solver: each iteration solves weighted least squares.

    Supports numpy / cupy / torch backends (auto-detect X type).
    """

    def __init__(self, family, max_iter=100, tol=1e-4):
        self.family = family
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X,
        y,
        init_coef=None,
        sample_weight=None,
        ridge_alpha=0.0,
        ridge_penalize_intercept=False,
        backend="auto",
        penalty_matrix=None,
    ):
        """Run IRLS loop.

        Parameters
        ----------
        ridge_alpha : float
            L2 regularization (lambda = 1/(2*C) format).
        ridge_penalize_intercept : bool
            Whether to penalize the intercept.
        penalty_matrix : array, optional
            Additional penalty matrix for the normal equations.
        """
        return irls_solver(
            self.family,
            X,
            y,
            max_iter=self.max_iter,
            tol=self.tol,
            init_coef=init_coef,
            sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=ridge_penalize_intercept,
            backend=backend,
            penalty_matrix=penalty_matrix,
        )
