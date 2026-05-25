"""
Unified IRLS solver for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch.
Single implementation works on numpy/cupy/torch backends via auto detection.
"""

import warnings
from typing import Optional

import numpy as np


from statgpu.backends import _resolve_backend
from statgpu.backends._array_ops import (
    _clip,
    _copy_arr,
    _diag,
    _norm2,
    _solve_linear_system,
    _to_backend,
    _zeros,
)


def _promote_torch_irls_inputs(X, y, init_coef=None, sample_weight=None):
    """Keep all Torch IRLS operands on one floating dtype/device."""
    import torch

    def _floating_dtype(arr):
        if hasattr(arr, "is_floating_point") and arr.is_floating_point():
            return arr.dtype
        return torch.float64

    dtype = torch.promote_types(_floating_dtype(X), _floating_dtype(y))
    if init_coef is not None:
        dtype = torch.promote_types(dtype, _floating_dtype(init_coef))
    if sample_weight is not None:
        dtype = torch.promote_types(dtype, _floating_dtype(sample_weight))

    device = X.device
    X = X.to(device=device, dtype=dtype)
    y = torch.as_tensor(y, device=device, dtype=dtype)
    if init_coef is not None:
        init_coef = torch.as_tensor(init_coef, device=device, dtype=dtype)
    if sample_weight is not None:
        sample_weight = torch.as_tensor(sample_weight, device=device, dtype=dtype)
    return X, y, init_coef, sample_weight


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

    Returns
    -------
    params : array
        Fitted parameters.
    n_iter : int
        Number of iterations.
    """
    if backend == "auto":
        backend = _resolve_backend("auto", X)

    if backend == "torch":
        X, y, init_coef, sample_weight = _promote_torch_irls_inputs(
            X, y, init_coef=init_coef, sample_weight=sample_weight
        )
    else:
        y = _to_backend(y, backend, X)
        if sample_weight is not None:
            sample_weight = _to_backend(sample_weight, backend, X)

    if init_coef is None:
        n_features = X.shape[1]
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = init_coef

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
            W = W * sample_weight

        # Step 4: working response
        z = family.irls_working_response(mu, y, eta)

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

        params_new = _solve_linear_system(XtWX, Xtz, backend)

        # Armijo backtracking line search: find step in (0, 1] that
        # gives sufficient decrease in the loss (deviance).
        _fname = getattr(family, 'name', '')
        _tweedie_power = float(getattr(family, 'power', 1.5)) if _fname == "tweedie" else 0.0
        _nb_alpha = float(getattr(family, 'alpha', 1.0)) if _fname == "negative_binomial" else 0.0

        def _dev_val(mu_arr):
            """Compute family-specific deviance (lower is better).

            Returns device-side value (no GPU→CPU sync) for torch/cupy.
            Correct Tweedie deviance for power p (p != 1, p != 2):
              d(y, mu) = y*(y^(1-p) - mu^(1-p))/(1-p) - (y^(2-p) - mu^(2-p))/(2-p)
            """
            _y = y
            if backend == "torch":
                import torch
                if _fname in ("gaussian", "squared_error"):
                    return torch.sum((_y - mu_arr) ** 2)
                elif _fname == "gamma":
                    return torch.sum(_y / mu_arr - torch.log(_y / mu_arr) - 1.0)
                elif _fname == "inverse_gaussian":
                    return torch.sum((_y - mu_arr) ** 2 / (_y * mu_arr ** 2))
                elif _fname == "negative_binomial":
                    _mu_c = torch.clamp(mu_arr, min=1e-10)
                    _y_c = torch.clamp(_y, min=1e-10)
                    _a = _nb_alpha
                    return torch.sum(
                        2.0 * (_y_c * torch.log(_y_c / _mu_c)
                               - (_y_c + 1.0 / _a) * torch.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c)))
                    )
                elif _fname == "tweedie":
                    p = _tweedie_power
                    if abs(p - 1.0) < 0.01:
                        return torch.sum(mu_arr - _y * torch.log(mu_arr))
                    elif abs(p - 2.0) < 0.01:
                        return torch.sum(_y / mu_arr - torch.log(_y / mu_arr) - 1.0)
                    else:
                        _y_pow_1mp = torch.zeros_like(_y)
                        _y_pow_2mp = torch.zeros_like(_y)
                        _mask = _y > 0.0
                        if torch.any(_mask):
                            _y_pos = _y[_mask]
                            _y_pow_1mp[_mask] = torch.pow(_y_pos, 1.0 - p)
                            _y_pow_2mp[_mask] = torch.pow(_y_pos, 2.0 - p)
                        return torch.sum(
                            _y * (_y_pow_1mp - torch.pow(mu_arr, 1.0 - p)) / (1.0 - p)
                            - (_y_pow_2mp - torch.pow(mu_arr, 2.0 - p)) / (2.0 - p)
                        )
                elif _fname in ("binomial", "logistic"):
                    _mu_c = torch.clamp(mu_arr, min=1e-10, max=1.0 - 1e-10)
                    return -2.0 * torch.sum(
                        _y * torch.log(_mu_c) + (1.0 - _y) * torch.log(1.0 - _mu_c)
                    )
                else:
                    return torch.sum(mu_arr - _y * torch.log(mu_arr))
            elif backend == "cupy":
                import cupy as cp
                if _fname in ("gaussian", "squared_error"):
                    return cp.sum((_y - mu_arr) ** 2)
                elif _fname == "gamma":
                    return cp.sum(_y / mu_arr - cp.log(_y / mu_arr) - 1.0)
                elif _fname == "inverse_gaussian":
                    return cp.sum((_y - mu_arr) ** 2 / (_y * mu_arr ** 2))
                elif _fname == "negative_binomial":
                    _mu_c = cp.clip(mu_arr, 1e-10)
                    _y_c = cp.clip(_y, 1e-10)
                    _a = _nb_alpha
                    return cp.sum(
                        2.0 * (_y_c * cp.log(_y_c / _mu_c)
                               - (_y_c + 1.0 / _a) * cp.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c)))
                    )
                elif _fname == "tweedie":
                    p = _tweedie_power
                    if abs(p - 1.0) < 0.01:
                        return cp.sum(mu_arr - _y * cp.log(mu_arr))
                    elif abs(p - 2.0) < 0.01:
                        return cp.sum(_y / mu_arr - cp.log(_y / mu_arr) - 1.0)
                    else:
                        _y_pow_1mp = cp.zeros_like(_y)
                        _y_pow_2mp = cp.zeros_like(_y)
                        _mask = _y > 0.0
                        if bool(cp.any(_mask)):
                            _y_pos = _y[_mask]
                            _y_pow_1mp[_mask] = cp.power(_y_pos, 1.0 - p)
                            _y_pow_2mp[_mask] = cp.power(_y_pos, 2.0 - p)
                        return cp.sum(
                            _y * (_y_pow_1mp - cp.power(mu_arr, 1.0 - p)) / (1.0 - p)
                            - (_y_pow_2mp - cp.power(mu_arr, 2.0 - p)) / (2.0 - p)
                        )
                elif _fname in ("binomial", "logistic"):
                    _mu_c = cp.clip(mu_arr, 1e-10, 1.0 - 1e-10)
                    return -2.0 * cp.sum(
                        _y * cp.log(_mu_c) + (1.0 - _y) * cp.log(1.0 - _mu_c)
                    )
                else:
                    return cp.sum(mu_arr - _y * cp.log(mu_arr))
            else:
                if _fname in ("gaussian", "squared_error"):
                    return float(np.sum((_y - mu_arr) ** 2))
                elif _fname == "gamma":
                    return float(np.sum(_y / mu_arr - np.log(_y / mu_arr) - 1.0))
                elif _fname == "inverse_gaussian":
                    return float(np.sum((_y - mu_arr) ** 2 / (_y * mu_arr ** 2)))
                elif _fname == "negative_binomial":
                    _mu_c = np.clip(mu_arr, 1e-10, None)
                    _y_c = np.clip(_y, 1e-10, None)
                    _a = _nb_alpha
                    return float(np.sum(
                        2.0 * (_y_c * np.log(_y_c / _mu_c)
                               - (_y_c + 1.0 / _a) * np.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c)))
                    ))
                elif _fname == "tweedie":
                    p = _tweedie_power
                    if abs(p - 1.0) < 0.01:
                        return float(np.sum(mu_arr - _y * np.log(mu_arr)))
                    elif abs(p - 2.0) < 0.01:
                        return float(np.sum(_y / mu_arr - np.log(_y / mu_arr) - 1.0))
                    else:
                        _y_pow_1mp = np.zeros_like(_y)
                        _y_pow_2mp = np.zeros_like(_y)
                        _mask = _y > 0.0
                        if np.any(_mask):
                            _y_pos = _y[_mask]
                            _y_pow_1mp[_mask] = np.power(_y_pos, 1.0 - p)
                            _y_pow_2mp[_mask] = np.power(_y_pos, 2.0 - p)
                        return float(np.sum(
                            _y * (_y_pow_1mp - np.power(mu_arr, 1.0 - p)) / (1.0 - p)
                            - (_y_pow_2mp - np.power(mu_arr, 2.0 - p)) / (2.0 - p)
                        ))
                elif _fname in ("binomial", "logistic"):
                    _mu_c = np.clip(mu_arr, 1e-10, 1.0 - 1e-10)
                    return float(-2.0 * np.sum(
                        _y * np.log(_mu_c) + (1.0 - _y) * np.log(1.0 - _mu_c)
                    ))
                else:
                    return float(np.sum(mu_arr - _y * np.log(mu_arr)))

        # Current loss — use only eta clipping (prevent exp overflow),
        # NOT mu clipping (which distorts the deviance landscape).
        eta_cur = _clip(X @ params_old, -30, 30)
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
        _is_constant_W = _fname in ("gamma", "gaussian", "squared_error")

        # Convert dev_old to Python float for tolerance computation
        # (single sync per iteration, not per line-search step).  CuPy NB
        # needs a slightly looser tolerance; the stricter 1e-10 relative
        # check over-damps late Fisher steps and causes hundreds of extra
        # iterations while converging to the same objective as CPU/Torch.
        if backend == "torch":
            dev_old_f = float(dev_old_dev.item())
        elif backend == "cupy":
            dev_old_f = float(dev_old_dev)
        else:
            dev_old_f = float(dev_old_dev)
        if backend == "cupy" and _fname == "negative_binomial":
            _dev_tol = max(abs(dev_old_f) * 1e-6, 1e-4)
        else:
            _dev_tol = max(abs(dev_old_f) * 1e-10, 1e-6)

        def _dev_accept(dev_try_dev):
            """Check if trial deviance is acceptable (device-side NaN + comparison)."""
            if backend == "torch":
                import torch
                if torch.isnan(dev_try_dev):
                    return False
                return bool((dev_try_dev <= dev_old_dev + _dev_tol).item())
            elif backend == "cupy":
                import cupy as cp
                if cp.isnan(dev_try_dev):
                    return False
                return bool(dev_try_dev <= dev_old_dev + _dev_tol)
            else:
                if dev_try_dev != dev_try_dev:
                    return False
                return dev_try_dev <= dev_old_f + _dev_tol

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
                params = params_try if _accepted else params_old + 0.1 * _direction
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
                params = params_old + 0.1 * _direction

        # Convergence: gradient norm check (most reliable for all families)
        if iteration % 5 == 4 or iteration == max_iter - 1:
            try:
                grad_f = family.gradient(X, y, params)
                if ridge_alpha > 0:
                    grad_f[1:] = grad_f[1:] + (ridge_alpha / X.shape[0]) * params[1:]
                grad_norm = float(_norm2(grad_f))
            except Exception:
                # No gradient method available — fall back to param change
                _param_change = float(_norm2(params - params_old))
                _param_norm = max(float(_norm2(params)), 1.0)
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
    ):
        """Run IRLS loop.

        Parameters
        ----------
        ridge_alpha : float
            L2 regularization (lambda = 1/(2*C) format).
        ridge_penalize_intercept : bool
            Whether to penalize the intercept.
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
        )
