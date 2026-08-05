"""
Unified IRLS solver for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch.
Single implementation works on numpy/cupy/torch backends via auto detection.
"""

import warnings
from typing import Optional

import numpy as np


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
    if backend == "auto":
        backend = _infer_backend(A)

    try:
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.solve(A, b_col)
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.solve(A, b)
        else:
            return np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.lstsq(A, b_col).solution
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.lstsq(A, b)[0]
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _clip(x, lo, hi, backend):
    if backend == "torch":
        import torch
        lo_val = lo if lo is not None else float('-inf')
        hi_val = hi if hi is not None else float('inf')
        return torch.clamp(x, min=lo_val, max=hi_val)
    if backend == "cupy":
        import cupy as cp
        return cp.clip(x, lo, hi)
    return np.clip(x, lo, hi)


def _norm(x, backend):
    if backend == "torch":
        import torch

        return float(torch.linalg.norm(x).item())
    return float(np.linalg.norm(x))


def _zeros(n, backend, ref_tensor=None, dtype=np.float64):
    if backend == "cupy":
        import cupy as cp
        return cp.zeros(n, dtype=cp.float64)
    if backend == "torch":
        import torch
        device = ref_tensor.device if ref_tensor is not None else "cpu"
        return torch.zeros(n, dtype=torch.float64, device=device)
    return np.zeros(n, dtype=dtype)


def _diag(reg, backend, ref_tensor=None):
    """Create diagonal matrix from 1D array."""
    if backend == "cupy":
        import cupy as cp
        return cp.diag(cp.asarray(reg, dtype=cp.float64))
    if backend == "torch":
        import torch
        return torch.diag(
            torch.tensor(reg, dtype=torch.float64, device=ref_tensor.device if ref_tensor is not None else "cpu")
        )
    return np.diag(reg)


def _to_backend(arr, backend, ref_tensor):
    """Convert numpy array to the target backend."""
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(arr, dtype=cp.float64)
    if backend == "torch":
        import torch
        return torch.tensor(arr, dtype=torch.float64, device=ref_tensor.device if ref_tensor is not None else "cpu")
    return np.asarray(arr, dtype=float)


def _copy_arr(arr):
    """Copy array: .clone() for torch, .copy() for numpy/cupy."""
    if hasattr(arr, 'clone'):
        return arr.clone()
    return arr.copy()


# =============================================================================
# Torch.compile for IRLS elementwise chain fusion
# =============================================================================
# When backend is torch on CUDA, the per-iteration elementwise ops
# (link inverse, weight computation, working response, weighted matmul)
# can be fused via torch.compile to reduce kernel launch overhead.

_IRLS_STEP_COMPILED = None


from statgpu.backends._torch_compile import compile_torch


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

    _IRLS_STEP_COMPILED = compile_torch(
        _irls_weighted_gemm,
        workload="iterative",
        dynamic=True,
        fullgraph=False,
    )
    return _IRLS_STEP_COMPILED


def _irls_step_call(compiled_fn, *args):
    """Call the centrally managed compiled IRLS step."""
    return compiled_fn(*args)


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

    if init_coef is None:
        n_features = X.shape[1]
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = init_coef

    y_work = _to_backend(y, backend, X)
    family_name = getattr(family, "name", "")
    if backend == "torch":
        import torch
        invalid_y = torch.any(~torch.isfinite(y_work))
        if family_name == "gamma":
            invalid_y = invalid_y | torch.any(y_work <= 0)
        elif family_name == "tweedie":
            invalid_y = invalid_y | torch.any(y_work < 0)
    elif backend == "cupy":
        import cupy as cp
        invalid_y = cp.any(~cp.isfinite(y_work))
        if family_name == "gamma":
            invalid_y = invalid_y | cp.any(y_work <= 0)
        elif family_name == "tweedie":
            invalid_y = invalid_y | cp.any(y_work < 0)
    else:
        invalid_y = np.any(~np.isfinite(y_work))
        if family_name == "gamma":
            invalid_y = invalid_y or np.any(y_work <= 0)
        elif family_name == "tweedie":
            invalid_y = invalid_y or np.any(y_work < 0)
    if bool(invalid_y.item() if hasattr(invalid_y, "item") else invalid_y):
        requirement = "strictly positive" if family_name == "gamma" else "non-negative"
        raise ValueError(
            f"{family_name} IRLS requires finite, {requirement} y values."
        )
    sw_work = (
        _to_backend(sample_weight, backend, X)
        if sample_weight is not None else None
    )
    penalty_matrix_work = (
        _to_backend(penalty_matrix, backend, X)
        if penalty_matrix is not None else None
    )
    line_search_failed = False
    iteration = 0
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
            eta = _clip(eta_raw, -30, 30, backend)

        # Step 2: inverse link -> mean (clip mu to prevent extreme weights)
        # For identity link (squared_error), skip clipping — mu = eta.
        mu = family.link.inverse(eta)
        if _link_name not in ('identity', 'Identity'):
            mu = _clip(mu, 1e-10, 1e6, backend)

        # Step 3: IRLS weights
        W = family.irls_weights(mu, y_work)
        W = _clip(W, 1e-10, None, backend)

        if sw_work is not None:
            W = W * sw_work

        # Step 4: working response
        z = family.irls_working_response(mu, y_work, eta)

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
        if penalty_matrix_work is not None:
            XtWX = XtWX + penalty_matrix_work

        params_new = _solve(XtWX, Xtz, backend)

        # Armijo backtracking line search: find step in (0, 1] that
        # gives sufficient decrease in the loss (deviance).
        _fname = family_name
        _tweedie_power = float(getattr(family, 'power', 1.5)) if _fname == "tweedie" else 0.0
        _nb_alpha = float(getattr(family, 'alpha', 1.0)) if _fname == "negative_binomial" else 0.0

        def _dev_val(mu_arr):
            """Return weighted family deviance on the active backend."""
            _y = y_work
            if backend == "torch":
                import torch as xp
            elif backend == "cupy":
                import cupy as xp
            else:
                xp = np

            if _fname in ("gaussian", "squared_error"):
                terms = 0.5 * (_y - mu_arr) ** 2
            elif _fname in ("binomial", "logistic"):
                _mu_c = _clip(mu_arr, 1e-10, 1.0 - 1e-10, backend)
                terms = -_y * xp.log(_mu_c) - (1.0 - _y) * xp.log1p(-_mu_c)
            elif _fname == "gamma":
                terms = _y / mu_arr + xp.log(mu_arr)
            elif _fname == "inverse_gaussian":
                terms = _y / (2.0 * mu_arr ** 2) - 1.0 / mu_arr
            elif _fname == "negative_binomial":
                _mu_c = _clip(mu_arr, 1e-10, None, backend)
                _y_c = _clip(_y, 1e-10, None, backend)
                _a = _nb_alpha
                terms = (
                    _y_c * xp.log(_y_c / _mu_c)
                    - (_y_c + 1.0 / _a)
                    * xp.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c))
                )
            elif _fname == "tweedie":
                p = _tweedie_power
                if abs(p - 1.0) < 0.01:
                    terms = mu_arr - _y * xp.log(mu_arr)
                elif abs(p - 2.0) < 0.01:
                    terms = _y / mu_arr - xp.log(_y / mu_arr) - 1.0
                else:
                    terms = (
                        -_y * xp.pow(mu_arr, 1.0 - p) / (1.0 - p)
                        + xp.pow(mu_arr, 2.0 - p) / (2.0 - p)
                    )
            else:
                terms = mu_arr - _y * xp.log(mu_arr)

            if sw_work is not None:
                terms = terms * sw_work
            return xp.sum(terms)

        def _penalty_val(params_arr):
            value = 0.0
            if ridge_alpha > 0:
                penalized = (
                    params_arr if ridge_penalize_intercept else params_arr[1:]
                )
                if backend == "torch":
                    import torch
                    value = value + 0.5 * ridge_alpha * torch.sum(penalized ** 2)
                elif backend == "cupy":
                    import cupy as cp
                    value = value + 0.5 * ridge_alpha * cp.sum(penalized ** 2)
                else:
                    value = value + 0.5 * ridge_alpha * np.sum(penalized ** 2)
            if penalty_matrix_work is not None:
                value = value + 0.5 * (
                    params_arr @ penalty_matrix_work @ params_arr
                )
            return value

        def _objective_val(mu_arr, params_arr):
            return _dev_val(mu_arr) + _penalty_val(params_arr)

        # Current loss — use only eta clipping (prevent exp overflow),
        # NOT mu clipping (which distorts the deviance landscape).
        eta_cur = _clip(X @ params_old, -30, 30, backend)
        mu_cur = family.link.inverse(eta_cur)
        try:
            dev_old_dev = _objective_val(mu_cur, params_old)
        except Exception:
            dev_old_dev = float('inf')

        # Gaussian-identity and Gamma-log have constant Fisher weights.
        # Try their full Fisher-scoring step before backtracking.
        _direction = params_new - params_old
        _is_constant_W = (
            _fname in ("gaussian", "squared_error")
            or (_fname == "gamma" and _link_name == "log")
        )

        # Convert dev_old to Python float for tolerance computation
        # (single sync per iteration, not per line-search step)
        if backend == "torch":
            dev_old_f = float(dev_old_dev.item())
        elif backend == "cupy":
            dev_old_f = float(dev_old_dev)
        else:
            dev_old_f = float(dev_old_dev)
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
            eta_new = _clip(X @ params_new, -30, 30, backend)
            mu_new = family.link.inverse(eta_new)
            try:
                dev_new_dev = _objective_val(mu_new, params_new)
            except Exception:
                dev_new_dev = float('inf')
            if _dev_accept(dev_new_dev):
                params = params_new
            else:
                step = 1.0
                _accepted = False
                for _bt in range(30):
                    params_try = params_old + step * _direction
                    eta_try = _clip(X @ params_try, -30, 30, backend)
                    mu_try = family.link.inverse(eta_try)
                    try:
                        dev_try_dev = _objective_val(mu_try, params_try)
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
                    line_search_failed = True
                    break
        else:
            # Variable weights: Armijo backtracking on deviance
            step = 1.0
            _accepted = False
            for _bt in range(30):
                params_try = params_old + step * _direction
                eta_try = _clip(X @ params_try, -30, 30, backend)
                mu_try = family.link.inverse(eta_try)
                try:
                    dev_try_dev = _objective_val(mu_try, params_try)
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
                line_search_failed = True
                break

        # Convergence: normalized penalized score norm. Parameter changes can
        # be tiny merely because line search truncated a bad step.
        if iteration % 5 == 4 or iteration == max_iter - 1:
            eta_check = X @ params
            if _link_name not in ("identity", "Identity"):
                eta_check = _clip(eta_check, -30, 30, backend)
            mu_check = family.link.inverse(eta_check)
            if _link_name not in ("identity", "Identity"):
                mu_check = _clip(mu_check, 1e-10, 1e6, backend)
            score_eta = (
                (mu_check - y_work)
                / (family.variance(mu_check) * family.link.derivative(mu_check))
            )
            if sw_work is not None:
                score_eta = score_eta * sw_work
                sw_sum = sw_work.sum()
                n_eff = float(sw_sum.item() if hasattr(sw_sum, "item") else sw_sum)
            else:
                n_eff = float(X.shape[0])
            grad_f = X.T @ score_eta / n_eff
            if ridge_alpha > 0:
                ridge_grad = (ridge_alpha / n_eff) * params
                if not ridge_penalize_intercept:
                    ridge_grad = _copy_arr(ridge_grad)
                    ridge_grad[0] = 0.0
                grad_f = grad_f + ridge_grad
            if penalty_matrix_work is not None:
                grad_f = grad_f + (penalty_matrix_work @ params) / n_eff
            grad_norm = float(_norm(grad_f, backend))
            if grad_norm < tol:
                break

    n_iter = iteration + 1
    from statgpu.solvers._convergence import ConvergenceWarning
    if line_search_failed:
        warnings.warn(
            f"irls line search failed to find a decreasing step "
            f"(family={getattr(family, 'name', '?')}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    elif n_iter >= max_iter:
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
