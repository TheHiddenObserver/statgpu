"""
Unified IRLS solver for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch.
Single implementation works on numpy/cupy/torch backends via auto detection.
"""

import warnings
from numbers import Integral, Real
from typing import Optional

import numpy as np

from statgpu.backends._array_ops import _linalg_exception_is_rank_failure


def _infer_backend(X):
    """Detect backend from array type."""
    mod = type(X).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _solve(A, b, backend="auto"):
    """Solve a linear system, using least squares only for singular systems."""
    if backend == "auto":
        backend = _infer_backend(A)

    if backend == "torch":
        import torch

        b_col = b.unsqueeze(1) if b.ndim == 1 else b
        try:
            sol = torch.linalg.solve(A, b_col)
        except RuntimeError as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            sol = torch.linalg.lstsq(A, b_col).solution
        return sol.squeeze(1) if b.ndim == 1 else sol

    if backend == "cupy":
        import cupy as cp

        try:
            return cp.linalg.solve(A, b)
        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            return cp.linalg.lstsq(A, b)[0]

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
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
    if backend == "cupy":
        import cupy as cp

        return float(cp.linalg.norm(x).item())
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

        device = ref_tensor.device if ref_tensor is not None else "cpu"
        if torch.is_tensor(arr):
            return arr.to(dtype=torch.float64, device=device)
        return torch.as_tensor(arr, dtype=torch.float64, device=device)
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


class _BinomialFamilyObjective:
    """Bernoulli negative log-likelihood for an arbitrary Binomial link."""

    def __init__(self, family):
        from statgpu.glm_core._logistic import LogisticLoss

        self.family = family
        self._validator = LogisticLoss()

    def validate_response(self, y):
        return self._validator.validate_response(y)

    def per_sample_value(self, eta, y):
        from statgpu.backends._array_ops import _clip as _array_clip, _log

        mu = _array_clip(self.family.link.inverse(eta), 1e-10, 1 - 1e-10)
        return -y * _log(mu) - (1 - y) * _log(1 - mu)


def _objective_loss_for_family(family):
    """Return the objective matching the exact IRLS family and link."""
    from statgpu.glm_core._base import get_glm_loss

    family_name = str(getattr(family, "name", "")).lower()
    if family_name in {"binomial", "logistic"}:
        return _BinomialFamilyObjective(family)
    loss_names = {
        "gaussian": "squared_error",
        "squared_error": "squared_error",
        "poisson": "poisson",
        "gamma": "gamma",
        "inverse_gaussian": "inverse_gaussian",
        "negative_binomial": "negative_binomial",
        "tweedie": "tweedie",
    }
    if family_name not in loss_names:
        raise NotImplementedError(
            "IRLS line search requires a registered objective for family "
            f"{family_name!r}."
        )
    kwargs = {}
    if family_name == "gamma":
        kwargs["link"] = str(getattr(family.link, "name", "log")).lower()
    elif family_name == "negative_binomial":
        kwargs["alpha"] = float(getattr(family, "alpha", 1.0))
    elif family_name == "tweedie":
        kwargs["power"] = float(getattr(family, "power", 1.5))
    return get_glm_loss(loss_names[family_name], **kwargs)


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
    from statgpu.glm_core._validation import (
        validate_glm_design_matrix,
        validate_glm_sample_weight,
    )

    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(tol, bool) or not isinstance(tol, Real):
        raise ValueError("tol must be a finite positive real number")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a finite positive real number")
    if isinstance(ridge_alpha, bool) or not isinstance(ridge_alpha, Real):
        raise ValueError("ridge_alpha must be a finite non-negative real number")
    ridge_alpha = float(ridge_alpha)
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be a finite non-negative real number")
    if not isinstance(ridge_penalize_intercept, (bool, np.bool_)):
        raise ValueError("ridge_penalize_intercept must be boolean")
    max_iter = int(max_iter)

    X_validated = validate_glm_design_matrix(X)
    if backend == "auto":
        backend = _infer_backend(X_validated)
    backend = str(backend).lower()
    backend = {"cpu": "numpy", "cuda": "cupy"}.get(backend, backend)
    if backend not in {"numpy", "cupy", "torch"}:
        raise ValueError("backend must be one of: 'auto', 'numpy', 'cupy', 'torch'")
    X = _to_backend(X_validated, backend, X_validated)

    n_features = int(X.shape[1])
    if init_coef is None:
        params = _zeros(n_features, backend, ref_tensor=X)
    else:
        params = _to_backend(init_coef, backend, X).reshape(-1)
        if int(params.shape[0]) != n_features:
            raise ValueError("init_coef must have length X.shape[1].")
        params = _copy_arr(params)

    family_name = getattr(family, "name", "")
    objective_loss = _objective_loss_for_family(family)
    y_validated = objective_loss.validate_response(y)
    y_work = _to_backend(y_validated, backend, X)
    if int(y_work.shape[0]) != int(X.shape[0]):
        raise ValueError("Response length must match X.shape[0].")
    sw_validated = (
        validate_glm_sample_weight(sample_weight, X.shape[0])
        if sample_weight is not None else None
    )
    sw_work = (
        _to_backend(sw_validated, backend, X)
        if sw_validated is not None else None
    )
    penalty_matrix_validated = (
        validate_glm_design_matrix(penalty_matrix, name="penalty_matrix")
        if penalty_matrix is not None else None
    )
    if penalty_matrix_validated is not None and tuple(
        penalty_matrix_validated.shape
    ) != (n_features, n_features):
        raise ValueError(
            "penalty_matrix must have shape (X.shape[1], X.shape[1])"
        )
    penalty_matrix_work = (
        _to_backend(penalty_matrix_validated, backend, X)
        if penalty_matrix_validated is not None else None
    )
    if penalty_matrix_work is not None:
        if backend == "torch":
            import torch

            symmetric = bool(torch.allclose(
                penalty_matrix_work, penalty_matrix_work.T, rtol=1e-10, atol=1e-12
            ))
            min_eig = float(torch.linalg.eigvalsh(penalty_matrix_work).min().item())
            scale = max(1.0, float(torch.max(torch.abs(penalty_matrix_work)).item()))
        elif backend == "cupy":
            import cupy as cp

            symmetric = bool(cp.allclose(
                penalty_matrix_work, penalty_matrix_work.T, rtol=1e-10, atol=1e-12
            ).item())
            min_eig = float(cp.linalg.eigvalsh(penalty_matrix_work).min().item())
            scale = max(1.0, float(cp.max(cp.abs(penalty_matrix_work)).item()))
        else:
            symmetric = bool(np.allclose(
                penalty_matrix_work, penalty_matrix_work.T, rtol=1e-10, atol=1e-12
            ))
            min_eig = float(np.linalg.eigvalsh(penalty_matrix_work).min())
            scale = max(1.0, float(np.max(np.abs(penalty_matrix_work))))
        if not symmetric:
            raise ValueError("penalty_matrix must be symmetric")
        if min_eig < -1e-10 * scale:
            raise ValueError("penalty_matrix must be positive semidefinite")
    line_search_failed = False
    converged = False
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

        # Backtracking line search on the same registered loss used by the
        # public GLM objective.  Loss classes own link/domain clipping, so the
        # identity-link Gaussian path is never spuriously clipped to [-30, 30].
        def _loss_val(eta_arr):
            terms = objective_loss.per_sample_value(eta_arr, y_work)
            if sw_work is not None:
                terms = terms * sw_work
            if backend == "torch":
                import torch

                return torch.sum(terms)
            if backend == "cupy":
                import cupy as cp

                return cp.sum(terms)
            return np.sum(terms)

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

        def _objective_val(eta_arr, params_arr):
            return _loss_val(eta_arr) + _penalty_val(params_arr)

        def _scalar_float(value):
            return float(value.item() if hasattr(value, "item") else value)

        def _scalar_is_finite(value):
            if backend == "torch":
                import torch

                return bool(torch.isfinite(value).item())
            if backend == "cupy":
                import cupy as cp

                return bool(cp.isfinite(value).item())
            return bool(np.isfinite(value))

        eta_cur = X @ params_old
        objective_old = _objective_val(eta_cur, params_old)
        if not _scalar_is_finite(objective_old):
            raise FloatingPointError(
                "IRLS objective became non-finite at the current iterate."
            )
        objective_old_float = _scalar_float(objective_old)
        objective_tolerance = max(
            abs(objective_old_float) * 1e-10,
            1e-6,
        )

        def _objective_accept(objective_try):
            if not _scalar_is_finite(objective_try):
                return False
            return _scalar_float(objective_try) <= (
                objective_old_float + objective_tolerance
            )

        direction = params_new - params_old
        is_constant_weight = (
            family_name in ("gaussian", "squared_error")
            or (
                family_name == "gamma"
                and str(getattr(family.link, "name", "")).lower() == "log"
            )
        )

        if is_constant_weight:
            objective_new = _objective_val(X @ params_new, params_new)
            if _objective_accept(objective_new):
                params = params_new
            else:
                step = 1.0
                accepted = False
                for _ in range(30):
                    params_try = params_old + step * direction
                    objective_try = _objective_val(X @ params_try, params_try)
                    if _objective_accept(objective_try):
                        accepted = True
                        break
                    step *= 0.5
                if accepted:
                    params = params_try
                else:
                    params = params_old
                    line_search_failed = True
                    break
        else:
            step = 1.0
            accepted = False
            for _ in range(30):
                params_try = params_old + step * direction
                objective_try = _objective_val(X @ params_try, params_try)
                if _objective_accept(objective_try):
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                params = params_try
            else:
                params = params_old
                line_search_failed = True
                break

        # Convergence: normalized penalized score norm. Parameter changes can
        # be tiny merely because line search truncated a bad step.
        if is_constant_weight or iteration % 5 == 4 or iteration == max_iter - 1:
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
                converged = True
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
    elif not converged:
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
