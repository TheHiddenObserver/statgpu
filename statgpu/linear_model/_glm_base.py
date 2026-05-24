"""
Generalized Linear Model base classes.

Uses sklearn pattern: base class with subclasses overriding _get_family()
and, when needed, the family-to-GLM-loss mapping.
Supports IRLS (smooth penalty) and FISTA (any penalty) solvers.
"""

from typing import Optional, Union, Dict
import numpy as np


def _parse_formula_if_provided(formula, data, X, y):
    """Parse formula+data or fall back to raw arrays. Returns (y, X, info)."""
    if formula is not None:
        from statgpu.core.formula import parse_formula
        return parse_formula(formula, data)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy
from statgpu.glm_core._irls import IRLSSolver
from statgpu.glm_core._solver import fista_solver
from statgpu.glm_core._family import (
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    NegativeBinomial,
    Tweedie,
)


def _xp_arr(arr):
    """Get the array module (numpy/cupy) from array type.

    Unlike glm_core._family._xp, this does NOT return torch — it returns
    numpy for torch tensors since we need numpy-compatible indexing ops.
    """
    mod = type(arr).__module__
    if mod.startswith('cupy'):
        import cupy
        return cupy
    return np


class GeneralizedLinearModel(BaseEstimator):
    """GLM base class with shared IRLS + FISTA paths.

    Subclasses override _get_family() and optionally the GLM loss mapping.

    Parameters
    ----------
    family : str, default='gaussian'
        Distribution family: 'gaussian', 'binomial', 'poisson'.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    C : float, default=1.0
        Inverse regularization strength (for IRLS L2).
    device : str or Device, default='auto'
    solver : str, default='auto'
        'auto', 'irls', 'fista', 'newton', or 'lbfgs'.
    """

    def __init__(
        self,
        family: str = "gaussian",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.family = family
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.solver = solver.lower()
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None

    def _get_family(self):
        """Return the GLM Family instance. Override in subclass."""
        family_map = {
            "gaussian": Gaussian,
            "binomial": Binomial,
            "poisson": Poisson,
            "gamma": Gamma,
            "inverse_gaussian": InverseGaussian,
            "negative_binomial": NegativeBinomial,
            "tweedie": Tweedie,
        }
        kwargs = self._get_loss_kwargs()
        return family_map[self.family](**kwargs)

    def _get_penalty_alpha(self):
        """L2 regularization alpha for IRLS: lambda = 1/(2*C)."""
        return 1.0 / (2.0 * self.C) if self.C > 0 else 0.0

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit GLM model.

        Parameters
        ----------
        X : array-like or None
            Predictor matrix. Required if ``formula`` is None.
        y : array-like or None
            Response vector. Required if ``formula`` is None.
        sample_weight : array-like or None
            Sample weights.
        formula : str or None
            R-style formula string (e.g. ``"y ~ x1 + x2"``).
        data : pd.DataFrame or None
            DataFrame used with ``formula`` for column lookup.
        """
        # Handle formula interface
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            y_arr, X_arr, design_info = _parse_formula_if_provided(
                formula, data, None, None
            )
            self._design_info = design_info
            formula_column_names = list(design_info.column_names)
            self._formula_has_intercept = "Intercept" in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != "Intercept"]
            if self._formula_has_intercept:
                intercept_idx = formula_column_names.index("Intercept")
                X_arr = np.delete(X_arr, intercept_idx, axis=1)
                self.fit_intercept = True
            else:
                # Formula syntax owns intercept semantics, matching statsmodels/R.
                self.fit_intercept = False
        else:
            if X is None or y is None:
                raise ValueError(
                    "Either formula+data or X+y must be provided."
                )
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            X_arr = np.asarray(X)

        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        X_arr = self._to_array(X_arr, backend=backend_name)
        y_arr = self._to_array(y_arr, backend=backend_name)
        self._nobs = X_arr.shape[0]

        family = self._get_family()
        solver_name = self.solver if self.solver != "auto" else "irls"

        if solver_name == "irls":
            self._fit_irls(X_arr, y_arr, sample_weight, family, backend_name)
        elif solver_name == "fista":
            self._fit_fista(X_arr, y_arr, sample_weight, family, backend_name)
        elif solver_name in ("newton", "lbfgs"):
            self._fit_smooth_solver(
                X_arr, y_arr, sample_weight, solver_name, backend_name
            )
        else:
            raise ValueError(
                "solver must be one of: 'auto', 'irls', 'fista', 'newton', 'lbfgs'"
            )

        self._fitted = True
        return self

    def _fit_irls(self, X, y, sample_weight, family, backend_name="numpy"):
        """Fit using IRLS (per-iteration weighted least squares)."""
        # IRLSSolver solves the unnormalized WLS normal equations
        # X'WX + lambda I, while _get_penalty_alpha() is the normalized
        # objective penalty.  Scale by n to keep C semantics consistent.
        ridge_alpha = X.shape[0] * self._get_penalty_alpha()

        if self.fit_intercept:
            if backend_name == "cupy":
                import cupy as cp
                X_design = cp.column_stack([cp.ones(X.shape[0]), X])
            elif backend_name == "torch":
                import torch
                X_design = torch.column_stack([torch.ones(X.shape[0], dtype=torch.float64, device=X.device), X])
            else:
                X_design = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_design = X

        solver = IRLSSolver(family, max_iter=self.max_iter, tol=self.tol)
        params, n_iter = solver.fit(
            X_design, y,
            sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=not self.fit_intercept,
            backend=backend_name,
        )

        self.n_iter_ = n_iter
        self._params = params

        # Convert to numpy (params may be cupy/torch array)
        params_np = _to_numpy(params)

        if self.fit_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))

    def _fit_fista(self, X, y, sample_weight, family, backend_name="numpy"):
        """Fit using FISTA (no penalty; pure loss minimization).

        For GLM losses with intercept, uses iterated intercept estimation
        + coef refinement to converge to the correct joint optimum.
        """
        from statgpu.glm_core import get_glm_loss
        from statgpu.penalties._l2 import L2Penalty

        loss_kwargs = self._get_loss_kwargs()
        loss = get_glm_loss(self.family_to_loss(), **loss_kwargs)

        if not self.fit_intercept:
            X_centered = X
            init = None
            if self.family == "gamma" and loss_kwargs.get("link") == "inverse_power":
                if backend_name == "cupy":
                    import cupy as cp
                    y_cp = cp.asarray(y, dtype=cp.float64)
                    X_cp = cp.asarray(X_centered, dtype=cp.float64)
                    eta_target = 1.0 / cp.clip(y_cp, 1e-6, None)
                    try:
                        init_cp, *_ = cp.linalg.lstsq(X_cp, eta_target, rcond=None)
                    except cp.linalg.LinAlgError:
                        init_cp = cp.zeros(X.shape[1], dtype=cp.float64)
                    init = init_cp.astype(X.dtype if hasattr(X, "dtype") else cp.float64, copy=False)
                elif backend_name == "torch":
                    import torch
                    dtype = X.dtype if hasattr(X, "dtype") else torch.float64
                    y_t = y.to(X.device).to(torch.float64)
                    X_t = X_centered.to(X.device).to(torch.float64)
                    eta_target = 1.0 / torch.clamp(y_t, min=1e-6)
                    try:
                        init_t = torch.linalg.lstsq(X_t, eta_target).solution
                    except RuntimeError:
                        init_t = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
                    init = init_t.to(dtype)
                else:
                    y_np = np.asarray(y, dtype=np.float64)
                    X_np = np.asarray(X_centered, dtype=np.float64)
                    eta_target = 1.0 / np.clip(y_np, 1e-6, None)
                    try:
                        init = np.linalg.lstsq(X_np, eta_target, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        init = np.zeros(X.shape[1], dtype=np.float64)
            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
            self.coef_ = _to_numpy(coef)
            self.n_iter_ = n_iter
            self.intercept_ = 0.0
            self._df_resid = self._nobs - X.shape[1]
            return

        if loss.name != "squared_error":
            # All non-Gaussian GLM losses must optimize intercept jointly with
            # coefficients. Centering y is only valid for squared-error loss.
            # Augment X with intercept column (no penalty in _fit_fista).
            if backend_name == "cupy":
                import cupy as cp
                X_aug = cp.column_stack([X, cp.ones(X.shape[0])])
            elif backend_name == "torch":
                import torch
                X_aug = torch.column_stack([X, torch.ones(X.shape[0], dtype=torch.float64, device=X.device)])
            else:
                X_aug = np.column_stack([X, np.ones(X.shape[0])])
            p = X.shape[1]
            y_mean = max(float(np.mean(_to_numpy(y))), 1e-3)
            init = np.zeros(p + 1, dtype=np.float64)
            if self.family == "binomial":
                p_mean = np.clip(y_mean, 1e-3, 1.0 - 1e-3)
                init[-1] = np.log(p_mean / (1.0 - p_mean))
            elif self.family == "gamma" and loss_kwargs.get("link") == "inverse_power":
                init[-1] = 1.0 / y_mean
            elif self.family in (
                "poisson", "gamma", "inverse_gaussian",
                "negative_binomial", "tweedie",
            ):
                init[-1] = np.log(y_mean)
            if backend_name == "cupy":
                init = cp.asarray(init, dtype=X.dtype if hasattr(X, "dtype") else cp.float64)
            elif backend_name == "torch":
                init = torch.from_numpy(init).to(X.device).to(X.dtype if hasattr(X, "dtype") else torch.float64)

            full_coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_aug, y,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )

            full_np = _to_numpy(full_coef)
            self.coef_ = full_np[:p]
            self.intercept_ = float(full_np[p])
            self.n_iter_ = n_iter
        else:
            # Squared error: centering X and y preserves the objective.
            if backend_name == "cupy":
                import cupy as cp
                X_centered = X - cp.mean(X, axis=0)
                y_centered = y - cp.mean(y)
            elif backend_name == "torch":
                import torch
                X_centered = X - torch.mean(X, dim=0)
                y_centered = y - torch.mean(y)
            else:
                X_centered = X - X.mean(axis=0)
                y_centered = y - y.mean()

            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y_centered,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            X_mean = np.mean(_to_numpy(X), axis=0)
            y_mean = float(np.mean(_to_numpy(y)))
            self.coef_ = _to_numpy(coef)
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self.n_iter_ = n_iter

        self._df_resid = self._nobs - (X.shape[1] + 1)

    def _fit_smooth_solver(self, X, y, sample_weight, solver_name, backend_name):
        """Fit ordinary GLM with backend-native Newton or L-BFGS."""
        from statgpu.glm_core import get_glm_loss
        from statgpu.glm_core._solver import lbfgs_solver, newton_solver

        if sample_weight is not None:
            raise ValueError(
                f"solver='{solver_name}' does not support sample_weight yet; "
                "use solver='irls' or solver='fista'."
            )

        loss_kwargs = self._get_loss_kwargs()
        loss = get_glm_loss(self.family_to_loss(), **loss_kwargs)
        if not getattr(loss, "has_hessian", False):
            raise ValueError(f"solver='{solver_name}' requires a Hessian.")

        if self.fit_intercept:
            if backend_name == "cupy":
                import cupy as cp
                X_work = cp.column_stack([X, cp.ones(X.shape[0], dtype=X.dtype)])
            elif backend_name == "torch":
                import torch
                X_work = torch.column_stack([
                    X,
                    torch.ones(X.shape[0], dtype=X.dtype, device=X.device),
                ])
            else:
                X_work = np.column_stack([X, np.ones(X.shape[0], dtype=X.dtype)])
            p = X.shape[1]
        else:
            X_work = X
            p = X.shape[1]

        if solver_name == "newton":
            params, n_iter = newton_solver(
                loss, None, X_work, y, max_iter=self.max_iter, tol=self.tol
            )
        else:
            params, n_iter = lbfgs_solver(
                loss, None, X_work, y, max_iter=self.max_iter, tol=self.tol
            )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
        self._params = (
            np.concatenate([[self.intercept_], self.coef_])
            if self.fit_intercept
            else self.coef_.copy()
        )
        self._df_resid = self._nobs - (
            X.shape[1] + (1 if self.fit_intercept else 0)
        )

    def _get_loss_kwargs(self):
        """Override in subclass to pass extra kwargs to family/loss."""
        return {}

    def family_to_loss(self):
        """Map family name to loss name."""
        mapping = {
            "gaussian": "squared_error",
            "binomial": "logistic",
            "poisson": "poisson",
            "gamma": "gamma",
            "inverse_gaussian": "inverse_gaussian",
            "negative_binomial": "negative_binomial",
            "tweedie": "tweedie",
        }
        return mapping.get(self.family, "squared_error")

    def predict(self, X):
        """Predict using fitted model."""
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self._design_info is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None and isinstance(X, pd.DataFrame):
                from statgpu.core.formula import FormulaParser

                parser = FormulaParser.__new__(FormulaParser)
                parser._design_info = self._design_info
                parser.formula = None
                X = parser.transform(X)
                col_names = list(self._design_info.column_names)
                if self._formula_has_intercept and "Intercept" in col_names:
                    X = np.delete(X, col_names.index("Intercept"), axis=1)

        device = self._get_compute_device()
        family = self._get_family()
        if device == Device.CUDA:
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self.fit_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            return family.link.inverse(raw)
        if device == Device.TORCH:
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self.fit_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            return family.link.inverse(raw)

        X = np.asarray(X)
        raw = X @ self.coef_
        if self.fit_intercept:
            raw += self.intercept_

        return family.link.inverse(raw)


class OrderedGeneralizedLinearModel(GeneralizedLinearModel):
    """Ordered GLM base class.

    Jointly estimates coefficients + (K-1) thresholds.
    P(y <= j | X) = F(theta_j - X * beta)

    Parameters
    ----------
    n_categories : int, default=3
        Number of ordinal categories.
    family : str, default='binomial'
        Distribution family (should be Binomial for ordered models).
    ... : same as GeneralizedLinearModel
    """

    def __init__(
        self,
        n_categories: int = 3,
        family: str = "binomial",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(
            family=family,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )
        self.n_categories = n_categories
        self.thresholds_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit ordered GLM using L-BFGS.

        Supports numpy (CPU via scipy), cupy (GPU via native L-BFGS),
        and torch (GPU via torch.optim.LBFGS).
        """
        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        self._selected_backend_name = backend_name
        self._nobs = X.shape[0]

        # Convert to backend format (cupy→cupy zero-copy, numpy→cupy/torch)
        X = self._to_array(X, backend=backend_name)
        y = self._to_array(y, backend=backend_name)

        family = self._get_family()
        K = self.n_categories
        n = X.shape[0]
        p = X.shape[1]

        if backend_name == "cupy":
            self._fit_cupy_ordered(X, y, family, K, n, p)
        elif backend_name == "torch":
            self._fit_torch_ordered(X, y, family, K, n, p)
        else:
            self._fit_scipy_ordered(X, y, family, K, n, p)

        self._df_resid = self._nobs - (p + K - 1)
        self._fitted = True
        return self

    def _fit_scipy_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using scipy.optimize.minimize(L-BFGS-B)."""
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        Xs = (X - X_mean) / X_std

        theta_init = np.zeros(p + K - 1)
        theta_init[p:] = np.arange(0.5, K - 0.5, dtype=np.float64)

        cache = {"nll": None, "grad": None, "theta": None}

        def nll_and_grad(theta):
            if cache["theta"] is not None and np.array_equal(cache["theta"], theta):
                return cache["nll"], cache["grad"]

            beta = theta[:p]
            thresh = theta[p:]

            prob = self._ordered_category_probs(Xs, beta, thresh, family, K)
            prob_c = np.clip(prob, 1e-15, None)
            nll = -np.sum(np.log(prob_c[y, np.arange(n)])) / n

            grad = self._ordered_gradient(
                Xs, y, beta, thresh, prob, prob_c, family, K, n
            )

            cache["nll"] = nll
            cache["grad"] = grad
            cache["theta"] = theta
            return nll, grad

        def nll_func(theta):
            val, _ = nll_and_grad(theta)
            return val

        def grad_func(theta):
            _, g = nll_and_grad(theta)
            return g

        result = minimize(
            nll_func, theta_init, jac=grad_func, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol * 1e-3,
                     "gtol": self.tol, "disp": False},
        )

        theta = result.x
        beta_scaled = theta[:p]
        self.coef_ = beta_scaled / X_std
        thresh_est = np.sort(theta[p:])
        self.thresholds_ = np.concatenate([[-np.inf], thresh_est, [np.inf]])
        self._X_mean = X_mean
        self._X_std = X_std
        self.n_iter_ = result.nit if hasattr(result, "nit") else result.nfev

    def _fit_cupy_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using full CuPy L-BFGS on GPU.

        All computation stays on GPU — no scipy bridge, no CPU round-trips.
        Pre-allocates arrays (prob, prob_c, eta, diff, deriv_all) to amortize
        GPU memory allocation overhead across iterations.

        Warm start: reuses previous fit's solution if available (self.coef_ exists).

        For n=5000: GPU NLL+grad ≈ 2.4ms/call, ~67 evals ≈ 160ms total.
        For n=50000+: GPU compute dominates, kernel launch overhead is negligible.
        """
        import cupy as cp

        X = cp.asarray(X, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.int64)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-10] = 1.0
        Xs = (X - X_mean) / X_std

        # Pre-allocate reusable arrays (amortize GPU alloc overhead)
        _prob_pre = cp.zeros((K, n), dtype=cp.float64)
        _prob_c_pre = cp.zeros((K, n), dtype=cp.float64)
        _eta_pre = cp.zeros(n, dtype=cp.float64)
        _diff_pre = cp.zeros((K - 1, n), dtype=cp.float64)
        _deriv_all = cp.zeros((K - 1, n), dtype=cp.float64)
        _scalar = cp.zeros(n, dtype=cp.float64)
        _inv_prob = cp.zeros(n, dtype=cp.float64)
        _y_idx = cp.arange(n)

        def nll_and_grad_prealloc(theta_cp):
            """NLL + gradient with pre-allocated arrays."""
            beta = theta_cp[:p]
            thresh = theta_cp[p:]

            # Inline category probs using pre-allocated arrays
            _eta_pre[:] = Xs @ beta
            _diff_pre[:] = thresh[:, None] - _eta_pre[None, :]
            pi = family.link.inverse(_diff_pre)  # (K-1, n)

            _prob_pre[0] = pi[0]
            for j in range(1, K - 1):
                _prob_pre[j] = pi[j] - pi[j - 1]
            _prob_pre[K - 1] = 1.0 - pi[K - 2]
            _prob_c_pre[:] = cp.clip(_prob_pre, 1e-15, None)

            nll = -cp.sum(cp.log(_prob_c_pre[y, _y_idx])) / n

            # Gradient with pre-allocated arrays
            grad = cp.zeros(p + K - 1)
            for j in range(K - 1):
                _deriv_all[j] = self._ordered_link_derivative(_diff_pre[j], family)
            _inv_prob[:] = 1.0 / _prob_c_pre[y, _y_idx]

            for j in range(K - 1):
                mask_pos = (y == j)
                mask_neg = (y == j + 1)
                grad[p + j] = -cp.sum(
                    _inv_prob * (_deriv_all[j] * mask_pos - _deriv_all[j] * mask_neg)
                ) / n

            _scalar[:] = 0.0
            mask0 = (y == 0)
            mask_last = (y == K - 1)
            mask_mid = ~mask0 & ~mask_last
            _scalar[mask0] = -_deriv_all[0, mask0]
            _scalar[mask_last] = _deriv_all[K - 2, mask_last]
            idx_mid = cp.where(mask_mid)[0]
            _scalar[idx_mid] = (_deriv_all[y[idx_mid] - 1, idx_mid]
                                 - _deriv_all[y[idx_mid], idx_mid])
            grad[:p] -= Xs.T @ (_inv_prob * _scalar) / n

            return nll, grad

        # Initial theta (matching scipy and torch: start from scratch)
        theta = cp.zeros(p + K - 1, dtype=cp.float64)
        theta[p:] = cp.arange(0.5, K - 0.5, dtype=cp.float64)

        # L-BFGS parameters
        c1, c2 = 1e-4, 0.9
        max_ls = 25
        m_hist = 15
        min_iter = 5  # small guard against premature stop

        nll, grad = nll_and_grad_prealloc(theta)
        # Use infinity norm of gradient for convergence (matching scipy's gtol).
        gtol = self.tol
        grad_inf = float(cp.max(cp.abs(grad)))
        s_hist, y_hist, rho_hist = [], [], []
        H0 = 1.0
        n_iter = 0

        while n_iter < self.max_iter:
            # Check convergence using infinity norm (after min_iter iterations)
            if n_iter >= min_iter and grad_inf <= gtol:
                break
            s_old = theta.copy()
            g_old = grad.copy()
            nll_old = nll

            # Two-loop recursion
            q = grad.copy()
            alphas = []
            for i in range(len(s_hist) - 1, -1, -1):
                a = rho_hist[i] * cp.dot(s_hist[i], q)
                alphas.insert(0, a)
                q = q - a * y_hist[i]

            if s_hist:
                sy = float(cp.dot(s_hist[-1], y_hist[-1]))
                yy = float(cp.dot(y_hist[-1], y_hist[-1]))
                H0 = sy / (yy + 1e-30)

            r = H0 * q
            for i in range(len(s_hist)):
                b = rho_hist[i] * cp.dot(y_hist[i], r)
                r = r + s_hist[i] * (alphas[i] - b)

            d = -r
            gd = float(cp.dot(grad, d))
            if gd >= -1e-12:
                d = -grad
                gd = float(cp.dot(grad, d))

            slope = gd
            step = 1.0

            # Armijo line search
            for _ in range(max_ls):
                theta_new = theta + step * d
                nll_new, grad_new = nll_and_grad_prealloc(theta_new)
                if nll_new <= nll_old + c1 * step * slope:
                    break
                step *= 0.5
            else:
                theta_new = theta + step * d
                nll_new, grad_new = nll_and_grad_prealloc(theta_new)

            # Update L-BFGS history
            s_new = theta_new - s_old
            y_new_arr = grad_new - g_old
            sy_val = float(cp.dot(s_new, y_new_arr))
            if sy_val > 1e-12:
                if len(s_hist) >= m_hist:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
                s_hist.append(s_new)
                y_hist.append(y_new_arr)
                rho_hist.append(1.0 / sy_val)

            theta = theta_new
            nll = nll_new
            grad = grad_new
            grad_inf = float(cp.max(cp.abs(grad)))
            n_iter += 1

        # Extract results
        beta_scaled = theta[:p]
        self.coef_ = (beta_scaled / X_std).get()
        thresh_est = cp.sort(theta[p:])
        self.thresholds_ = np.concatenate([[-np.inf], thresh_est.get(), [np.inf]])
        self._X_mean = X_mean.get()
        self._X_std = X_std.get()
        self.n_iter_ = n_iter

    def _fit_torch_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using PyTorch autograd + LBFGS on GPU.

        X and y are already torch.Tensor on CUDA (converted by _to_array in fit()).
        No CuPy/NumPy bridge needed here — device purity is enforced upstream.
        """
        import torch

        assert isinstance(X, torch.Tensor), (
            f"_fit_torch_ordered expects torch.Tensor, got {type(X)}. "
            "Input should be converted by _to_array() before entering this method."
        )

        torch_device = X.device
        if X.dtype != torch.float64:
            X = X.to(torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(torch_device)
        elif y.dtype != torch.int64:
            y = y.to(torch.int64)

        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X_std = torch.clamp(X_std, 1e-10)
        Xs = (X - X_mean) / X_std

        # Parameters: [beta (p), thresholds (K-1)]
        # Initialize thresholds uniformly
        theta_init = torch.zeros(p + K - 1, dtype=torch.float64, device=torch_device)
        theta_init[p:] = torch.arange(0.5, K - 0.5, dtype=torch.float64, device=torch_device)
        theta = torch.nn.Parameter(theta_init.clone())

        n_samples = torch.tensor(float(n), dtype=torch.float64, device=torch_device)
        y_idx = torch.arange(n, device=torch_device)

        def closure():
            optimizer.zero_grad()
            beta = theta[:p]
            thresh = theta[p:]

            # Compute category probabilities with autograd
            eta = Xs @ beta  # (n,)
            diff = thresh[:, None] - eta[None, :]  # (K-1, n)

            # Link inverse via family
            pi = family.link.inverse(diff)  # (K-1, n)

            # Category probabilities P(y=j)
            prob = torch.zeros((K, n), dtype=torch.float64, device=torch_device)
            prob[0] = pi[0]
            for j in range(1, K - 1):
                prob[j] = pi[j] - pi[j - 1]
            prob[K - 1] = 1.0 - pi[K - 2]

            # Negative log-likelihood
            prob_c = torch.clamp(prob[y, y_idx], 1e-15, None)
            nll = -torch.mean(torch.log(prob_c))

            nll.backward()
            return nll

        # Torch L-BFGS — use strong_wolfe line_search for robust convergence
        # (ordered logit NLL landscape has steep gradients that cause lr=1.0
        #  without line search to diverge into degenerate local minima)
        try:
            optimizer = torch.optim.LBFGS(
                [theta],
                lr=1.0,
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                tolerance_change=self.tol * 1e-3,
                line_search_fn='strong_wolfe',
                max_eval=self.max_iter * 25,
            )
        except TypeError:
            raise RuntimeError(
                "torch.optim.LBFGS with line_search_fn='strong_wolfe' is required "
                "for ordered model fitting. Upgrade to PyTorch >= 1.13 or use "
                "a different backend (numpy or cupy)."
            )

        loss = optimizer.step(closure)

        # Extract results
        theta_final = theta.detach()
        beta_scaled = theta_final[:p]
        thresh_est = torch.sort(theta_final[p:])[0]

        self.coef_ = (beta_scaled / X_std).cpu().numpy()
        self.thresholds_ = np.concatenate([[-np.inf], thresh_est.cpu().numpy(), [np.inf]])
        self._X_mean = X_mean.cpu().numpy()
        self._X_std = X_std.cpu().numpy()
        try:
            state_dict = optimizer.state_dict()
            n_iter = 0
            for group in state_dict.get('state', {}).values():
                n_iter = max(n_iter, group.get('n_iter', 0))
            self.n_iter_ = n_iter if n_iter > 0 else self.max_iter
        except Exception:
            self.n_iter_ = self.max_iter

    def _ordered_category_probs(self, X, beta, thresh, family, K):
        """Compute category probabilities P(y=j|X), shape (K, n)."""
        eta = X @ beta  # (n,)
        pi = family.link.inverse(thresh[:, None] - eta[None, :])  # (K-1, n)

        xp = _xp_arr(X)
        prob = xp.zeros((K, X.shape[0]), dtype=getattr(X, 'dtype', None))
        prob[0] = pi[0]
        for j in range(1, K - 1):
            prob[j] = pi[j] - pi[j - 1]
        prob[K - 1] = 1.0 - pi[K - 2]
        return prob

    def _ordered_gradient(self, X, y, beta, thresh, prob, prob_clipped, family, K, n):
        """Compute analytical gradient of the negative log-likelihood (vectorized)."""
        xp = _xp_arr(X)
        p = X.shape[1]
        n_thresh = K - 1
        dim = p + n_thresh
        grad = xp.zeros(dim)

        eta = X @ beta  # (n,)

        # Link derivative at all threshold positions: shape (n_thresh, n)
        diff = thresh[:, None] - eta[None, :]  # (n_thresh, n)
        deriv_all = xp.empty_like(diff)
        for j in range(n_thresh):
            deriv_all[j] = self._ordered_link_derivative(diff[j], family)

        # inv_prob[i] = 1 / P(y[i] | X[i]), shape (n,)
        inv_prob = 1.0 / prob_clipped[y, xp.arange(n)]  # (n,)

        # dP_dthresh contribution for each (j, i):
        #   +deriv_all[j, i] if j == y[i]
        #   -deriv_all[j, i] if j == y[i] - 1
        # Vectorized: for each j, count how many samples have y==j (positive)
        # and y==j+1 (negative).
        dP_dthresh_j = xp.zeros(n_thresh)
        for j in range(n_thresh):
            mask_pos = (y == j)
            mask_neg = (y == j + 1)
            dP_dthresh_j[j] = xp.sum(inv_prob * (deriv_all[j] * mask_pos - deriv_all[j] * mask_neg))

        grad[p:] -= dP_dthresh_j / n

        # dP_dbeta for sample i: X[i] * scalar_i
        # scalar_i = -(deriv_all[0, i]) if y[i]==0
        #           (deriv_all[y[i]-1, i] - deriv_all[y[i], i]) if 0 < y[i] < K-1
        #           (deriv_all[n_thresh-1, i]) if y[i]==K-1
        scalar = xp.empty(n)
        mask0 = (y == 0)
        mask_last = (y == K - 1)
        mask_mid = ~mask0 & ~mask_last
        scalar[mask0] = -deriv_all[0, mask0]
        scalar[mask_last] = deriv_all[n_thresh - 1, mask_last]
        # For middle: deriv[y[i]-1] - deriv[y[i]]
        idx_mid = xp.where(mask_mid)[0]
        scalar[idx_mid] = (deriv_all[y[idx_mid] - 1, idx_mid]
                           - deriv_all[y[idx_mid], idx_mid])

        grad[:p] -= X.T @ (inv_prob * scalar) / n

        return grad

    def _ordered_link_derivative(self, x, family):
        """First derivative of link inverse F'(x) = density at x.

        For logit: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
        For probit: normal PDF φ(x).
        Both paths are backend-agnostic (numpy/cupy/torch).
        """
        if family.link.name == "probit":
            mod = type(x).__module__
            if mod.startswith('cupy'):
                from statgpu.inference._distributions_backend import get_distribution
                norm_dist = get_distribution("norm", backend="cupy")
                return norm_dist.pdf(x)
            elif mod.startswith('torch'):
                import torch
                return torch.exp(-0.5 * x.square()) / torch.sqrt(2.0 * torch.pi)
            from scipy.stats import norm
            return norm.pdf(x)
        # logit: F * (1 - F) — element-wise, works for any backend
        F = family.link.inverse(x)
        return F * (1.0 - F)

    def _ordered_link_second_derivative(self, x, family):
        """Second derivative of link inverse F''(x)."""
        mod = type(x).__module__
        is_cupy = mod.startswith('cupy')
        is_torch = mod.startswith('torch')

        if family.link.name == "logit":
            F = family.link.inverse(x)
            return F * (1.0 - F) * (1.0 - 2.0 * F)
        elif family.link.name == "probit":
            # F''(x) = -x * φ(x) for standard normal PDF φ
            if is_cupy:
                from statgpu.inference._distributions_backend import get_distribution
                norm_dist = get_distribution("norm", backend="cupy")
                return -x * norm_dist.pdf(x)
            elif is_torch:
                import torch
                return -x * torch.exp(-0.5 * x.square()) / torch.sqrt(2.0 * torch.pi)
            from scipy.stats import norm
            return -x * norm.pdf(x)
        F = family.link.inverse(x)
        return F * (1.0 - F) * (1.0 - 2.0 * F)

    def predict_proba(self, X):
        """Predict class probabilities P(y=j|X).

        Backend-agnostic: uses the same backend that was used during fit().
        Returns a NumPy array by convention (small output, consumed on CPU).
        """
        self._check_is_fitted()
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = self.n_categories

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        X_arr = self._to_array(X, backend=backend_name)

        if backend_name == "cupy":
            import cupy as cp
            coef = cp.asarray(self.coef_)
            X_mean = cp.asarray(self._X_mean)
            X_std = cp.asarray(self._X_std)
            thresholds = cp.asarray(self.thresholds_)
        elif backend_name == "torch":
            import torch
            torch_device = X_arr.device if hasattr(X_arr, 'device') else torch.device('cuda')
            coef = torch.from_numpy(np.asarray(self.coef_)).to(torch_device)
            X_mean = torch.from_numpy(np.asarray(self._X_mean)).to(torch_device)
            X_std = torch.from_numpy(np.asarray(self._X_std)).to(torch_device)
            thresholds = torch.from_numpy(np.asarray(self.thresholds_)).to(torch_device)
        else:
            coef = self.coef_
            X_mean = self._X_mean
            X_std = self._X_std
            thresholds = self.thresholds_

        X_scaled = (X_arr - X_mean) / X_std
        eta = X_scaled @ coef
        family = self._get_family()
        diff = thresholds[:, None] - eta[None, :]
        pi = family.link.inverse(diff)  # (K+1, n) with -inf/+inf thresholds

        if backend_name == "torch":
            import torch
            proba = torch.diff(pi, dim=0).T  # (n, K)
            return _to_numpy(proba)
        elif backend_name == "cupy":
            import cupy as cp
            proba = cp.diff(pi, axis=0).T  # (n, K)
            return _to_numpy(proba)
        else:
            return np.diff(pi, axis=0).T  # (n, K)

    def predict(self, X):
        """Predict class labels.

        Backend-agnostic: computes argmax on the native backend, returns NumPy.
        """
        self._check_is_fitted()

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        X_arr = self._to_array(X, backend=backend_name)

        # Inline the prediction path to avoid double GPU→CPU round-trip
        K = self.n_categories
        if backend_name == "cupy":
            import cupy as cp
            coef = cp.asarray(self.coef_)
            X_mean = cp.asarray(self._X_mean)
            X_std = cp.asarray(self._X_std)
            thresholds = cp.asarray(self.thresholds_)
            X_scaled = (X_arr - X_mean) / X_std
            eta = X_scaled @ coef
            family = self._get_family()
            diff = thresholds[:, None] - eta[None, :]
            pi = family.link.inverse(diff)
            proba = cp.diff(pi, axis=0).T
            return _to_numpy(cp.argmax(proba, axis=1))
        elif backend_name == "torch":
            import torch
            torch_device = X_arr.device if hasattr(X_arr, 'device') else torch.device('cuda')
            coef = torch.from_numpy(np.asarray(self.coef_)).to(torch_device)
            X_mean = torch.from_numpy(np.asarray(self._X_mean)).to(torch_device)
            X_std = torch.from_numpy(np.asarray(self._X_std)).to(torch_device)
            thresholds = torch.from_numpy(np.asarray(self.thresholds_)).to(torch_device)
            X_scaled = (X_arr - X_mean) / X_std
            eta = X_scaled @ coef
            family = self._get_family()
            diff = thresholds[:, None] - eta[None, :]
            pi = family.link.inverse(diff)
            proba = torch.diff(pi, dim=0).T
            return _to_numpy(torch.argmax(proba, dim=1))
        else:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

    def score(self, X, y):
        """Return mean accuracy on the given test data and labels.

        Uses the same backend as fit() for the computation.
        """
        self._check_is_fitted()

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        y_true = self._to_array(y, backend=backend_name)
        y_pred = self.predict(X)
        y_pred_arr = self._to_array(y_pred, backend=backend_name)

        if backend_name == "cupy":
            import cupy as cp
            return float(cp.mean(y_pred_arr == y_true).item())
        elif backend_name == "torch":
            import torch
            y_pred_t = y_pred_arr.to(torch.int64) if hasattr(y_pred_arr, 'to') else y_pred_arr
            y_true_t = y_true.to(torch.int64) if hasattr(y_true, 'to') else y_true
            return float(torch.mean((y_pred_t == y_true_t).to(torch.float64)).item())
        else:
            return float(np.mean(np.asarray(y_pred) == np.asarray(y_true)))
