"""
Penalized GLM estimators.

This module keeps the GLM-specific optimization path explicit.  The central
implementation accepts a GLM loss name, while public typed estimators expose
gaussian, logistic, and poisson models without the old ``loss=...`` switch on
``PenalizedLinearRegression``.
"""

from typing import Optional, Union, Any, Dict, List
import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import get_backend, _get_torch_device_str, _to_numpy


class PenalizedGeneralizedLinearModel(BaseEstimator):
    """
    Penalized generalized linear model with pluggable GLM loss and penalty.

    Minimizes: loss(X, y, w) + penalty(w)

    Parameters
    ----------
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'logistic', 'poisson'.
    penalty : str or Penalty
        Penalty type: 'l1', 'l2', 'elasticnet', or a Penalty instance.
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'logistic', 'poisson'.
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'irls', 'newton'.
        'auto' selects the current best path for the resolved backend:
        exact for Gaussian L2, CPU IRLS for smooth logistic/poisson L2, and
        GPU/Torch FISTA for GPU-backed penalized GLMs.
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Only used when penalty='elasticnet'.
    penalty_kwargs : dict, optional
        Additional arguments passed to the penalty constructor.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    cpu_solver : str, default='fista'
        CPU solver: 'fista' or 'coordinate_descent'.
    solver : str, default='fista'
        GPU solver: 'fista'.
    lipschitz_L : float, optional
        Pre-computed Lipschitz constant.
    gpu_memory_cleanup : bool, default=False
        If True, free GPU memory pool after fitting.

    Examples
    --------
    # Lasso
    >>> model = PenalizedLinearRegression(penalty='l1', alpha=0.1)

    # Ridge
    >>> model = PenalizedLinearRegression(penalty='l2', alpha=1.0)

    # Elastic Net
    >>> model = PenalizedLinearRegression(
    ...     penalty='elasticnet', alpha=0.1, l1_ratio=0.5
    ... )
    """

    def __init__(
        self,
        loss: str = "squared_error",
        penalty: Union[str, "Penalty"] = "l1",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[Dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty_kwargs = penalty_kwargs or {}
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cpu_solver = cpu_solver.lower()
        self.solver = solver.lower()
        self.lipschitz_L = lipschitz_L
        self.gpu_memory_cleanup = gpu_memory_cleanup
        self.compute_inference = compute_inference
        self.cov_type = str(cov_type).lower()
        self.hac_maxlags = hac_maxlags
        self.stopping = str(stopping).lower()

        # Internal state
        self._penalty: Optional["Penalty"] = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._selected_solver = None
        self._selected_backend_name = None

    def _resolve_penalty(self) -> "Penalty":
        """Resolve penalty string or instance to a Penalty object."""
        # Lazy import to avoid circular dependency
        from statgpu.penalties import get_penalty, Penalty

        if isinstance(self.penalty, Penalty):
            return self.penalty

        kwargs = {**self.penalty_kwargs, "alpha": self.alpha}
        if self.penalty == "elasticnet":
            kwargs["l1_ratio"] = self.l1_ratio

        return get_penalty(self.penalty, **kwargs)

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """
        Fit penalized GLM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Training data. Required when ``formula`` is None.
        y : array-like of shape (n_samples,), optional
            Target values. Required when ``formula`` is None.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        formula : str, optional
            R-style formula string, e.g. ``"y ~ x1 + C(group)"``.
        data : pandas.DataFrame, optional
            Data used to evaluate ``formula``.

        Returns
        -------
        self : PenalizedLinearRegression
            Fitted estimator.
        """
        _orig_fit_intercept = self.fit_intercept
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            from statgpu.core.formula import FormulaParser

            parser = FormulaParser(formula)
            y, X, design_info = parser.eval(data)
            formula_column_names = list(design_info.column_names)
            self._design_info = design_info
            self._formula_has_intercept = "Intercept" in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != "Intercept"]
            if self._formula_has_intercept:
                X = np.delete(X, formula_column_names.index("Intercept"), axis=1)
                self.fit_intercept = True
            else:
                # Formula syntax owns intercept semantics, matching statsmodels/R.
                self.fit_intercept = False
        else:
            if X is None or y is None:
                raise ValueError("Either formula+data or X+y must be provided.")
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None

        self.fit_intercept = _orig_fit_intercept
        self._penalty = self._resolve_penalty()
        self._validate_solver_penalty()
        self._loss = self._resolve_loss()

        # Resolve the actual backend before auto-selecting the solver. This
        # keeps solver="auto" device-aware: CPU can use IRLS for smooth GLMs,
        # while GPU/Torch stays on accelerator-capable FISTA.
        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        selected_solver = self._select_solver(self._loss, backend_name=backend_name)
        self._selected_solver = selected_solver
        self._selected_backend_name = backend_name

        # Handle penalties requiring initialization (e.g., Adaptive Lasso)
        if self._penalty.requires_init:
            init_coef = self._fit_initial(X, y)
            self._penalty.set_weights(init_coef)

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)

        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif backend_name == "cupy":
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        self._fitted = True
        return self

    def _resolve_loss(self):
        """Resolve loss string to a GLMLoss object."""
        from statgpu.glm_core import get_glm_loss

        return get_glm_loss(self.loss)

    def _validate_solver_penalty(self):
        """Validate solver/penalty combinations before backend dispatch."""
        solver_name = self.solver
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        non_smooth = {
            "l1",
            "elasticnet",
            "adaptive_l1",
            "adaptive_lasso",
            "group_lasso",
            "gl",
            "scad",
            "mcp",
        }
        if self.solver == "exact":
            if self.loss != "squared_error" or penalty_name != "l2":
                raise ValueError(
                    "solver='exact' is only supported for squared-error L2/Ridge models."
                )
            return
        if solver_name == "irls" and penalty_name != "l2":
            raise ValueError(
                "solver='irls' only supports smooth L2 penalized GLM objectives."
            )
        if solver_name in ("newton", "lbfgs") and penalty_name in non_smooth:
            raise ValueError(
                f"solver='{solver_name}' only supports smooth objectives; "
                f"use solver='fista' for penalty='{penalty_name}'."
            )
        if solver_name != "lbfgs":
            return

    def _select_solver(self, loss, backend_name=None):
        """Auto-select solver based on loss, penalty, and resolved backend."""
        if self.solver != "auto":
            return self.solver
        if self.loss == "squared_error" and self._penalty.name == "l2":
            return "exact"
        if self._penalty.name in ("l1", "elasticnet"):
            return "fista"
        if backend_name in ("cupy", "torch"):
            return "fista"
        if getattr(loss, "has_hessian", False):
            return "irls"
        return "fista"

    def _fit_initial(self, X, y):
        """Fit initial model for penalties requiring initialization."""
        # Use Ridge or OLS for initial estimate
        from statgpu.linear_model._ridge import Ridge

        init_model = Ridge(
            alpha=0.1,
            fit_intercept=self.fit_intercept,
            device=self.device,
        )
        init_model.fit(X, y)
        return init_model.coef_

    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU (FISTA or coordinate descent)."""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Route to loss-aware solver for non-squared_error loss
        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="numpy"
        )
        if self.loss != "squared_error":
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "numpy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "numpy")
            return
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "numpy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "numpy")
            return

        # Original squared-error path (backward compatible)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = 0.0
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        # Precompute for gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered.flatten()

        pen = self._penalty
        if solver_name == "exact":
            if pen.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            self.coef_ = self._solve_exact_numpy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self.fit_intercept:
                self.intercept_ = float(y_mean - X_mean @ self.coef_)
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self._params = self.coef_.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
            return

        # Lipschitz constant: L = λ_max(XtX) / n
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            eigvals = np.linalg.eigvalsh(XtX)
            L = float(eigvals[-1]) / n_samples

        if L <= 0:
            self.coef_ = np.zeros(n_features)
            self.n_iter_ = 0
        else:
            step = 1.0 / L

            if self.cpu_solver == "fista":
                # FISTA
                coef = np.zeros(n_features)
                y_k = coef.copy()
                t_k = 1.0

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    # Gradient step
                    grad = (XtX @ y_k - Xty) / n_samples
                    w_tilde = y_k - step * grad

                    # Proximal step
                    coef = pen.proximal(w_tilde, step, backend="numpy")

                    # Momentum update
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                    beta = (t_k - 1.0) / t_new
                    y_k = coef + beta * (coef - coef_old)
                    t_k = t_new

                    self.n_iter_ = iteration + 1

                    # Convergence check
                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        break
            else:
                # Coordinate descent (for L1-type penalties)
                X_sq_norms = np.diag(XtX)
                coef = np.zeros(n_features)

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    for j in range(n_features):
                        rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]

                        if pen.name == "l1":
                            # Soft thresholding
                            thresh = self.alpha * n_samples
                            if X_sq_norms[j] > 1e-10:
                                coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                            else:
                                coef[j] = 0.0
                        elif pen.name == "elasticnet":
                            # Elastic net soft thresholding
                            thresh = self.alpha * self.l1_ratio * n_samples
                            l2_scale = 1.0 + self.alpha * (1 - self.l1_ratio)
                            if X_sq_norms[j] > 1e-10:
                                st = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0)
                                coef[j] = st / (X_sq_norms[j] * l2_scale)
                            else:
                                coef[j] = 0.0

                    self.n_iter_ = iteration + 1

                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        break

        # Compute intercept and store results
        if L > 0:
            self.coef_ = coef

        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self._params = self.coef_.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU (CuPy) with FISTA."""
        import cupy as cp

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="cupy"
        )
        if solver_name not in ("fista", "auto", "exact", "irls", "newton", "lbfgs"):
            raise ValueError(
                "CuPy backend supports solver='fista', 'exact', 'irls', "
                "'newton', and 'lbfgs'."
            )

        n_samples, n_features = X.shape
        self._nobs = n_samples

        if self.loss != "squared_error":
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "cupy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "cupy")
            return
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "cupy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "cupy")
            return

        X = cp.asarray(X)
        y = cp.asarray(y)

        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw

        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1)

        # Precompute
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            coef = self._solve_exact_cupy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            coef_np = coef.get()
            if self.fit_intercept:
                self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
            self._cleanup_cuda_memory()
            return

        # Lipschitz constant
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            eigvals = cp.linalg.eigvalsh(XtX)
            L = float(eigvals[-1]) / n_samples

        if L <= 0:
            coef = cp.zeros(n_features, dtype=X.dtype)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            coef = cp.zeros(n_features, dtype=X.dtype)
            y_k = coef.copy()
            t_k = cp.array(1.0, dtype=X.dtype)

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad

                coef = self._penalty.proximal(w_tilde, step, backend="cupy")

                t_new = (1.0 + cp.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                self.n_iter_ = iteration + 1

                if float(cp.sum(cp.abs(coef - coef_old))) < self.tol:
                    break

        # Transfer to CPU
        coef_np = coef.get()

        if self.fit_intercept:
            self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        # Cleanup
        self._cleanup_cuda_memory()

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FISTA."""
        import torch

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="torch"
        )
        if solver_name not in ("fista", "auto", "exact", "irls", "newton", "lbfgs"):
            raise ValueError(
                "Torch backend supports solver='fista', 'exact', 'irls', "
                f"'newton', and 'lbfgs', got '{self.solver}'."
            )

        n_samples, n_features = X.shape
        self._nobs = n_samples

        if self.loss != "squared_error":
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "torch")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "torch")
            return
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "torch")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "torch")
            return

        torch_device = _get_torch_device_str()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch_device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(torch_device)
        if X.dtype != torch.float64:
            X = X.to(torch.float64)
        if y.dtype != torch.float64:
            y = y.to(torch.float64)

        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                sample_weight = torch.from_numpy(sample_weight).to(torch_device)
            sqrt_sw = torch.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        if self.fit_intercept:
            X_mean = torch.mean(X, dim=0)
            y_mean = torch.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = torch.tensor(0.0, dtype=torch.float64, device=torch_device)
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1)

        # Precompute
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            coef = self._solve_exact_torch(XtX, Xty, n_samples)
            self.n_iter_ = 1
            coef_np = coef.cpu().numpy()
            if self.fit_intercept:
                self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
            self._cleanup_torch_memory()
            return

        # Lipschitz constant
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            eigvals = torch.linalg.eigvalsh(XtX)
            L = float(eigvals[-1]) / n_samples

        if L <= 0:
            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            y_k = coef.clone()
            t_k = torch.tensor(1.0, dtype=X.dtype, device=X.device)

            for iteration in range(self.max_iter):
                coef_old = coef.clone()

                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad

                coef = self._penalty.proximal(w_tilde, step, backend="torch")

                t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                self.n_iter_ = iteration + 1

                if float(torch.sum(torch.abs(coef - coef_old)).item()) < self.tol:
                    break

        # Transfer to CPU
        coef_np = coef.cpu().numpy()

        if self.fit_intercept:
            self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        self._cleanup_torch_memory()

    def _fit_gpu_loss(self, X, y, sample_weight=None):
        """CuPy FISTA for non-squared-error losses (logistic, poisson).

        Mirrors _fit_cpu_loss but keeps arrays on GPU during computation.
        """
        import cupy as cp
        from statgpu.glm_core._solver import fista_solver

        X_arr = cp.asarray(X)
        y_arr = cp.asarray(y)

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            X_aug = cp.column_stack([X_arr, cp.ones(X_arr.shape[0])])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="cupy"):
                    result = pen.proximal(w, step, backend=backend)
                    result[-1] = w[-1]
                    return result
                name = pen.name

            full_coef, n_iter = fista_solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = full_coef.get()[:p]
            self.intercept_ = float(full_coef.get()[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            X_arr = X_arr - cp.mean(X_arr, axis=0)
            y_arr = y_arr - cp.mean(y_arr)

            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.get()
            self.n_iter_ = n_iter
            self.intercept_ = float(cp.mean(y_arr) - cp.mean(X_arr, axis=0) @ self.coef_)
        else:
            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.get()
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))
        self._cleanup_cuda_memory()

    def _fit_torch_loss(self, X, y, sample_weight=None):
        """Torch FISTA for non-squared-error losses (logistic, poisson).

        Mirrors _fit_cpu_loss but keeps arrays on GPU during computation.
        """
        import torch
        from statgpu.glm_core._solver import fista_solver

        torch_device = _get_torch_device_str()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch_device).to(torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(torch_device).to(torch.float64)

        X_arr = X
        y_arr = y

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            ones_col = torch.ones(X_arr.shape[0], dtype=torch.float64, device=torch_device)
            X_aug = torch.column_stack([X_arr, ones_col])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="torch"):
                    result = pen.proximal(w, step, backend=backend)
                    result[-1] = w[-1]
                    return result
                name = pen.name

            full_coef, n_iter = fista_solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            full_np = full_coef.cpu().numpy()
            self.coef_ = full_np[:p]
            self.intercept_ = float(full_np[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            X_arr = X_arr - torch.mean(X_arr, dim=0)
            y_arr = y_arr - torch.mean(y_arr)

            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.cpu().numpy()
            self.n_iter_ = n_iter
            self.intercept_ = float(torch.mean(y_arr) - torch.mean(X_arr, dim=0) @ self.coef_)
        else:
            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.cpu().numpy()
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))
        self._cleanup_torch_memory()

    def _ridge_alpha_for_exact(self) -> float:
        """Return L2 alpha for the exact Ridge normal equations."""
        return float(getattr(self._penalty, "alpha", self.alpha))

    def _solve_exact_numpy(self, XtX, Xty, n_samples):
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * np.eye(p, dtype=XtX.dtype)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ Xty

    def _solve_exact_cupy(self, XtX, Xty, n_samples):
        import cupy as cp

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * cp.eye(p, dtype=XtX.dtype)
        try:
            return cp.linalg.solve(A, Xty)
        except Exception:
            return cp.linalg.pinv(A) @ Xty

    def _solve_exact_torch(self, XtX, Xty, n_samples):
        import torch

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * torch.eye(
            p, dtype=XtX.dtype, device=XtX.device
        )
        try:
            return torch.linalg.solve(A, Xty)
        except RuntimeError:
            return torch.linalg.pinv(A) @ Xty

    def _prepare_predict_X(self, X):
        """Apply stored formula design metadata to DataFrame inputs."""
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
        return np.asarray(X)

    def predict(self, X):
        """
        Predict using fitted model.

        For squared_error: returns linear prediction.
        For logistic: returns binary class labels.
        For poisson: returns exp(linear prediction) (count values).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self.fit_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            if self.loss == "logistic":
                p = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -500, 500)))
                return (p > 0.5).astype(float)
            if self.loss == "poisson":
                return cp.exp(raw)
            return raw
        if device == Device.TORCH:
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self.fit_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            if self.loss == "logistic":
                p = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -500, 500)))
                return (p > 0.5).to(raw.dtype)
            if self.loss == "poisson":
                return torch.exp(raw)
            return raw

        raw = X @ self.coef_
        if self.fit_intercept:
            raw += self.intercept_

        # Apply link inverse for GLM losses
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
            return (p > 0.5).astype(float)
        elif self.loss == "poisson":
            return np.exp(raw)
        return raw

    def score(self, X, y):
        """
        Return R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        r2 : float
            R² score.
        """
        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            yb = cp.asarray(self._to_array(y, Device.CUDA))
            ss_res = cp.sum((yb - y_pred) ** 2)
            ss_tot = cp.sum((yb - cp.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).get()) if float(ss_tot.get()) > 0 else 0.0
        if device == Device.TORCH:
            import torch
            yb = self._to_array(y, Device.TORCH, backend="torch").to(y_pred.dtype)
            ss_res = torch.sum((yb - y_pred) ** 2)
            ss_tot = torch.sum((yb - torch.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
        y = np.asarray(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _family_for_loss(self):
        from statgpu.glm_core._family import Binomial, Gaussian, Poisson

        if self.loss == "logistic":
            return Binomial()
        if self.loss == "poisson":
            return Poisson()
        return Gaussian()

    def _column_stack(self, arrays, backend_name):
        if backend_name == "cupy":
            import cupy as cp
            return cp.column_stack(arrays)
        if backend_name == "torch":
            import torch
            return torch.column_stack(arrays)
        return np.column_stack(arrays)

    def _ones(self, n, backend_name, ref):
        if backend_name == "cupy":
            import cupy as cp
            return cp.ones(n, dtype=ref.dtype)
        if backend_name == "torch":
            import torch
            return torch.ones(n, dtype=ref.dtype, device=ref.device)
        return np.ones(n, dtype=getattr(ref, "dtype", np.float64))

    def _selective_penalty(self, p, backend_name):
        """Penalty wrapper that leaves the last intercept coefficient free."""
        pen = self._penalty
        alpha = float(getattr(pen, "alpha", self.alpha))

        class SelectivePenalty:
            name = pen.name

            def proximal(self, w, step, backend=backend_name):
                result = pen.proximal(w, step, backend=backend)
                result[-1] = w[-1]
                return result

            def smooth_value(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                active = coef[:p]
                if backend_name == "cupy":
                    import cupy as cp
                    return 0.5 * alpha * cp.sum(active * active)
                if backend_name == "torch":
                    import torch
                    return 0.5 * alpha * torch.sum(active * active)
                return 0.5 * alpha * np.sum(active * active)

            def smooth_gradient(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                if backend_name == "cupy":
                    import cupy as cp
                    grad = cp.zeros_like(coef)
                elif backend_name == "torch":
                    import torch
                    grad = torch.zeros_like(coef)
                else:
                    grad = np.zeros_like(coef)
                grad[:p] = alpha * coef[:p]
                return grad

            def smooth_hessian(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                if backend_name == "cupy":
                    import cupy as cp
                    diag = cp.zeros(coef.shape[0], dtype=coef.dtype)
                    diag[:p] = alpha
                    return cp.diag(diag)
                if backend_name == "torch":
                    import torch
                    diag = torch.zeros(
                        coef.shape[0], dtype=coef.dtype, device=coef.device
                    )
                    diag[:p] = alpha
                    return torch.diag(diag)
                diag = np.zeros(coef.shape[0], dtype=coef.dtype)
                diag[:p] = alpha
                return np.diag(diag)

        return SelectivePenalty()

    def _fit_loss_backend(self, X, y, sample_weight, solver_name, backend_name):
        """Fit GLMLoss + Penalty without changing the selected backend."""
        from statgpu.glm_core._solver import (
            fista_solver,
            lbfgs_solver,
            newton_solver,
        )

        X_arr = X
        y_arr = y
        if self.fit_intercept:
            p = X_arr.shape[1]
            X_work = self._column_stack(
                [X_arr, self._ones(X_arr.shape[0], backend_name, X_arr)],
                backend_name,
            )
            pen = self._selective_penalty(p, backend_name)
        else:
            p = X_arr.shape[1]
            X_work = X_arr
            pen = self._penalty

        if solver_name in ("auto", "fista"):
            params, n_iter = fista_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )
        elif solver_name == "newton":
            params, n_iter = newton_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )
        elif solver_name == "lbfgs":
            params, n_iter = lbfgs_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None,
            )
        else:
            raise ValueError(f"Unsupported solver: {solver_name}")

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self.fit_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _fit_irls_backend(self, X, y, sample_weight=None, backend_name="numpy"):
        """Fit smooth L2 GLM via IRLS on the selected backend."""
        from statgpu.glm_core._irls import IRLSSolver

        if str(getattr(self._penalty, "name", self.penalty)).lower() != "l2":
            raise ValueError("solver='irls' only supports L2 penalties.")

        X_arr = X
        y_arr = y
        n_samples = X_arr.shape[0]
        if self.fit_intercept:
            X_work = self._column_stack(
                [self._ones(X_arr.shape[0], backend_name, X_arr), X_arr],
                backend_name,
            )
        else:
            X_work = X_arr

        solver = IRLSSolver(
            self._family_for_loss(), max_iter=self.max_iter, tol=self.tol
        )
        params, n_iter = solver.fit(
            X_work, y_arr,
            sample_weight=sample_weight,
            ridge_alpha=float(n_samples * self.alpha),
            ridge_penalize_intercept=False if self.fit_intercept else True,
            backend=backend_name,
        )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self.fit_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _fit_cpu_loss(self, X, y, sample_weight=None, solver="fista"):
        """Fit using loss-aware solver (FISTA with arbitrary loss).

        For GLM losses (logistic, poisson) with intercept, augments X with
        a column of ones and uses a selective penalty (no penalty on intercept)
        to converge to the correct joint optimum.
        """
        from statgpu.glm_core._solver import fista_solver

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            # Augment X with intercept column
            X_aug = np.column_stack([X_arr, np.ones(X_arr.shape[0])])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="numpy"):
                    result = pen.proximal(w, step, backend=backend)
                    result[-1] = w[-1]  # intercept not penalized
                    return result
                name = pen.name

            full_coef, n_iter = fista_solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = full_coef[:p]
            self.intercept_ = float(full_coef[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            # Squared error: center X and y, fit once
            X_arr = X_arr - X_arr.mean(axis=0)
            y_arr = y_arr - y_arr.mean()

            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef
            self.n_iter_ = n_iter
            self.intercept_ = float(np.mean(y) - np.mean(X, axis=0) @ coef)
        else:
            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))

    def _fit_cpu_irls(self, X, y, sample_weight=None):
        """Fit using IRLS for smooth penalty + smooth loss (e.g., Logistic/Poisson + L2).

        Each IRLS iteration:
            1. Compute working response z and weights W
            2. Solve: (X'WX + n*alpha*I) params = X'Wz
        """
        from statgpu.glm_core._irls import IRLSSolver
        from statgpu.glm_core._family import Binomial, Poisson, Gaussian

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = X_arr.shape[0]

        # Add intercept column if needed
        if self.fit_intercept:
            X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        # L2 penalty: for objective min loss/n + alpha*0.5*||w||^2,
        # IRLS uses unnormalized X'WX, so ridge = n * alpha.
        # Don't penalize the intercept column (matches sklearn/FISTA behavior).
        ridge_alpha = float(n_samples * self.alpha)
        ridge_penalize_intercept = False if self.fit_intercept else True

        # Select family
        if self.loss == "logistic":
            family = Binomial()
        elif self.loss == "poisson":
            family = Poisson()
        else:
            family = Gaussian()

        solver = IRLSSolver(family, max_iter=self.max_iter, tol=self.tol)
        params, n_iter = solver.fit(
            X_arr, y_arr, sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=ridge_penalize_intercept,
            backend="numpy",
        )

        self.n_iter_ = n_iter

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
            self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
        else:
            self.intercept_ = 0.0
            self.coef_ = params.copy()
            self._params = np.asarray(self.coef_).copy()

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))

    def _cleanup_cuda_memory(self):
        """Free CuPy memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Free Torch memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class PenalizedLinearRegression(PenalizedGeneralizedLinearModel):
    """Gaussian penalized regression.

    This typed estimator replaces the old ``PenalizedLinearRegression(loss=...)``
    entry point.  Use ``PenalizedLogisticRegression`` or
    ``PenalizedPoissonRegression`` for non-gaussian GLMs.
    """

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l1",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[Dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
    ):
        super().__init__(
            loss="squared_error",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
        )


class PenalizedLogisticRegression(PenalizedGeneralizedLinearModel):
    """Binomial/logistic penalized GLM."""

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[Dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
    ):
        super().__init__(
            loss="logistic",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
        )

    def predict_proba(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_predict_X(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self.fit_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            p1 = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -500, 500)))
            return cp.column_stack([1.0 - p1, p1])
        if device == Device.TORCH:
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self.fit_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            p1 = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -500, 500)))
            return torch.column_stack([1.0 - p1, p1])
        raw = X @ self.coef_
        if self.fit_intercept:
            raw += self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
        return np.column_stack([1.0 - p1, p1])


class PenalizedPoissonRegression(PenalizedGeneralizedLinearModel):
    """Poisson penalized GLM."""

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[Dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
    ):
        super().__init__(
            loss="poisson",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
        )
