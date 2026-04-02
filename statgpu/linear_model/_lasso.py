"""
Lasso regression with full statistical inference and GPU support.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class Lasso(BaseEstimator):
    """
    Lasso regression (L1 regularization) with GPU acceleration
    and full statistical inference.

    CPU solver supports multiple algorithms (coordinate descent by default, and FISTA when cpu_solver='fista').
    GPU solver supports multiple algorithms via `solver` (e.g. FISTA / ADMM).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
        Must be non-negative.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations for coordinate descent.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    cpu_solver : str, default='coordinate_descent'
        CPU optimization algorithm: 'coordinate_descent' or 'fista'.
        GPU uses the `solver` parameter instead.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        inference_method: str = "naive_ols",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.stopping = stopping.lower()
        self.inference_method = inference_method.lower()
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.compute_inference = compute_inference
        self.solver = solver.lower()
        self.cpu_solver = cpu_solver.lower()
        self.lipschitz_L = lipschitz_L
        self.admm_rho = admm_rho
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

        # Internal storage for inference
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._bse = None
        self._tvalues = None
        self._pvalues = None
        self._conf_int = None

    def fit(self, X, y, sample_weight=None):
        """Fit Lasso regression model using coordinate descent."""
        # Avoid GPU->CPU copies when we don't need diagnostics/inference.
        # - CPU path always needs y on host for residuals/RSS.
        # - GPU path with compute_inference=False: residuals/scale are not computed,
        #   and properties depending on them return None.
        device = self._get_compute_device()
        if device != Device.CUDA:
            self._y = np.asarray(y)
        else:
            # If we compute inference fully on GPU (gpu_naive_ols), avoid copying y.
            if self.compute_inference and self.inference_method == "gpu_naive_ols":
                self._y = None
            else:
                # y may already be a CuPy array; use safe conversion.
                self._y = self._to_numpy(y)

        X_arr = self._to_array(X)
        y_arr = self._to_array(y)

        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        if self.compute_inference and self.inference_method != "gpu_naive_ols":
            self._compute_inference()
        self._fitted = True
        return self

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator: S(x, gamma) = sign(x) * max(|x| - gamma, 0)."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU (coordinate descent or FISTA)."""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self._nobs = n_samples

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

        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        Xty = X_centered.T @ y_centered.flatten()
        XtX = X_centered.T @ X_centered

        coef = np.zeros(n_features)

        if self.cpu_solver in ("fista",):
            # Proximal gradient / FISTA for L1-regularized least squares:
            #   minimize (1/(2n)) * ||y - Xw||^2 + alpha * ||w||_1
            # Uses the same stopping criterion as coordinate descent in this codebase:
            #   sum(abs(coef - coef_old)) < tol

            if self.lipschitz_L is not None:
                L = float(self.lipschitz_L)
            else:
                L_frob = float(np.sum(X_centered**2) / n_samples)
                try:
                    eigvals = np.linalg.eigvalsh(XtX)
                    L = float(eigvals[-1] / n_samples)
                except Exception:
                    L = L_frob

            if L <= 0:
                coef = np.zeros(n_features)
                self.n_iter_ = 0
            else:
                step = 1.0 / L
                thresh = self.alpha * step

                # FISTA variables
                y_k = coef.copy()
                t_k = 1.0

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    # grad = (XtX @ y_k - Xty) / n
                    grad = (XtX @ y_k - Xty) / n_samples

                    coef = self._soft_threshold(y_k - step * grad, thresh)

                    # Momentum update
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_k**2))) / 2.0
                    beta = (t_k - 1.0) / t_new
                    y_k = coef + beta * (coef - coef_old)
                    t_k = t_new

                    if self.stopping == "kkt":
                        # KKT violation for Lasso:
                        # grad_sse = (XtX @ w - Xty) / n
                        # optimality: |grad_sse_j| <= alpha when w_j == 0
                        # violation measure: max_j max(|grad_sse_j| - alpha, 0)
                        grad_sse = (XtX @ coef - Xty) / n_samples
                        violation = np.max(np.maximum(np.abs(grad_sse) - self.alpha, 0.0))
                        if violation < self.tol:
                            self.n_iter_ = iteration + 1
                            break
                    else:
                        # Legacy stopping: coefficient delta
                        if np.sum(np.abs(coef - coef_old)) < self.tol:
                            self.n_iter_ = iteration + 1
                            break
                else:
                    self.n_iter_ = self.max_iter

        else:
            # Coordinate descent (legacy CPU path)
            # Precompute squared norms for each feature
            X_sq_norms = np.diag(XtX)

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                for j in range(n_features):
                    # Compute partial residual
                    rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]

                    # Update coefficient with soft thresholding
                    if X_sq_norms[j] > 1e-10:
                        coef[j] = self._soft_threshold(rho_j, self.alpha * n_samples) / X_sq_norms[j]
                    else:
                        coef[j] = 0.0

                # Check convergence
                if self.stopping == "kkt":
                    grad_sse = (XtX @ coef - Xty) / n_samples
                    violation = np.max(np.maximum(np.abs(grad_sse) - self.alpha, 0.0))
                    if violation < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
            self.coef_ = coef
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            self._params = coef.copy()
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        if self.compute_inference:
            if self.fit_intercept:
                self._X_design = np.column_stack(
                    [np.ones(n_samples, dtype=X.dtype), X]
                )
            else:
                self._X_design = X.copy()

            y_pred = self._X_design @ self._params
            self._resid = self._y - y_pred

            if self._df_resid > 0:
                self._scale = np.sum(self._resid ** 2) / self._df_resid
            else:
                self._scale = np.nan
        else:
            self._X_design = None
            self._resid = None
            self._scale = np.nan

    def _soft_threshold_cupy(self, x, gamma):
        """Soft thresholding operator for CuPy arrays."""
        import cupy as cp
        return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0)

    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU solver."""
        import cupy as cp
        from .._gpu_utils import compute_r2_gpu

        if self.solver not in ("fista", "admm"):
            raise ValueError("solver must be one of: 'fista', 'admm'")

        if self.solver == "admm":
            return self._fit_gpu_admm(X, y, sample_weight=sample_weight)

        # Default: FISTA
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Ensure CuPy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)

        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw

        # Ensure vector y on GPU
        y = y.reshape(-1)

        # Center X/y when fitting intercept to match sklearn Lasso convention.
        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        # Precompute XtX / Xty for FISTA gradient: grad(w) = (XtX @ w - Xty) / n
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Lipschitz constant L for grad(w): L = lambda_max(XtX) / n
        # If user provides lipschitz_L, trust it (should be safe for convergence).
        if self.lipschitz_L is not None:
            L = cp.array(float(self.lipschitz_L), dtype=X.dtype)
        else:
            L_frob = cp.sum(X_centered ** 2) / n_samples
            try:
                eigvals = cp.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L = L_frob

        if L <= 0:
            # Degenerate case: return all-zero coefficients
            coef = cp.zeros(n_features, dtype=X.dtype)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            thresh = self.alpha * step

            # FISTA variables
            coef = cp.zeros(n_features, dtype=X.dtype)  # w_k
            y_k = coef.copy()  # y_k
            t_k = cp.array(1.0, dtype=X.dtype)

            for iteration in range(self.max_iter):
                coef_old = coef

                # Gradient at y_k: (1/n) XtX @ y_k - (1/n) Xty
                grad = (XtX @ y_k - Xty) / n_samples

                # Prox step for L1
                coef = self._soft_threshold_cupy(y_k - step * grad, thresh)

                # Momentum update
                t_new = (1 + cp.sqrt(1 + 4 * (t_k ** 2))) / 2
                beta = (t_k - 1) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                # Convergence test
                if self.stopping == "kkt":
                    grad_sse = (XtX @ coef - Xty) / n_samples
                    violation = cp.max(cp.maximum(cp.abs(grad_sse) - self.alpha, 0.0))
                    if float(violation.get()) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    # Legacy stopping: coefficient delta (fast but not guaranteed objective optimality)
                    if cp.sum(cp.abs(coef - coef_old)) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Build full coefficients and (optionally) residuals for inference/R^2
        if self.fit_intercept:
            intercept_gpu = y_mean - X_mean @ coef
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef])
        else:
            coef_full = coef

        # Always transfer coefficients; remaining transfers depend on compute_inference.
        coef_full_np = coef_full.get()

        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._df_resid = df_resid

        # Inference/diagnostics require residuals and design matrix.
        if self.compute_inference:
            # Only build the design matrix when we need residuals/inference.
            if self.fit_intercept:
                X_design = cp.concatenate(
                    [cp.ones((n_samples, 1), dtype=X.dtype), X], axis=1
                )
            else:
                X_design = X

            y_pred = X_design @ coef_full
            resid = y - y_pred

            if df_resid > 0:
                scale = cp.sum(resid ** 2) / df_resid
                self._scale = float(scale.get()) if not cp.isnan(scale) else np.nan
            else:
                self._scale = np.nan
                scale = cp.nan

            if self.inference_method == "gpu_naive_ols":
                # Compute inference fully on GPU, then transfer only small vectors.
                XtX = X_design.T @ X_design
                try:
                    XtX_inv = cp.linalg.inv(XtX)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX)

                bse_gpu = cp.sqrt(scale * cp.diag(XtX_inv))

                # Transfer inference vectors only
                bse_cpu = cp.asnumpy(bse_gpu)
                params_cpu = self._params  # already a CPU vector
                tvalues_cpu = params_cpu / (bse_cpu + 1e-30)
                pvalues_cpu = 2 * (1 - stats.t.cdf(np.abs(tvalues_cpu), df_resid))

                alpha = 0.05
                t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
                conf_int = np.column_stack([
                    params_cpu - t_crit * bse_cpu,
                    params_cpu + t_crit * bse_cpu,
                ])

                self._bse = bse_cpu
                self._tvalues = tvalues_cpu
                self._pvalues = pvalues_cpu
                self._conf_int = conf_int

                # R^2 / keep diagnostics consistent without transferring residuals.
                y_mean_gpu = cp.mean(y)
                ss_tot = cp.sum((y - y_mean_gpu) ** 2)
                ss_res = cp.sum(resid ** 2)
                self._rsquared_gpu = float(cp.asnumpy(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            else:
                # Default: transfer residuals and design to CPU.
                self._resid = resid.get()
                self._X_design = X_design.get()

        else:
            # Strict GPU mode: avoid large residual/host design transfers.
            self._scale = np.nan
            self._resid = None
            self._X_design = None
            # R^2 is optional; keep behavior as None when no residuals are available.
            self._rsquared_gpu = None

    def _fit_gpu_admm(self, X, y, sample_weight=None):
        """Fit using GPU with ADMM solver.

        Objective matches sklearn:
          (1/(2n)) * ||y - Xw||^2 + alpha * ||w||_1
        """
        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Ensure CuPy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)

        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw

        # Ensure vector y on GPU
        y = y.reshape(-1)

        # Center for intercept
        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        # ADMM variables for constraint w=z
        coef = cp.zeros(n_features, dtype=X.dtype)  # w
        z = cp.zeros(n_features, dtype=X.dtype)  # z
        u = cp.zeros(n_features, dtype=X.dtype)  # scaled dual

        # Precompute XtX and Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # w-update solves:
        # (XtX + rho*n*I) w = Xty + rho*n * (z - u)
        rho = float(self.admm_rho)
        if rho <= 0:
            raise ValueError("admm_rho must be > 0")

        lhs = XtX + (rho * n_samples) * cp.eye(n_features, dtype=X.dtype)

        # Pre-factorize once
        Lmat = cp.linalg.cholesky(lhs)

        def solve_w(rhs):
            # Solve Lmat @ (Lmat.T @ w) = rhs
            tmp = cpx_linalg.solve_triangular(Lmat, rhs, lower=True)
            return cpx_linalg.solve_triangular(Lmat.T, tmp, lower=False)

        thresh = self.alpha / rho

        for iteration in range(self.max_iter):
            coef_old = coef

            rhs = Xty + (rho * n_samples) * (z - u)
            coef = solve_w(rhs)

            # z-update (prox of l1)
            z_old = z
            z = self._soft_threshold_cupy(coef + u, thresh)

            # dual update
            u = u + (coef - z)

            # Convergence test
            if self.stopping == "kkt":
                grad_sse = (XtX @ coef - Xty) / n_samples
                violation = cp.max(cp.maximum(cp.abs(grad_sse) - self.alpha, 0.0))
                if float(violation.get()) < self.tol:
                    self.n_iter_ = iteration + 1
                    break
            else:
                # Legacy stopping: coefficient delta
                if cp.sum(cp.abs(coef - coef_old)) < self.tol:
                    self.n_iter_ = iteration + 1
                    break
            z = z  # keep for clarity
        else:
            self.n_iter_ = self.max_iter

        # Build full coefficients and (optionally) residuals for inference/R^2
        if self.fit_intercept:
            intercept_gpu = y_mean - X_mean @ coef
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef])
            X_design = cp.concatenate([cp.ones((n_samples, 1), dtype=X.dtype), X], axis=1)
        else:
            coef_full = coef
            X_design = X

        coef_full_np = coef_full.get()
        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._df_resid = df_resid

        if self.compute_inference:
            y_pred = X_design @ coef_full
            resid = y - y_pred
            if df_resid > 0:
                scale = cp.sum(resid ** 2) / df_resid
                self._scale = float(scale.get()) if not cp.isnan(scale) else np.nan
            else:
                self._scale = np.nan

            self._resid = resid.get()
            self._X_design = X_design.get()
        else:
            self._scale = np.nan
            self._resid = None
            self._X_design = None
            self._rsquared_gpu = None

    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        if self.inference_method == "bootstrap":
            return self._compute_inference_bootstrap()
        if self.inference_method == "gpu_naive_ols":
            # Inference already computed on GPU in _fit_gpu().
            return
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return

        X = self._X_design

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        self._bse = np.sqrt(self._scale * np.diag(XtX_inv))
        self._tvalues = self._params / self._bse
        self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))

        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, self._df_resid)
        self._conf_int = np.column_stack([
            self._params - t_crit * self._bse,
            self._params + t_crit * self._bse
        ])

    def _compute_inference_bootstrap(self) -> None:
        """
        Bootstrap inference for Lasso via residual resampling.

        Notes
        -----
        This is more robust than the naive OLS-based inference, but it is still
        not full "post-selection inference" for Lasso.
        """
        if self._X_design is None or self._resid is None or self._y is None:
            return

        if self.n_bootstrap <= 0:
            return

        rng = np.random.default_rng(self.bootstrap_random_state)
        X = self._X_design
        y = self._y
        y_pred = y - self._resid
        resid = self._resid

        params_dim = self._params.shape[0]
        boot_params = np.zeros((self.n_bootstrap, params_dim), dtype=float)

        # Precompute Lipschitz constant if needed for CPU FISTA.
        lipschitz_L = self.lipschitz_L
        if self.cpu_solver == "fista" and lipschitz_L is None:
            # L = lambda_max(Xc^T Xc) / n for centered design
            X_nopen = X[:, 1:] if self.fit_intercept else X
            X_centered = X_nopen - X_nopen.mean(axis=0, keepdims=True)
            XtX = X_centered.T @ X_centered
            eigvals = np.linalg.eigvalsh(XtX)
            lipschitz_L = float(eigvals[-1] / X_nopen.shape[0])

        for b in range(self.n_bootstrap):
            eps_star = rng.choice(resid, size=resid.shape[0], replace=True)
            y_star = y_pred + eps_star

            refit = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                stopping=self.stopping,
                inference_method="naive_ols",
                n_bootstrap=0,
                bootstrap_random_state=None,
                device="cpu",
                compute_inference=False,
                solver=self.solver,
                cpu_solver=self.cpu_solver,
                lipschitz_L=lipschitz_L,
                admm_rho=self.admm_rho,
            )

            # Refit expects raw X (without intercept column).
            if self.fit_intercept:
                X_refit = X[:, 1:]
            else:
                X_refit = X

            refit.fit(X_refit, y_star)
            boot_params[b, :] = refit._params

        # Standard errors and bootstrap-based p-values/CI.
        self._bse = np.std(boot_params, axis=0, ddof=1)
        self._params = np.asarray(self._params, dtype=float)

        # Two-sided p-values using sign-change probability.
        pvalues = np.zeros(params_dim, dtype=float)
        for i in range(params_dim):
            coef_b = boot_params[:, i]
            p_lower = np.mean(coef_b <= 0.0)
            p_upper = np.mean(coef_b >= 0.0)
            p = 2.0 * min(p_lower, p_upper)
            pvalues[i] = min(p, 1.0)
        self._pvalues = pvalues

        # Percentile confidence intervals.
        lower_q = (0.05 / 2.0) * 1.0
        upper_q = 1.0 - (0.05 / 2.0) * 1.0
        self._conf_int = np.column_stack([
            np.quantile(boot_params, lower_q, axis=0),
            np.quantile(boot_params, upper_q, axis=0),
        ])

        # t-stats (approx) from bootstrap SE.
        self._tvalues = self._params / (self._bse + 1e-30)

    @property
    def rsquared(self):
        """R-squared."""
        if self._resid is None:
            # In compute_inference=False GPU mode we may avoid transferring residuals.
            if hasattr(self, "_rsquared_gpu") and self._rsquared_gpu is not None:
                return self._rsquared_gpu
            return None
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @property
    def rsquared_adj(self):
        """Adjusted R-squared."""
        if self._nobs is None:
            return None
        r2 = self.rsquared
        if r2 is None:
            return None
        k = len(self.coef_)
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid

    @property
    def fvalue(self):
        """F-statistic."""
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        ss_reg = ss_tot - ss_res
        k = len(self.coef_)
        if k == 0 or ss_res <= 0:
            return np.inf
        return (ss_reg / k) / (ss_res / self._df_resid)

    @property
    def f_pvalue(self):
        """p-value for F-statistic."""
        fv = self.fvalue
        if fv is None or fv == np.inf:
            return 1.0
        k = len(self.coef_)
        return 1 - stats.f.cdf(fv, k, self._df_resid)

    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        return -2 * self.llf + 2 * len(self._params)

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        n = self._nobs
        k = len(self._params)
        return -2 * self.llf + k * np.log(n)

    @property
    def llf(self):
        """Log-likelihood."""
        if self._nobs is None or self._resid is None:
            return None
        n = self._nobs
        sigma2_mle = np.sum(self._resid ** 2) / n
        return -n/2 * np.log(2 * np.pi * sigma2_mle) - n/2

    def summary(self):
        """Print summary table."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if self._bse is None or self._pvalues is None or self._conf_int is None:
            raise RuntimeError(
                "compute_inference=False: inference statistics are not available. "
                "Re-fit with compute_inference=True (default) to use summary()."
            )

        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]

        print("=" * 80)
        print("                            Lasso Regression Results")
        print(f"                            (alpha = {self.alpha:.4f})")
        print("=" * 80)
        print(f"No. Observations:           {self._nobs:>15}")
        print(f"Degrees of Freedom:         {self._df_resid:>15}")
        print(f"Iterations:                 {self.n_iter_:>15}")
        print(f"R-squared:                  {self.rsquared:>15.4f}")
        print(f"Adj. R-squared:             {self.rsquared_adj:>15.4f}")
        print(f"F-statistic:                {self.fvalue:>15.4f}")
        print(f"Prob (F-statistic):         {self.f_pvalue:>15.4e}")
        print(f"Log-Likelihood:             {self.llf:>15.4f}")
        print(f"AIC:                        {self.aic:>15.4f}")
        print(f"BIC:                        {self.bic:>15.4f}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {'t':>10} {'P>|t|':>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)

        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._tvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")

        print("=" * 80)

    def predict(self, X):
        """Predict using the Lasso model."""
        self._check_is_fitted()
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return R^2 score."""
        y_pred = self.predict(X)
        y = np.asarray(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0