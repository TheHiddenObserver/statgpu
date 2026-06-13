"""
LassoCV: Cross-validated Lasso regression with GPU support.

This module exports LassoCV which delegates to _select_lasso_alpha_cv
from _lasso.py for all CV logic (cache, fast-refit, backend-aware).
"""

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.linear_model._cv_base import CVEstimatorBase
from statgpu.linear_model._lasso import (
    Lasso,
    _normalize_lassocv_method,
    _normalize_cd_kkt_check_every,
)


# Shared hash function from _cv_base.py
from statgpu.linear_model._cv_base import hash_cv_data as _hash_data


# =============================================================================
# LassoCV Class
# =============================================================================

class LassoCV(CVEstimatorBase):
    """
    Cross-validated Lasso regression with GPU support.

    This class implements K-fold cross-validation to select the optimal
    regularization parameter alpha for Lasso regression.

    Parameters
    ----------
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None). Default is 12.
    alpha_min_ratio : float
        Minimum alpha as a ratio of max alpha.
    cv : int
        Number of CV folds. Default is 5.
    fit_intercept : bool
        Whether to fit intercept. Default is False.
    device : str or Device
        Computation device: 'cpu', 'cuda', or 'auto'.
    max_iter : int
        Maximum iterations for Lasso solver. Default is 3000.
    tol : float
        Convergence tolerance. Default is 1e-4.
    compute_inference : bool
        Whether to compute standard errors, t-stats, p-values and CI.
    random_state : int or None
        Random seed for CV splits.
    gpu_cv_mixed_precision : bool
        Whether to use mixed precision on GPU.

    Attributes
    ----------
    alpha_ : float
        Selected alpha value.
    alphas_ : ndarray
        All alpha values tested.
    cv_results_ : dict
        CV results including mse_path and mean_mse.
    best_score_ : float
        Best (minimum) MSE across CV folds.
    coef_ : ndarray
        Coefficients of the final model.
    intercept_ : float
        Intercept of the final model.
    estimator_ : Lasso
        The fitted Lasso estimator with selected alpha.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import LassoCV
    >>> X = np.random.randn(1000, 20)
    >>> y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)
    >>> model = LassoCV(cv=5, device='cuda')
    >>> model.fit(X, y)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"Best CV score: {model.best_score_:.4f}")
    """

    def __init__(
        self,
        alphas=None,
        n_alphas: int = 12,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = False,
        max_iter: int = 3000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        method: str = "standard",
        cd_kkt_check_every: Optional[int] = None,
        inference_method: str = "cpu_ols_inference",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
        gpu_cv_mixed_precision: bool = True,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.alphas = alphas
        self.n_alphas = int(n_alphas)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.compute_inference = bool(compute_inference)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.stopping = str(stopping)
        self.solver = str(solver)
        self.cpu_solver = str(cpu_solver)
        self.method = _normalize_lassocv_method(method)
        self.cd_kkt_check_every = _normalize_cd_kkt_check_every(cd_kkt_check_every)
        self.inference_method = str(inference_method)
        self.lipschitz_L = lipschitz_L
        self.admm_rho = float(admm_rho)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.gpu_cv_mixed_precision = bool(gpu_cv_mixed_precision)

        self.alpha_ = None
        self.alphas_ = None
        self.cv_results_ = None
        self.mse_path_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit Lasso regression with cross-validation to select alpha.

        Delegates to ``_select_lasso_alpha_cv`` for CV with cache, fast-refit,
        and backend-aware optimizations.

        Parameters
        ----------
        X : array-like
            Training data (n_samples, n_features).
        y : array-like
            Target values.
        sample_weight : array-like or None
            Sample weights.

        Returns
        -------
        self : LassoCV
            Fitted estimator.
        """
        from statgpu.linear_model._lasso import _select_lasso_alpha_cv, Lasso

        device_name = self._get_compute_device().value
        effective_cpu_solver = (
            "coordinate_descent" if str(self.method).lower() == "glmnet" else str(self.cpu_solver)
        )
        effective_cd_kkt = self.cd_kkt_check_every
        if effective_cd_kkt is None:
            effective_cd_kkt = 4 if str(self.method).lower() == "glmnet" else 1

        details = _select_lasso_alpha_cv(
            X, y,
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self.fit_intercept,
            device=device_name,
            max_iter=self.max_iter,
            tol=self.tol,
            cpu_solver=effective_cpu_solver,
            method=self.method,
            cd_kkt_check_every=effective_cd_kkt,
            gpu_cv_mixed_precision=self.gpu_cv_mixed_precision,
            return_details=True,
        )

        # Store CV results
        self.alpha_ = float(details["alpha"])
        self.alphas_ = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)

        self.cv_results_ = {"mse_path": mse_path}
        self.mse_path_ = mse_path
        self.mean_mse_ = mean_mse
        # sklearn convention: best_score_ is negative MSE (higher is better)
        self.best_score_ = -float(np.nanmin(mean_mse)) if np.any(np.isfinite(mean_mse)) else np.nan

        # Fit final model with selected alpha
        estimator = Lasso(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            stopping=self.stopping,
            inference_method=self.inference_method,
            device=self.device,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            solver=self.solver,
            cpu_solver=effective_cpu_solver,
            lipschitz_L=self.lipschitz_L,
            admm_rho=self.admm_rho,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )
        estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)

        # Copy inference attributes if available (preserve underscore prefix)
        for attr in ('_bse', '_pvalues', '_tvalues', '_conf_int'):
            val = getattr(estimator, attr, None)
            if val is not None:
                setattr(self, attr, np.asarray(val))

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted Lasso model."""
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def score(self, X, y):
        """Return R² score."""
        self._check_is_fitted()
        return self.estimator_.score(X, y)

    def summary(self):
        """Return summary of the fitted model."""
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted estimator available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(f"{self.estimator_.__class__.__name__} does not implement summary().")
        return self.estimator_.summary()
