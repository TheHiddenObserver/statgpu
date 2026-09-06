"""
LassoCV: Cross-validated Lasso regression with GPU support.

This module exports LassoCV which delegates to _select_lasso_alpha_cv
from _lasso.py for all CV logic (cache, fast-refit, backend-aware).
"""

__all__ = ["LassoCV"]

from typing import Optional, Union
import warnings

import numpy as np

from statgpu._config import Device
from statgpu.cross_validation._base import CVEstimatorBase
from statgpu.linear_model.wrappers._lasso import (
    Lasso,
    _normalize_lassocv_method,
    _normalize_cd_kkt_check_every,
)


# Shared hash function from _cv_base.py
from statgpu.cross_validation._base import hash_cv_data as _hash_data


# =============================================================================
# LassoCV Class
# =============================================================================

class LassoCV(CVEstimatorBase):
    """
    Cross-validated Lasso regression with GPU support.

    This class implements K-fold cross-validation to select the optimal
    regularization parameter alpha for Lasso regression.

    ``solver`` controls the final full-data refit. ``cv_solver`` controls the
    cross-validation path. The deprecated ``cpu_solver`` parameter is retained
    temporarily as a compatibility alias for CPU CV behavior.

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
        Whether to fit intercept. Default is True.
    device : str or Device
        Computation device: 'cpu', 'cuda', 'torch', or 'auto'.
    max_iter : int
        Maximum iterations for Lasso solver. Default is 3000.
    tol : float
        Convergence tolerance. Default is 1e-4.
    solver : str, default='fista'
        Solver for the final full-data Lasso refit.
    cv_solver : {'auto', 'coordinate_descent', 'fista'}, default='auto'
        Solver for the CV folds/path. ``auto`` chooses coordinate descent on
        CPU and FISTA on CUDA/Torch. ``method='glmnet'`` forces coordinate
        descent only on the CPU path; GPU paths retain backend-native FISTA.
    cpu_solver : str or None, deprecated
        Deprecated compatibility control for the historical CPU CV solver.
        On CPU it is treated as the legacy alias for ``cv_solver``. On
        CUDA/Torch it warns but remains non-authoritative, preserving the old
        behavior where this CPU-only control did not change the GPU CV solver.
    compute_inference : bool
        Whether to compute inference on the final refit.
    inference_method : str, default='post_selection_ols'
        Final-refit inference method. ``post_selection_ols`` is the canonical
        hardware-neutral active-set OLS/WLS diagnostic. Older
        ``cpu_ols_inference``/``gpu_ols_inference`` spellings remain accepted at
        this CV compatibility boundary and normalize to the same method.
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
        Best score (negative MSE; higher is better).
    coef_ : ndarray
        Coefficients of the final model.
    intercept_ : float
        Intercept of the final model.
    estimator_ : Lasso
        The fitted Lasso estimator with selected alpha.
    cv_solver_ : str
        Actual solver used by the CV path after device/method resolution.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import LassoCV
    >>> X = np.random.randn(1000, 20)
    >>> y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)
    >>> model = LassoCV(cv=5, device='cpu')
    >>> model.fit(X, y)
    >>> print(f"Selected alpha: {model.alpha_:.4f}")
    >>> print(f"CV solver: {model.cv_solver_}")
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
        cpu_solver: Optional[str] = None,
        method: str = "standard",
        cd_kkt_check_every: Optional[int] = None,
        inference_method: str = "post_selection_ols",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
        gpu_cv_mixed_precision: bool = True,
        cv_solver: str = "auto",
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
        self.cpu_solver = cpu_solver
        self.cv_solver = str(cv_solver)
        self.method = _normalize_lassocv_method(method)
        self.cd_kkt_check_every = _normalize_cd_kkt_check_every(cd_kkt_check_every)
        # Preserve the public spelling for get_params()/clone. The installed
        # compatibility layer owns the private runtime normalization.
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
        self.cv_solver_ = None

    def _reset_cv_fit_state(self):
        """Clear all fitted outputs before a new CV attempt."""
        self._fitted = False
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
        self.cv_solver_ = None
        for attr in ("_bse", "_pvalues", "_tvalues", "_conf_int"):
            self.__dict__.pop(attr, None)

    def _prepare_cv_inputs_for_resolved_device(
        self, X, y, sample_weight, device_name: str
    ):
        """Preserve the concrete resolved backend before entering the CV helper.

        ``device='auto'`` may resolve through the global device configuration.
        Once ``_get_compute_device()`` has produced a concrete CUDA/Torch device,
        the CV selector must not reinterpret that decision through its legacy
        auto-backend inference.
        """
        resolved = str(device_name).strip().lower()
        if resolved == Device.CUDA.value:
            target_device = Device.CUDA
            backend_name = "cupy"
        elif resolved == Device.TORCH.value:
            target_device = Device.TORCH
            backend_name = "torch"
        else:
            return X, y, sample_weight

        # Explicit conversion is itself the availability gate: Torch conversion
        # requires CUDA, while CuPy conversion raises when the requested CuPy/CUDA
        # backend is unavailable. This applies equally to constructor-explicit and
        # globally resolved device choices.
        X_cv = self._to_array(X, target_device, backend=backend_name)
        y_cv = self._to_array(y, target_device, backend=backend_name)
        sample_weight_cv = (
            None
            if sample_weight is None
            else self._to_array(
                sample_weight,
                target_device,
                backend=backend_name,
            )
        )
        return X_cv, y_cv, sample_weight_cv

    def _resolve_cv_solver(self, device_name: str) -> str:
        """Resolve the actual CV solver without changing legacy GPU behavior."""
        requested = str(self.cv_solver).strip().lower()
        allowed = {"auto", "coordinate_descent", "fista"}
        if requested not in allowed:
            raise ValueError(
                "cv_solver must be one of 'auto', 'coordinate_descent', or 'fista'"
            )

        device_name = str(device_name).strip().lower()
        legacy = None
        if self.cpu_solver is not None:
            legacy = str(self.cpu_solver).strip().lower()
            if legacy not in {"coordinate_descent", "fista"}:
                raise ValueError(
                    "deprecated cpu_solver must be 'coordinate_descent' or 'fista'"
                )
            # #138 adds one transparent fit wrapper around this call. Keep the
            # #135 caller-facing warning location in both direct and fit routes.
            fit_wrapper_active = int(
                getattr(self, "_statgpu_post_selection_fit_device_depth", 0)
            ) > 0
            warnings.warn(
                "LassoCV(cpu_solver=...) is deprecated; use cv_solver=... for "
                "the cross-validation path. solver=... controls only the final "
                "full-data refit. On CUDA/Torch, historical cpu_solver values "
                "remain non-authoritative and do not replace GPU FISTA. "
                "cpu_solver will be removed in a future breaking release.",
                FutureWarning,
                stacklevel=5 if fit_wrapper_active else 2,
            )
            if (
                device_name == "cpu"
                and requested != "auto"
                and requested != legacy
            ):
                raise ValueError(
                    "cv_solver and deprecated cpu_solver specify different CV solvers"
                )

        method = str(self._method).strip().lower()

        # The maintained GPU CV engine is FISTA-based. Historically cpu_solver
        # was a CPU-only control and did not change that GPU path, including in
        # method='glmnet' mode. Preserve that behavior for the deprecated alias
        # and make cv_solver_ report the algorithm that actually executes.
        if device_name != "cpu":
            if requested == "coordinate_descent":
                raise ValueError(
                    "cv_solver='coordinate_descent' is CPU-only; use "
                    "cv_solver='auto' or cv_solver='fista' for CUDA/Torch CV"
                )
            return "fista" if requested == "auto" else requested

        # CPU glmnet mode historically forced coordinate descent regardless of
        # cpu_solver. Preserve that result for the deprecated alias, while a
        # conflicting value supplied through the new cv_solver API is explicit.
        if method == "glmnet":
            if requested not in {"auto", "coordinate_descent"}:
                raise ValueError(
                    "method='glmnet' requires cv_solver='coordinate_descent' or 'auto' on CPU"
                )
            return "coordinate_descent"

        if requested == "auto" and legacy is not None:
            requested = legacy
        if requested == "auto":
            return "coordinate_descent"
        return requested

    def fit(self, X, y, sample_weight=None):
        """Fit Lasso regression with cross-validation to select alpha."""
        from statgpu.linear_model.wrappers._lasso import _select_lasso_alpha_cv, Lasso

        self._reset_cv_fit_state()
        device_name = self._get_compute_device().value
        effective_cv_solver = self._resolve_cv_solver(device_name)
        X_cv, y_cv, sample_weight_cv = self._prepare_cv_inputs_for_resolved_device(
            X,
            y,
            sample_weight,
            device_name,
        )

        effective_cd_kkt = self._cd_kkt_check_every
        if effective_cd_kkt is None:
            effective_cd_kkt = 4 if str(self._method).lower() == "glmnet" else 1

        details = _select_lasso_alpha_cv(
            X_cv, y_cv,
            alphas=self.alphas,
            n_alphas=self._n_alphas,
            alpha_min_ratio=self._alpha_min_ratio,
            cv_folds=self._cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight_cv,
            fit_intercept=self._fit_intercept,
            device=device_name,
            max_iter=self._max_iter,
            tol=self._tol,
            cpu_solver=effective_cv_solver,
            method=self._method,
            cd_kkt_check_every=effective_cd_kkt,
            gpu_cv_mixed_precision=self._gpu_cv_mixed_precision,
            return_details=True,
        )

        # Keep candidate CV results local until the final full-data refit
        # succeeds, matching the failure-safe contract of the other CV classes.
        selected_alpha = float(details["alpha"])
        selected_alphas = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)
        best_score = (
            -float(np.nanmin(mean_mse))
            if np.any(np.isfinite(mean_mse))
            else np.nan
        )

        estimator = Lasso(
            alpha=selected_alpha,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol,
            stopping=self._stopping,
            inference_method=self._inference_method,
            device=self._device,
            n_jobs=self.n_jobs,
            compute_inference=self._compute_inference_enabled,
            solver=self._solver,
            lipschitz_L=self.lipschitz_L,
            admm_rho=self._admm_rho,
            gpu_memory_cleanup=self._gpu_memory_cleanup,
        )
        estimator.fit(X, y, sample_weight=sample_weight)

        self.alpha_ = selected_alpha
        self.alphas_ = selected_alphas
        self.cv_results_ = {"mse_path": mse_path}
        self.mse_path_ = mse_path
        self.mean_mse_ = mean_mse
        self.best_score_ = best_score
        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)
        self.cv_solver_ = effective_cv_solver

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
