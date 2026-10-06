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


def _validate_lassocv_selection_details(details):
    """Require complete finite fold evidence for genuine multi-alpha selection.

    The shared private selector has other internal consumers, so the stricter
    selection policy belongs at the public ``LassoCV`` boundary. Candidates with
    a non-finite validation MSE on any executed fold are ineligible. Degenerate
    paths that do not perform genuine multi-alpha CV keep their historical
    single-refit semantics. The selector already represents those degenerate
    cases with fewer than two evaluated MSE columns, so this guard does not need
    to inspect the caller's input container or sample count.
    """
    if not isinstance(details, dict):
        raise RuntimeError("LassoCV selector must return structured details")

    alphas = np.asarray(details.get("alphas", ()), dtype=np.float64).reshape(-1)
    mse_path = np.asarray(details.get("mse_path", ()), dtype=np.float64)
    if mse_path.ndim != 2 or mse_path.shape[0] != alphas.size:
        raise RuntimeError("LassoCV returned an inconsistent validation-MSE layout")

    n_folds_evaluated = int(mse_path.shape[1])
    if alphas.size <= 1 or n_folds_evaluated < 2:
        return details

    complete = np.all(np.isfinite(mse_path), axis=1)
    if not np.any(complete):
        raise FloatingPointError(
            "LassoCV produced no candidate with finite validation MSE on every "
            "fold; refusing to select alpha"
        )

    complete_rows = mse_path[complete]
    if np.any(complete_rows < 0.0):
        raise FloatingPointError(
            "LassoCV produced a negative validation MSE; refusing to select alpha"
        )

    # MSE is non-negative, so divide before summing to avoid an intermediate
    # float64 overflow when every fold score is finite but close to DBL_MAX.
    candidate_means = np.sum(
        complete_rows / float(n_folds_evaluated),
        axis=1,
    )
    finite_mean = np.isfinite(candidate_means)
    if not np.any(finite_mean):
        raise FloatingPointError(
            "LassoCV produced no candidate with a finite aggregate validation MSE"
        )

    complete_indices = np.flatnonzero(complete)
    eligible_indices = complete_indices[finite_mean]
    eligible_means = candidate_means[finite_mean]
    mean_mse = np.full(alphas.size, np.nan, dtype=np.float64)
    mean_mse[eligible_indices] = eligible_means
    best_local = int(np.argmin(eligible_means))
    best_index = int(eligible_indices[best_local])

    checked = dict(details)
    checked["alpha"] = float(alphas[best_index])
    checked["mean_mse"] = mean_mse
    return checked


# =============================================================================
# LassoCV Class
# =============================================================================

class LassoCV(CVEstimatorBase):
    """Cross-validated Lasso regression for one continuous response.

    Select alpha by mean held-out MSE, then refit all supplied training rows.
    Final inference conditions on selected alpha and does not adjust tuning
    uncertainty. Generated folds are shuffled K-fold, not time/group-aware.

    Parameters
    ----------
    alphas : array-like or None, default=None
        Explicit positive finite candidates; omitted: generate a data-
        dependent grid. Invalid/nonpositive entries are filtered; an empty
        surviving grid falls back to automatic generation.
    n_alphas : int, default=12
        Automatic grid size when alphas is omitted.
    alpha_min_ratio : float, default=0.001
        Minimum/maximum ratio for the automatic grid; choose a positive
        value normally no greater than 1.
    cv : int, default=5
        Generated shuffled K-fold count, at least 2.
    cv_splits : list or None, default=None
        Explicit reusable list of (train_indices, validation_indices);
        validate nonempty disjoint integer subsets yourself.
    fit_intercept : bool, default=True
        Fit an intercept in CV and final refit.
    device : str, default='auto'
        cpu, cuda (CuPy), torch (Torch CUDA), auto; explicit unavailable GPU
        requests raise.
    n_jobs : int or None, default=None
        Shared worker configuration; no candidate-parallelism guarantee.
    compute_inference : bool, default=False
        Compute supported inference only on the final full-data refit,
        conditional on selected alpha.
    max_iter : int, default=3000
        Iteration budget for CV solves and the final direct fit.
    tol : float, default=0.0001
        Convergence tolerance for CV solves and final fitting.
    stopping : str, default='coef_delta'
        Final-refit request only; current direct Gaussian stopping does not
        honor the kkt setting. It does not select the CV path stopping
        check.
    solver : str, default='fista'
        Final full-data Lasso solver; does not choose the CV solver.
    cpu_solver : str or None, default=None
        Deprecated CPU CV alias; use cv_solver. Conflicting explicit CPU
        requests raise. On GPU it warns but does not replace FISTA.
    method : str, default='standard'
        standard or glmnet CV path mode. On CPU, glmnet requires coordinate
        descent; GPU CV remains FISTA. Not an inference method.
    cd_kkt_check_every : int or None, default=None
        Positive integer CPU CV coordinate-descent KKT-check interval. None
        resolves to 1 for standard or 4 for glmnet. Not a final-refit
        certificate.
    inference_method : str, default='post_selection_ols'
        Final-refit post_selection_ols, debiased, bootstrap or supported
        auto request; see Lasso inference restrictions. Deprecated aliases
        normalize at the compatibility boundary.
    lipschitz_L : float or None, default=None
        Optional compatible final-refit Lipschitz bound; not a CV-path
        control.
    admm_rho : float, default=1.0
        Forwarded to final Lasso; currently ignored by unified ADMM, which
        starts at rho=1.0.
    gpu_memory_cleanup : bool, default=False
        Request best-effort GPU cache cleanup for final fitting.
    random_state : int or None, default=None
        Seed for generated folds; not residual-bootstrap randomness.
    gpu_cv_mixed_precision : bool, default=True
        Enable mixed precision during GPU CV; final-refit inference has its
        own numerical path.
    cv_solver : str, default='auto'
        auto, coordinate_descent or fista for CV. auto selects CPU CD or GPU
        FISTA; explicit coordinate_descent is CPU-only.
    nodewise_alpha : float or None, default=None
        Keyword-only final-refit debiased precision tuning; does not change
        the grid, fold losses or selected alpha.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit finite X (n, p), one-dimensional y (n,), optional analytic weights;
        return self. There is no formula interface.
    predict(X)
        Return NumPy predictions of shape (m,) through the final estimator.
    score(X, y)
        Return unweighted evaluation R-squared; no sample_weight argument.
    summary()
        Print final inference and return None; requires successful inference.
    get_params(deep=True), set_params(**params)
        Read/update constructor configuration; valid updates require refitting.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Inherited helpers; they do not automatically repeat CV or model refits.

    Attributes
    ----------
    alpha_ : float
        Selected penalty, minimizing mean validation MSE.
    alphas_, mean_mse_ : numpy.ndarray of shape (n_alphas,)
        Actual candidates and their mean validation losses.
    cv_results_ : dict
        Only mse_path, of shape (n_alphas, n_folds); no mean_mse dictionary key.
    best_score_ : float
        Negative selected mean MSE (larger is better), not final-model R-squared.
    coef_ : numpy.ndarray of shape (n_features,)
        Final full-data prediction slopes.
    intercept_ : float
        Final intercept, or zero when omitted.
    n_iter_ : int
        Final estimator iteration count.
    estimator_ : object
        Fitted direct estimator; read its supported inference outputs here.
    mse_path_ : numpy.ndarray
        The same (n_alphas, n_folds) array stored in cv_results_["mse_path"].
    cv_solver_ : str
        Resolved CV algorithm; solver instead controls the final Lasso fit.
    nodewise_alpha_ : float or None
        Resolved final-refit debiased precision tuning, where applicable.

    Notes
    -----
    nodewise_alpha is installed as a keyword-only runtime constructor control.
    Direct Lasso simultaneous/bootstrap-draw controls are not CV constructor
    arguments. Validate custom split indices yourself; acceptance does not
    establish that training and validation are disjoint or free of duplicates.
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
        for attr in (
            "_params",
            "_bse",
            "_pvalues",
            "_tvalues",
            "_zvalues",
            "_conf_int",
            "_inference_result",
            "_conf_int_simultaneous",
            "_simultaneous_enabled",
            "_simultaneous_method",
            "_simultaneous_alpha",
            "_simultaneous_n_bootstrap",
            "_simultaneous_critical_value",
            "_simultaneous_target_mask",
        ):
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
        details = _validate_lassocv_selection_details(details)

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

        inference_result = getattr(estimator, "_inference_result", None)
        if inference_result is not None:
            inference_result.apply_to(self)

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted Lasso model."""
        self._check_is_fitted()
        return self.estimator_.predict(X)