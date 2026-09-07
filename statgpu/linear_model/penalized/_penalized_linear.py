"""PenalizedLinearRegression — thin wrapper over PenalizedGeneralizedLinearModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy import stats

from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

if TYPE_CHECKING:
    from statgpu.penalties._base import Penalty


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
        penalty_kwargs: Optional[dict] = None,
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
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
        loss_kwargs: Optional[dict] = None,
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
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
            loss_kwargs=loss_kwargs,
            inference_method=inference_method,
        )

    @staticmethod
    def _weighted_sparse_gpu_working_data(
        X,
        y,
        sample_weight,
        *,
        fit_intercept,
        xp,
        n_eff,
    ):
        """Map analytic-weight Gaussian data to an equivalent unweighted problem.

        The CPU sparse Gaussian path minimizes the weighted squared loss divided
        by ``sum(sample_weight)`` after weighted centering.  The shared GPU FISTA
        kernel is unweighted and divides by ``n_samples``.  Scaling the
        weighted-centered rows by ``sqrt(n_samples / sum(weight))`` makes those
        two objectives identical while preserving the existing fused GPU solver.
        """
        n_samples = int(X.shape[0])
        if fit_intercept:
            X_mean = xp.sum(X * sample_weight[:, None], axis=0) / n_eff
            y_mean = xp.sum(y * sample_weight) / n_eff
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_mean = None
            y_mean = None
            X_centered = X
            y_centered = y

        row_scale = xp.sqrt(
            sample_weight * (float(n_samples) / float(n_eff))
        )
        return (
            X_centered * row_scale[:, None],
            y_centered * row_scale,
            X_mean,
            y_mean,
        )

    def _fit_gpu_backend(self, X, y, sample_weight=None, backend_name="cupy"):
        """Fit Gaussian sparse models with CPU-equivalent analytic weights.

        Weighted L1/ElasticNet fits use the same average-loss objective as the
        NumPy path.  Other solver/penalty combinations delegate unchanged to the
        maintained shared backend implementation.
        """
        penalty_name = str(
            getattr(self._penalty, "name", self.penalty)
        ).lower()
        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name=backend_name
        )
        if (
            sample_weight is None
            or penalty_name not in ("l1", "elasticnet", "en")
            or solver_name not in ("fista", "fista_bb")
        ):
            return super()._fit_gpu_backend(
                X,
                y,
                sample_weight,
                backend_name=backend_name,
            )

        from statgpu.backends import _to_numpy
        from statgpu.backends._utils import _get_xp, xp_asarray
        from statgpu.linear_model.penalized._fit_mixin import (
            _validate_sample_weight_backend,
        )

        xp = _get_xp(backend_name)
        X_arr = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
        y_arr = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=y).reshape(-1)
        sample_weight_arr = xp_asarray(
            sample_weight,
            dtype=X_arr.dtype,
            xp=xp,
            ref_arr=X_arr,
        ).reshape(-1)
        n_samples, n_features = X_arr.shape
        n_eff = _validate_sample_weight_backend(
            sample_weight_arr,
            n_samples,
            backend_name,
        )
        original_intercept = bool(self._effective_intercept)
        X_work, y_work, X_mean, y_mean = (
            self._weighted_sparse_gpu_working_data(
                X_arr,
                y_arr,
                sample_weight_arr,
                fit_intercept=original_intercept,
                xp=xp,
                n_eff=n_eff,
            )
        )

        # The transformed problem already contains the correct weighted
        # centering, so the shared unweighted kernel must not center a second
        # time.  Its n_samples normalization is exactly compensated by the row
        # scaling above.  Disable in-kernel inference so post-fit inference runs
        # once on the original X/y/sample_weight with restored intercept
        # semantics.
        saved_use_intercept = self._use_intercept
        saved_compute_inference = self._compute_inference_enabled
        cache_sentinel = object()
        saved_cv_cache = getattr(self, "_cv_cache", cache_sentinel)
        self._use_intercept = False
        self._compute_inference_enabled = False
        if saved_cv_cache is not cache_sentinel:
            del self._cv_cache
        try:
            super()._fit_gpu_backend(
                X_work,
                y_work,
                None,
                backend_name=backend_name,
            )
        finally:
            self._use_intercept = saved_use_intercept
            self._compute_inference_enabled = saved_compute_inference
            if saved_cv_cache is not cache_sentinel:
                self._cv_cache = saved_cv_cache

        coef = np.asarray(self.coef_, dtype=np.float64)
        self.coef_ = coef
        if original_intercept:
            X_mean_np = np.asarray(_to_numpy(X_mean), dtype=np.float64)
            y_mean_value = float(
                np.asarray(_to_numpy(y_mean), dtype=np.float64)
            )
            self.intercept_ = float(y_mean_value - X_mean_np @ coef)
            self._params = np.concatenate([[self.intercept_], coef])
        else:
            self.intercept_ = 0.0
            self._params = coef.copy()
        self._nobs = int(n_samples)
        self._df_resid = int(
            n_samples - (n_features + int(original_intercept))
        )

    def _fit_diagnostic_state(self):
        """Return response/residual state and optional L2 analytic weights."""
        if self._y is None or self._resid is None:
            return None
        y = np.asarray(self._y, dtype=float)

        # Raw residual/weight snapshots are installed by the Gaussian-L2
        # transaction. Do not reuse them after the same estimator is reconfigured
        # and successfully fitted with another penalty.
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        use_l2_diagnostics = penalty_name == "l2"
        raw_resid = (
            getattr(self, "_raw_resid", None) if use_l2_diagnostics else None
        )
        resid = np.asarray(
            self._resid if raw_resid is None else raw_resid,
            dtype=float,
        )
        weights = (
            getattr(self, "_sample_weight_fit", None)
            if use_l2_diagnostics
            else None
        )
        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if weights.shape[0] != y.shape[0]:
                raise RuntimeError(
                    "Stored sample weights no longer match the fitted response state."
                )
        return y, resid, weights

    @property
    def rsquared(self):
        state = self._fit_diagnostic_state()
        if state is None:
            return None
        y, resid, weights = state
        if weights is None:
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum(resid ** 2)
        else:
            y_mean = np.average(y, weights=weights)
            ss_tot = np.sum(weights * (y - y_mean) ** 2)
            ss_res = np.sum(weights * resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @property
    def rsquared_adj(self):
        if self._nobs is None or self._resid is None:
            return None
        if self._df_resid is None or self._df_resid <= 0:
            return np.nan
        r2 = self.rsquared
        if r2 is None:
            return None
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid

    @property
    def fvalue(self):
        state = self._fit_diagnostic_state()
        if state is None:
            return None
        k = len(self.coef_) if self.coef_ is not None else 0
        if k <= 0 or self._df_resid is None or self._df_resid <= 0:
            return np.nan
        y, resid, weights = state
        if weights is None:
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ss_res = float(np.sum(resid ** 2))
        else:
            y_mean = np.average(y, weights=weights)
            ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
            ss_res = float(np.sum(weights * resid ** 2))
        if not np.isfinite(ss_tot) or not np.isfinite(ss_res) or ss_tot <= 0:
            return np.nan
        ss_reg = max(0.0, ss_tot - ss_res)
        tol = np.finfo(float).eps * max(1.0, ss_tot)
        if ss_res <= tol:
            return np.inf if ss_reg > tol else np.nan
        return (ss_reg / k) / (ss_res / self._df_resid)

    @property
    def f_pvalue(self):
        fv = self.fvalue
        if fv is None:
            return None
        if np.isnan(fv):
            return np.nan
        if np.isposinf(fv):
            return 0.0
        k = len(self.coef_) if self.coef_ is not None else 0
        return float(stats.f.sf(fv, k, self._df_resid))

    @property
    def llf(self):
        if self._nobs is None or self._resid is None:
            return None
        n = int(self._nobs)
        if n <= 0:
            return np.nan
        sigma2_mle = float(np.sum(np.asarray(self._resid, dtype=float) ** 2) / n)
        if not np.isfinite(sigma2_mle) or sigma2_mle < 0:
            return np.nan
        if sigma2_mle == 0:
            return np.inf
        return -n / 2 * (np.log(2 * np.pi * sigma2_mle) + 1.0)

    @property
    def aic(self):
        if self._nobs is None or self._scale is None:
            return None
        if np.any(np.isnan(np.asarray(self._scale, dtype=float))):
            return None
        _llf = self.llf
        if _llf is None:
            return None
        return -2 * _llf + 2 * len(self._params)

    @property
    def bic(self):
        if self._nobs is None or self._scale is None:
            return None
        if np.any(np.isnan(np.asarray(self._scale, dtype=float))):
            return None
        _llf = self.llf
        if _llf is None:
            return None
        n = self._nobs
        k = len(self._params)
        return -2 * _llf + k * np.log(n)

    def summary(self):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not self._compute_inference_enabled:
            raise RuntimeError(
                "compute_inference=False: summary/inference statistics are not available. "
                "Re-fit with compute_inference=True to use summary()."
            )
        if self._bse is None:
            raise RuntimeError("Inference statistics are not available.")

        if self._feature_names is not None:
            feature_names = list(self._feature_names)
            if self._effective_intercept:
                feature_names.insert(0, "(Intercept)")
        elif self._effective_intercept:
            feature_names = ["(Intercept)"] + [f"x{i+1}" for i in range(len(self.coef_))]
        else:
            feature_names = [f"x{i+1}" for i in range(len(self.coef_))]

        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        inference_method = str(
            getattr(self, "_inference_method", getattr(self, "inference_method", "debiased"))
        ).lower()
        is_debiased = penalty_name in ("l1", "elasticnet", "en") and "debiased" in inference_method
        is_post_selection_ols = (
            penalty_name in ("l1", "elasticnet", "en")
            and inference_method == "post_selection_ols"
        )

        if is_debiased:
            title = "Debiased Lasso Results"
            stat_label = "z"
            pval_label = "P>|z|"
        elif is_post_selection_ols:
            title = "Post-selection OLS Diagnostic"
            stat_label = str(
                getattr(getattr(self, "_inference_result", None), "statistic_name", "t")
            )
            pval_label = f"P>|{stat_label}|"
        elif penalty_name == "l2":
            title = "Ridge Regression Results"
            stat_label = "t"
            pval_label = "P>|t|"
        else:
            title = "Penalized Linear Regression Results"
            stat_label = "t"
            pval_label = "P>|t|"

        print("=" * 80)
        print(f"{title:^80}")
        print("=" * 80)
        def _fmt(val, spec):
            if val is None:
                return f"{'N/A':>15}"
            return format(val, spec)

        print(f"Alpha:                      {float(self.alpha):>15.4f}")
        if not is_debiased:
            print(f"Covariance Type:            {self._cov_type:>15}")
        print(f"No. Observations:           {self._nobs:>15}")
        if is_post_selection_ols:
            refit_df = getattr(self, "_post_selection_df_resid", None)
            if refit_df is None:
                result_metadata = dict(
                    getattr(getattr(self, "_inference_result", None), "metadata", {})
                    or {}
                )
                refit_df = result_metadata.get("refit_df_resid")
            if refit_df is not None:
                refit_df = int(refit_df)
            print(
                f"Penalized-fit Residual DoF:  {_fmt(self._df_resid, '>15')}"
            )
            print(
                f"Post-selection Refit DoF:    {_fmt(refit_df, '>15')}"
            )
            print("Penalized-fit diagnostics:")
        else:
            print(f"Degrees of Freedom:         {self._df_resid:>15}")
        print(f"R-squared:                  {_fmt(self.rsquared, '>15.4f')}")
        print(f"Adj. R-squared:             {_fmt(self.rsquared_adj, '>15.4f')}")
        print(f"F-statistic:                {_fmt(self.fvalue, '>15.4f')}")
        print(f"Prob (F-statistic):         {_fmt(self.f_pvalue, '>15.4e')}")
        print(f"Log-Likelihood:             {_fmt(self.llf, '>15.4f')}")
        print(f"AIC:                        {_fmt(self.aic, '>15.4f')}")
        print(f"BIC:                        {_fmt(self.bic, '>15.4f')}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {stat_label:>10} {pval_label:>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)

        zvals = self._zvalues if getattr(self, '_tvalues', None) is None else self._tvalues
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{zvals[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")

        if getattr(self, '_simultaneous_enabled', False) and self._conf_int_simultaneous is not None:
            alpha_sim = float(getattr(self, 'simultaneous_alpha',
                                      getattr(self, '_simultaneous_alpha', 0.05)))
            B = int(getattr(self, 'simultaneous_n_bootstrap',
                            getattr(self, '_simultaneous_n_bootstrap', 1000)))
            crit = getattr(self, '_simultaneous_critical_value', None)
            print("-" * 80)
            print("Simultaneous inference (max-|Z| bootstrap)")
            print(f"  alpha:          {alpha_sim:.6f}")
            print(f"  n_bootstrap:    {B}")
            if crit is not None:
                print(f"  critical value (max|Z|): {crit:.4f}")
            print("-" * 80)
            for i, name in enumerate(feature_names):
                lo = self._conf_int_simultaneous[i, 0]
                hi = self._conf_int_simultaneous[i, 1]
                print(f"{name:<15} {'':>12} {'':>12} {'':>10} {'':>10} {lo:>12.4f} {hi:>12.4f}")

        print("=" * 80)