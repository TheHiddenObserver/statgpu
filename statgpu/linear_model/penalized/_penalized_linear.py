"""PenalizedLinearRegression — thin wrapper over PenalizedGeneralizedLinearModel."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import stats

from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


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
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
            loss_kwargs=loss_kwargs,
        )

    @property
    def rsquared(self):
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    @property
    def rsquared_adj(self):
        if self._nobs is None or self._resid is None:
            return None
        r2 = self.rsquared
        if r2 is None:
            return None
        k = len(self.coef_) if self.coef_ is not None else 0
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid

    @property
    def fvalue(self):
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        ss_reg = ss_tot - ss_res
        k = len(self.coef_) if self.coef_ is not None else 0
        if k == 0 or ss_res <= 0:
            return np.inf
        return (ss_reg / k) / (ss_res / self._df_resid)

    @property
    def f_pvalue(self):
        fv = self.fvalue
        if fv is None:
            return 1.0
        k = len(self.coef_) if self.coef_ is not None else 0
        if k == 0:
            return None  # No predictors — F-test is undefined
        if np.isposinf(fv):
            return 0.0
        return 1 - stats.f.cdf(fv, k, self._df_resid)

    @property
    def llf(self):
        if self._nobs is None or self._resid is None:
            return None
        n = self._nobs
        sigma2_mle = np.sum(self._resid ** 2) / n
        return -n / 2 * np.log(2 * np.pi * sigma2_mle) - n / 2

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
        if not self.compute_inference:
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
        inference_method = str(getattr(self, "inference_method", "debiased")).lower()
        is_debiased = penalty_name in ("l1", "elasticnet", "en") and "debiased" in inference_method

        if is_debiased:
            title = "Debiased Lasso Results"
            stat_label = "z"
            pval_label = "P>|z|"
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
            print(f"Covariance Type:            {self.cov_type:>15}")
        print(f"No. Observations:           {self._nobs:>15}")
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

        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._tvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
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



