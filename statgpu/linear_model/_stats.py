"""
Statistical inference for linear models.
Computes standard errors, t-statistics, p-values, etc.
"""

import numpy as np
from statgpu.inference import t as t_dist, f as f_dist


class RegressionResults:
    """
    Results class for linear regression with statistical inference.
    
    Similar to statsmodels RegressionResultsWrapper.
    """
    
    def __init__(self, model, params, resid, scale, nobs, df_resid):
        """
        Initialize results object.
        
        Parameters
        ----------
        model : fitted model instance
        params : ndarray
            Estimated parameters (including intercept if fitted)
        resid : ndarray
            Residuals
        scale : float
            Estimate of error variance (sigma^2)
        nobs : int
            Number of observations
        df_resid : int
            Degrees of freedom of residuals
        """
        self.model = model
        self.params = params
        self.resid = resid
        self.scale = scale
        self.nobs = nobs
        self.df_resid = df_resid
        
        # Compute standard errors and statistics
        self._compute_inference()
    
    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values, confidence intervals."""
        # Get design matrix
        X = self.model._X_design
        
        # Compute (X'X)^-1
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        # Standard errors: sqrt(scale * diag((X'X)^-1))
        self.bse = np.sqrt(self.scale * np.diag(XtX_inv))
        
        # t-statistics: coef / std_err.  Use explicit division semantics so
        # exact-fit zero standard errors produce signed infinities without a
        # spurious RuntimeWarning.
        self.tvalues = np.divide(
            self.params,
            self.bse,
            out=np.full_like(np.asarray(self.params, dtype=float), np.nan),
            where=self.bse != 0,
        )
        zero_bse = self.bse == 0
        self.tvalues[zero_bse & (self.params > 0)] = np.inf
        self.tvalues[zero_bse & (self.params < 0)] = -np.inf
        
        # p-values: two-tailed t-test
        self.pvalues = 2 * t_dist.sf(np.abs(self.tvalues), df=self.df_resid)

        # Confidence intervals (95%)
        alpha = 0.05
        t_crit = float(t_dist.ppf(1 - alpha/2, df=self.df_resid))
        self._conf_int = np.column_stack([
            self.params - t_crit * self.bse,
            self.params + t_crit * self.bse
        ])
    
    @property
    def rsquared(self):
        """R-squared."""
        y = self.model._y
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum(self.resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    @property
    def rsquared_adj(self):
        """Adjusted R-squared."""
        if self.df_resid <= 0:
            return np.nan
        return 1 - (1 - self.rsquared) * (self.nobs - 1) / self.df_resid
    
    @property
    def fvalue(self):
        """F-statistic for overall model significance."""
        k = len(self.params) - 1
        if k <= 0 or self.df_resid <= 0:
            return np.nan
        y = np.asarray(self.model._y, dtype=float)
        resid = np.asarray(self.resid, dtype=float)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(resid ** 2))
        if not np.isfinite(ss_tot) or not np.isfinite(ss_res) or ss_tot <= 0:
            return np.nan
        ss_reg = max(0.0, ss_tot - ss_res)
        tol = np.finfo(float).eps * max(1.0, ss_tot)
        if ss_res <= tol:
            return np.inf if ss_reg > tol else np.nan
        return (ss_reg / k) / (ss_res / self.df_resid)
    
    @property
    def f_pvalue(self):
        """Upper-tail p-value for the overall F-test."""
        fv = self.fvalue
        if np.isnan(fv):
            return np.nan
        if np.isposinf(fv):
            return 0.0
        k = len(self.params) - 1
        return float(f_dist.sf(fv, dfn=k, dfd=self.df_resid))
    
    @property
    def aic(self):
        """Akaike Information Criterion."""
        llf = self.llf
        if np.isnan(llf):
            return np.nan
        return -2 * llf + 2 * len(self.params)
    
    @property
    def bic(self):
        """Bayesian Information Criterion."""
        llf = self.llf
        if np.isnan(llf):
            return np.nan
        return -2 * llf + len(self.params) * np.log(self.nobs)
    
    def summary(self):
        """Print summary table similar to R's summary(lm())."""
        # Get feature names
        if hasattr(self.model, '_feature_names'):
            feature_names = self.model._feature_names
        else:
            feature_names = ['(Intercept)'] + [f'x{i}' for i in range(len(self.params) - 1)]
        
        # Build summary table
        print("=" * 80)
        print("Linear Regression Results")
        print("=" * 80)
        print(f"No. Observations:           {self.nobs:>15}")
        print(f"Degrees of Freedom:         {self.df_resid:>15}")
        print(f"R-squared:                  {self.rsquared:>15.4f}")
        print(f"Adj. R-squared:             {self.rsquared_adj:>15.4f}")
        print(f"F-statistic:                {self.fvalue:>15.4f}")
        print(f"Prob (F-statistic):         {self.f_pvalue:>15.4e}")
        print(f"Log-Likelihood:             {self.llf:>15.4f}")
        print(f"AIC:                        {self.aic:>15.4f}")
        print(f"BIC:                        {self.bic:>15.4f}")
        print("-" * 80)
        print(f"{'':<20} {'coef':>12} {'std err':>12} {'t':>10} {'P>|t|':>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)
        
        ci = self.conf_int()
        for i, name in enumerate(feature_names):
            print(f"{name:<20} {self.params[i]:>12.4f} {self.bse[i]:>12.4f} "
                  f"{self.tvalues[i]:>10.3f} {self.pvalues[i]:>10.4f} "
                  f"{ci[i, 0]:>12.4f} {ci[i, 1]:>12.4f}")
        
        print("=" * 80)
    
    @property
    def llf(self):
        """Log-likelihood."""
        scale = float(self.scale)
        if not np.isfinite(scale) or scale < 0:
            return np.nan
        if scale == 0:
            return np.inf
        return -self.nobs / 2 * (np.log(2 * np.pi * scale) + 1)
    
    def conf_int(self, alpha=0.05):
        """Confidence intervals for parameters."""
        t_crit = float(t_dist.ppf(1 - alpha/2, df=self.df_resid))
        return np.column_stack([
            self.params - t_crit * self.bse,
            self.params + t_crit * self.bse
        ])
