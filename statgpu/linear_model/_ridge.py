"""
Ridge regression with full statistical inference and GPU support.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class Ridge(BaseEstimator):
    """
    Ridge regression (L2 regularization) with GPU acceleration
    and full statistical inference.

    Uses closed-form solution: beta = (X'X + alpha*I)^(-1) X'y

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

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
        """Fit Ridge regression model."""
        self._y = np.asarray(y)

        X_arr = self._to_array(X)
        y_arr = self._to_array(y)

        device = self._get_compute_device()

        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        self._compute_inference()
        self._fitted = True
        return self

    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU with closed-form solution."""
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
            # Center X and y for intercept handling
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = 0.0

        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        # Ridge closed-form: beta = (X'X + alpha*I)^(-1) X'y
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Add regularization to diagonal (excluding intercept)
        I = np.eye(n_features)
        XtX_reg = XtX + self.alpha * I

        # Solve for coefficients
        try:
            coef = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]

        coef = coef.flatten()

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
            self.coef_ = coef
            self._params = np.concatenate([[self.intercept_], self.coef_])
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            self._params = self.coef_.copy()
            self._X_design = X.copy()

        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan

    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU with closed-form solution."""
        import cupy as cp

        n_samples, n_features = X.shape
        self._nobs = n_samples

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
            y_mean = 0.0

        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        # Ridge closed-form on GPU
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        I = cp.eye(n_features)
        XtX_reg = XtX + self.alpha * I

        try:
            coef = cp.linalg.solve(XtX_reg, Xty)
        except Exception:
            coef = cp.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]

        coef_np = coef.get().flatten()

        if self.fit_intercept:
            self.intercept_ = float(y_mean.get() - (X_mean @ coef).get())
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
            self._X_design = np.column_stack([np.ones(n_samples, dtype=np.float64), X.get()])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np
            self._X_design = X.get()

        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan

    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return

        # For Ridge, we use the regularized covariance estimate
        X = self._X_design
        n_samples = self._nobs
        n_features = len(self.coef_)

        try:
            XtX = X.T @ X
            # Add regularization for Ridge standard errors
            I = np.eye(XtX.shape[0])
            if self.fit_intercept:
                # Don't regularize intercept for inference
                I[0, 0] = 0
            XtX_inv = np.linalg.inv(XtX + self.alpha * I)
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

    @property
    def rsquared(self):
        """R-squared."""
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

        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]

        print("=" * 80)
        print("                            Ridge Regression Results")
        print(f"                            (alpha = {self.alpha:.4f})")
        print("=" * 80)
        print(f"No. Observations:           {self._nobs:>15}")
        print(f"Degrees of Freedom:         {self._df_resid:>15}")
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
        """Predict using the Ridge model."""
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