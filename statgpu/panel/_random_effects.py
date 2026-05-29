"""
Random effects panel data model.

Implements the Swamy-Arora random effects estimator via feasible GLS.
The model is::

    y_{it} = alpha + X_{it}' beta + a_i + epsilon_{it}

where ``a_i ~ iid(0, sigma2_a)`` is the individual random effect and
``epsilon_{it} ~ iid(0, sigma2_e)`` is the idiosyncratic error.

Note: ``X`` should include a constant column if an intercept is desired;
the model does not add one automatically.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _get_torch_device_str, _torch_dev, _to_numpy, xp_astype, xp_zeros, xp_cholesky_solve

from ._utils import within_transform, group_means, group_sizes, ols_inference_nonrobust


class RandomEffects(BaseEstimator):
    """Random effects estimator for panel data.

    Implements feasible GLS random effects (Swamy-Arora) with variance
    component estimation.

    Parameters
    ----------
    device : str or Device, default='auto'
        Computation device.

    Attributes
    ----------
    coef_ : ndarray, shape (k,)
        Estimated slope coefficients.
    bse_ : ndarray, shape (k,)
        Standard errors.
    tvalues_ : ndarray, shape (k,)
        t-statistics.
    pvalues_ : ndarray, shape (k,)
        Two-sided p-values.
    conf_int_ : ndarray, shape (k, 2)
        95 % confidence intervals.
    theta_ : float
        GLS transformation parameter.
    variance_components_ : dict
        ``{'sigma2_e': float, 'sigma2_a': float}``.
    nobs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    """

    def __init__(
        self,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)

        # Public attributes
        self.coef_ = None
        self.bse_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.conf_int_ = None
        self.theta_ = None
        self.variance_components_ = None
        self.nobs = None
        self.df_resid = None

        # Internal
        self._params = None
        self._resid = None
        self._scale = None
        self._X_design = None
        self._y_transformed = None

    def fit(self, y, X, entity_ids=None, time_ids=None):
        """Fit the random effects model.

        Parameters
        ----------
        y : array-like, shape (n,)
            Outcome vector.
        X : array-like, shape (n, k)
            Regressor matrix.
        entity_ids : array-like, shape (n,)
            Entity (individual) identifiers.  **Required.**
        time_ids : array-like, shape (n,), optional
            Time-period identifiers (currently unused but reserved for
            future extensions).

        Returns
        -------
        self
        """
        if entity_ids is None:
            raise ValueError("entity_ids is required for RandomEffects")

        # Resolve backend
        backend = self._get_backend(backend='auto')
        backend_name = backend.name
        xp = backend.xp

        # Convert inputs
        y_arr = xp_astype(self._to_array(y, backend=backend_name).ravel(), xp.float64, xp)
        X_arr = xp_astype(self._to_array(X, backend=backend_name), xp.float64, xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        entity_arr = self._to_array(entity_ids, backend=backend_name).ravel()
        n, k = X_arr.shape
        self.nobs = n

        # --- Step 1: Between estimation (group means) ---
        y_bar_i = group_means(y_arr, entity_arr, xp=xp)
        X_bar_i = xp.zeros_like(X_arr)
        for j in range(k):
            X_bar_i[:, j] = group_means(X_arr[:, j], entity_arr, xp=xp)

        # Between OLS: beta_between = (X_bar'X_bar)^{-1} X_bar' y_bar
        # (Only need group-level data, but the aligned arrays work too.)
        XtX_b = X_bar_i.T @ X_bar_i
        Xty_b = X_bar_i.T @ y_bar_i
        try:
            beta_between = xp.linalg.solve(XtX_b, Xty_b)
        except Exception:
            beta_between = xp.linalg.pinv(XtX_b) @ Xty_b

        # Between residuals
        resid_between = y_bar_i - X_bar_i @ beta_between
        rss_between = float(xp.sum(resid_between ** 2))

        # --- Step 2: Within estimation (entity demeaning) ---
        y_within = within_transform(y_arr, entity_arr, xp=xp)
        X_within = xp.zeros_like(X_arr)
        for j in range(k):
            X_within[:, j] = within_transform(X_arr[:, j], entity_arr, xp=xp)

        XtX_w = X_within.T @ X_within
        Xty_w = X_within.T @ y_within
        try:
            beta_within = xp.linalg.solve(XtX_w, Xty_w)
        except Exception:
            beta_within = xp.linalg.pinv(XtX_w) @ Xty_w

        resid_within = y_within - X_within @ beta_within
        rss_within = float(xp.sum(resid_within ** 2))

        # --- Step 3: Variance components ---
        unique_entities = xp.unique(entity_arr)
        n_entities = len(unique_entities)
        T_i = group_sizes(entity_arr, xp=xp)
        T_i_np = _to_numpy(T_i)

        # Harmonic mean of group sizes: n / sum(1/T_i) for all groups
        T_bar = float(n_entities) / float(np.sum(1.0 / T_i_np))

        # df for within residuals: n*T - k - (n_entities - 1)
        df_within = n - k - (n_entities - 1)
        if df_within <= 0:
            raise ValueError(
                f"Not enough observations for within df: n={n}, k={k}, "
                f"n_entities={n_entities}, df_within={df_within}"
            )

        sigma2_e = rss_within / df_within
        sigma2_a_raw = rss_between / n_entities - sigma2_e / T_bar
        sigma2_a = max(0.0, sigma2_a_raw)

        self.variance_components_ = {
            'sigma2_e': sigma2_e,
            'sigma2_a': sigma2_a,
        }

        # --- Step 4: GLS transformation ---
        # theta_i = 1 - sqrt(sigma2_e / (sigma2_e + T_i * sigma2_a))
        T_i_unique = np.unique(T_i_np)
        theta_map = {}
        for Ti in T_i_unique:
            denom = sigma2_e + Ti * sigma2_a
            if denom > 0:
                theta_map[Ti] = 1.0 - np.sqrt(sigma2_e / denom)
            else:
                theta_map[Ti] = 0.0

        # Build theta per observation
        theta_arr = xp_zeros(n, xp.float64, xp, X_arr)
        for Ti, th in theta_map.items():
            mask = T_i == Ti
            theta_arr[mask] = th

        self.theta_ = float(np.mean(list(theta_map.values())))

        # Transformed variables: y* = y - theta * y_bar
        y_star = y_arr - theta_arr * y_bar_i
        X_star = xp.zeros_like(X_arr)
        for j in range(k):
            X_star[:, j] = X_arr[:, j] - theta_arr * X_bar_i[:, j]

        self._y_transformed = _to_numpy(y_star)

        # --- Step 5: OLS on transformed data ---
        XtX_s = X_star.T @ X_star
        Xty_s = X_star.T @ y_star
        try:
            beta_gls = xp_cholesky_solve(XtX_s, Xty_s, xp)
        except Exception:
            beta_gls = xp.linalg.solve(XtX_s, Xty_s)

        resid_gls = y_star - X_star @ beta_gls
        df_resid = n - k
        self.df_resid = df_resid
        self._scale = float(xp.sum(resid_gls ** 2)) / df_resid

        # Store coefficients
        coef_np = _to_numpy(beta_gls).ravel()
        self._params = coef_np
        self.coef_ = coef_np
        self._X_design = _to_numpy(X_star)
        self._resid = _to_numpy(resid_gls).ravel()

        # --- Step 6: Inference ---
        self._compute_inference()

        self._fitted = True
        return self

    def _compute_inference(self):
        """Compute SE, t-values, p-values, and confidence intervals."""
        self.bse_, self.tvalues_, self.pvalues_, self.conf_int_ = \
            ols_inference_nonrobust(self._params, self._X_design, self._scale, self.df_resid)

    def predict(self, X):
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Regressor matrix.

        Returns
        -------
        y_pred : ndarray, shape (n,)
            Predicted values.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr @ self.coef_

    def summary(self):
        """Print a coefficient table with SE, t, p, and 95 % CI."""
        self._check_is_fitted()

        k = len(self._params)
        feat_names = [f'x{i+1}' for i in range(k)]

        print("=" * 72)
        print("                      Random Effects Results")
        print("=" * 72)
        print(f"No. Observations:   {self.nobs:>10}")
        print(f"Degrees of Freedom: {self.df_resid:>10}")
        if self.variance_components_:
            print(f"sigma2_e:           {self.variance_components_['sigma2_e']:>10.6f}")
            print(f"sigma2_a:           {self.variance_components_['sigma2_a']:>10.6f}")
        print(f"theta (avg):        {self.theta_:>10.4f}")
        print("-" * 72)
        header = f"{'':<12} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        print(header)
        print("-" * 72)
        for i, name in enumerate(feat_names):
            print(
                f"{name:<12} {self._params[i]:>10.4f} {self.bse_[i]:>10.4f} "
                f"{self.tvalues_[i]:>8.3f} {self.pvalues_[i]:>10.4f} "
                f"{self.conf_int_[i, 0]:>10.4f} {self.conf_int_[i, 1]:>10.4f}"
            )
        print("=" * 72)
