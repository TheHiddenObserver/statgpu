"""
Fixed effects panel data model (PanelOLS).

Implements one-way and two-way fixed effects estimation with support
for non-robust, HC1 robust, and clustered standard errors.  GPU
acceleration is provided transparently via the statgpu backend system.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _get_torch_device_str, _torch_dev, _to_numpy, xp_astype, xp_cholesky_solve

from ._utils import demean_variables, ols_inference_nonrobust
from ._covariance import clustered_covariance, two_way_clustered_covariance


class PanelOLS(BaseEstimator):
    """Fixed effects estimator for panel data.

    Supports entity (individual) fixed effects, time fixed effects,
    and two-way fixed effects via the within transformation.

    Parameters
    ----------
    entity_effects : bool, default=False
        Include entity (individual) fixed effects.
    time_effects : bool, default=False
        Include time fixed effects.
    cov_type : str, default='nonrobust'
        Covariance estimator: ``'nonrobust'``, ``'robust'`` (HC1), or
        ``'clustered'``.
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
    rsquared_within : float
        Within R-squared (variance explained by regressors after demeaning).
    nobs : int
        Number of observations used in estimation.
    df_resid : int
        Residual degrees of freedom.
    """

    def __init__(
        self,
        entity_effects: bool = False,
        time_effects: bool = False,
        cov_type: str = 'nonrobust',
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.cov_type = cov_type.lower()
        if self.cov_type not in ('nonrobust', 'robust', 'clustered'):
            raise ValueError(
                "cov_type must be 'nonrobust', 'robust', or 'clustered'"
            )

        # Public attributes set by fit()
        self.coef_ = None
        self.bse_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.conf_int_ = None
        self.rsquared_within = None
        self.nobs = None
        self.df_resid = None

        # Internal storage
        self._params = None
        self._resid = None
        self._scale = None
        self._X_design = None
        self._y_demeaned = None

    def fit(self, y, X, entity_ids=None, time_ids=None, cluster=None):
        """Fit the fixed effects model.

        Parameters
        ----------
        y : array-like, shape (n,)
            Outcome vector.
        X : array-like, shape (n, k)
            Regressor matrix. Do NOT include a constant column when using
            entity_effects or time_effects (it would be collinear with the
            dummies and cause a singular matrix error).
        entity_ids : array-like, shape (n,), optional
            Entity (individual) identifiers.  Required when
            ``entity_effects=True``.
        time_ids : array-like, shape (n,), optional
            Time-period identifiers.  Required when ``time_effects=True``.
        cluster : array-like, shape (n,), optional
            Cluster labels for clustered standard errors.  Required when
            ``cov_type='clustered'``.

        Returns
        -------
        self
        """
        # Resolve backend
        backend = self._get_backend(backend='auto')
        backend_name = backend.name
        xp = backend.xp

        # Convert inputs to backend arrays
        y_arr = xp_astype(self._to_array(y, backend=backend_name).ravel(), xp.float64, xp)
        X_arr = xp_astype(self._to_array(X, backend=backend_name), xp.float64, xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n, k = X_arr.shape
        self.nobs = n

        # Validate
        if self.entity_effects and entity_ids is None:
            raise ValueError("entity_ids is required when entity_effects=True")
        if self.time_effects and time_ids is None:
            raise ValueError("time_ids is required when time_effects=True")
        if self.cov_type == 'clustered' and cluster is None:
            raise ValueError("cluster is required when cov_type='clustered'")

        entity_arr = None
        time_arr = None
        if entity_ids is not None:
            entity_arr = self._to_array(entity_ids, backend=backend_name).ravel()
        if time_ids is not None:
            time_arr = self._to_array(time_ids, backend=backend_name).ravel()

        # Demean if fixed effects requested
        if self.entity_effects or self.time_effects:
            y_d, X_d = demean_variables(
                y_arr, X_arr,
                entity_ids=entity_arr if self.entity_effects else None,
                time_ids=time_arr if self.time_effects else None,
                xp=xp,
            )
        else:
            y_d = y_arr
            X_d = X_arr

        self._y_demeaned = _to_numpy(y_d)

        # OLS on demeaned data: beta = (X'X)^{-1} X'y
        XtX = X_d.T @ X_d
        Xty = X_d.T @ y_d

        try:
            coef = xp_cholesky_solve(XtX, Xty, xp)
        except _LINALG_ERRORS:
            coef = xp.linalg.solve(XtX, Xty)

        # Degrees of freedom (compute from numpy to avoid GPU sync)
        entity_np_dof = _to_numpy(entity_arr) if entity_arr is not None else None
        time_np_dof = _to_numpy(time_arr) if time_arr is not None else None
        n_entities = len(np.unique(entity_np_dof)) if entity_np_dof is not None else 0
        n_times = len(np.unique(time_np_dof)) if time_np_dof is not None else 0
        n_effects = 0
        if self.entity_effects:
            n_effects += n_entities - 1
        if self.time_effects:
            n_effects += n_times - 1
        self.df_resid = n - k - n_effects

        if self.df_resid <= 0:
            raise ValueError(
                f"Not enough observations: n={n}, k={k}, n_effects={n_effects}, "
                f"df_resid={self.df_resid}.  Check that N*T >> k + effects."
            )

        # Residuals and scale (on the demeaned data)
        y_pred = X_d @ coef
        resid = y_d - y_pred
        self._resid = _to_numpy(resid).ravel()
        self._scale = float(xp.sum(resid ** 2)) / self.df_resid

        # Store coefficients (numpy)
        coef_np = _to_numpy(coef).ravel()
        self._params = coef_np
        self.coef_ = coef_np
        self._X_design = _to_numpy(X_d)

        # Store fixed effect estimates for prediction
        self._entity_ids_fit = _to_numpy(entity_arr) if entity_arr is not None else None
        self._time_ids_fit = _to_numpy(time_arr) if time_arr is not None else None
        if self.entity_effects or self.time_effects:
            y_np = _to_numpy(y_arr)
            X_np = _to_numpy(X_arr)
            y_hat = X_np @ coef_np
            resid_fe = y_np - y_hat
            if self.entity_effects:
                entity_np = self._entity_ids_fit
                unique_entities = np.unique(entity_np)
                self._entity_effects_ = {}
                for ent in unique_entities:
                    mask = entity_np == ent
                    self._entity_effects_[ent] = float(np.mean(resid_fe[mask]))
            if self.time_effects:
                time_np = self._time_ids_fit
                unique_times = np.unique(time_np)
                self._time_effects_ = {}
                # For two-way FE: demean time effects to avoid double-counting grand mean
                # (entity effects already absorb the grand mean)
                grand_mean = float(np.mean(resid_fe)) if self.entity_effects else 0.0
                for t in unique_times:
                    mask = time_np == t
                    self._time_effects_[t] = float(np.mean(resid_fe[mask])) - grand_mean

        # Inference
        self._compute_inference(xp, cluster, backend_name)

        self._fitted = True
        return self

    def _compute_inference(self, xp, cluster, backend_name):
        """Compute SE, t-values, p-values, and confidence intervals."""
        X = self._X_design
        n, k = X.shape
        params = self._params
        resid = self._resid
        df = self.df_resid

        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        alpha = 0.05

        if self.cov_type == 'nonrobust':
            self.bse_, self.tvalues_, self.pvalues_, self.conf_int_ = \
                ols_inference_nonrobust(params, X, self._scale, df)

        elif self.cov_type == 'robust':
            # HC1 sandwich: use df_resid for small-sample correction
            e2 = resid ** 2
            Xw = X * e2[:, np.newaxis]
            meat = X.T @ Xw
            cov_params = XtX_inv @ meat @ XtX_inv
            if self.df_resid > 0:
                cov_params *= (n / self.df_resid)
            self.bse_ = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
            self.tvalues_ = params / (self.bse_ + 1e-30)
            # Use t-distribution for consistency with nonrobust path
            self.pvalues_ = 2 * (1 - stats.t.cdf(np.abs(self.tvalues_), df=self.df_resid))
            t_crit = stats.t.ppf(1 - alpha / 2, df=self.df_resid)
            self.conf_int_ = np.column_stack([
                params - t_crit * self.bse_,
                params + t_crit * self.bse_,
            ])

        else:  # clustered
            cluster_np = _to_numpy(cluster)
            # Check if two-way clustering is needed
            # (user passes a combined cluster for one-way,
            #  or we detect multiple columns)
            if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
                V = two_way_clustered_covariance(
                    X, resid, cluster_np[:, 0], cluster_np[:, 1], xp=np
                )
            else:
                V = clustered_covariance(X, resid, cluster_np, xp=np)

            self.bse_ = np.sqrt(np.maximum(np.diag(V), 0.0))
            self.tvalues_ = params / (self.bse_ + 1e-30)
            # Cluster-robust: use t with min(n_clusters-1, df_resid) df
            # For two-way clustering, use min across individual dimensions
            if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
                n_clust_1 = len(np.unique(cluster_np[:, 0]))
                n_clust_2 = len(np.unique(cluster_np[:, 1]))
                n_clusters = min(n_clust_1, n_clust_2)
            else:
                n_clusters = len(np.unique(cluster_np))
            df_cluster = min(n_clusters - 1, self.df_resid)
            self.pvalues_ = 2 * (1 - stats.t.cdf(np.abs(self.tvalues_), df=df_cluster))
            t_crit = stats.t.ppf(1 - alpha / 2, df=df_cluster)
            self.conf_int_ = np.column_stack([
                params - t_crit * self.bse_,
                params + t_crit * self.bse_,
            ])

        # Within R-squared: ss_tot should be sum of squared demeaned y (not variance)
        y_d = self._y_demeaned
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum(y_d ** 2)
        self.rsquared_within = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, X, entity_ids=None, time_ids=None):
        """Predict using the fitted model.

        For models with fixed effects, the entity/time intercepts are added
        back to the linear prediction.  If entity_ids/time_ids are not
        provided, only the slope contribution is returned.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Regressor matrix.
        entity_ids : array-like, shape (n,), optional
            Entity identifiers for entity fixed effects.
        time_ids : array-like, shape (n,), optional
            Time identifiers for time fixed effects.

        Returns
        -------
        y_pred : ndarray, shape (n,)
            Predicted values.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_pred = X_arr @ self.coef_

        # Add fixed effect intercepts (vectorized)
        if hasattr(self, '_entity_effects_') and entity_ids is not None:
            entity_arr = np.asarray(entity_ids).ravel()
            entity_vec = np.vectorize(lambda e: self._entity_effects_.get(e, 0.0))
            y_pred += entity_vec(entity_arr)
        if hasattr(self, '_time_effects_') and time_ids is not None:
            time_arr = np.asarray(time_ids).ravel()
            time_vec = np.vectorize(lambda t: self._time_effects_.get(t, 0.0))
            y_pred += time_vec(time_arr)

        return y_pred

    def summary(self):
        """Print a coefficient table with SE, t, p, and 95 % CI."""
        self._check_is_fitted()

        k = len(self._params)
        feat_names = [f'x{i+1}' for i in range(k)]

        print("=" * 72)
        print("                        Panel OLS Results")
        print("=" * 72)
        print(f"Entity effects:     {str(self.entity_effects):>10}")
        print(f"Time effects:       {str(self.time_effects):>10}")
        print(f"Covariance type:    {self.cov_type:>10}")
        print(f"No. Observations:   {self.nobs:>10}")
        print(f"Degrees of Freedom: {self.df_resid:>10}")
        print(f"Within R-squared:   {self.rsquared_within:>10.4f}")
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
