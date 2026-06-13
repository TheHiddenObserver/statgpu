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
from statgpu.backends import _LINALG_ERRORS, _get_torch_device_str, _torch_dev, _to_float_scalar, _to_numpy, xp_astype, xp_cholesky_solve

from statgpu.panel._utils import PanelSummary, _scatter_add, demean_variables
from statgpu.panel._covariance import clustered_covariance, two_way_clustered_covariance


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
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.cov_type = cov_type.lower()
        self.alpha = alpha
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
        self._scale = None
        self._entity_effects_map = {}
        self._time_effects_map = {}

    def fit(self, X, y, entity_ids=None, time_ids=None, cluster=None):
        """Fit the fixed effects model.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Regressor matrix. Include a constant column if you want an
            intercept (the model does not add one automatically).
        y : array-like, shape (n,)
            Outcome vector.
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

        # Validate shapes
        if y_arr.shape[0] != n:
            raise ValueError(
                f"y has {y_arr.shape[0]} observations but X has {n} rows"
            )

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

        # OLS on demeaned data: beta = (X'X)^{-1} X'y
        XtX = X_d.T @ X_d
        Xty = X_d.T @ y_d

        try:
            coef = xp_cholesky_solve(XtX, Xty, xp)
        except _LINALG_ERRORS:
            coef = xp.linalg.solve(XtX, Xty)

        # Degrees of freedom
        n_entities = len(xp.unique(entity_arr)) if entity_arr is not None else 0
        n_times = len(xp.unique(time_arr)) if time_arr is not None else 0
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

        # Residuals and scale (on the demeaned data, all on device)
        y_pred = X_d @ coef
        resid = y_d - y_pred
        scale = _to_float_scalar(xp.sum(resid ** 2)) / self.df_resid
        self._scale = scale

        # Compute entity/time effects for predict()
        # Subtract grand mean to avoid double-counting in two-way FE
        self._entity_effects_map = {}
        self._time_effects_map = {}
        resid_orig = y_arr - X_arr @ coef
        grand_mean = float(xp.mean(resid_orig))
        resid_centered = resid_orig - grand_mean
        self._grand_mean = grand_mean

        if self.entity_effects and entity_arr is not None:
            ent_np = _to_numpy(entity_arr).ravel()
            unique_ent, idx_np = np.unique(ent_np, return_inverse=True)
            idx_dev = xp.asarray(idx_np, dtype=xp.int64)
            ent_sums = _scatter_add(xp, idx_dev, resid_centered, len(unique_ent))
            ent_counts = _scatter_add(xp, idx_dev, xp.ones_like(resid_centered), len(unique_ent))
            ent_effects = _to_numpy(ent_sums / xp.maximum(ent_counts, 1.0)).ravel()
            for i, eid in enumerate(unique_ent):
                self._entity_effects_map[eid] = float(ent_effects[i])
        if self.time_effects and time_arr is not None:
            time_np = _to_numpy(time_arr).ravel()
            unique_time, idx_np = np.unique(time_np, return_inverse=True)
            idx_dev = xp.asarray(idx_np, dtype=xp.int64)
            time_sums = _scatter_add(xp, idx_dev, resid_centered, len(unique_time))
            time_counts = _scatter_add(xp, idx_dev, xp.ones_like(resid_centered), len(unique_time))
            time_effects = _to_numpy(time_sums / xp.maximum(time_counts, 1.0)).ravel()
            for i, tid in enumerate(unique_time):
                self._time_effects_map[tid] = float(time_effects[i])

        # Keep arrays on device for inference — only transfer final results
        self._compute_inference(xp, cluster, backend_name,
                                X_d, coef, resid, y_d)

        # Single batch transfer of final results to CPU
        self._params = _to_numpy(coef).ravel()
        self.coef_ = self._params

        self._fitted = True
        return self

    def _compute_inference(self, xp, cluster, backend_name,
                           X_d, coef, resid, y_d):
        """Compute SE, t-values, p-values, and CIs — all on device.

        Uses statgpu's backend-agnostic inference framework for p-values,
        so no GPU→CPU transfer is needed for the computation.  Only the
        final numpy result vectors are stored for the user API.
        """
        from statgpu.inference._distributions_backend import get_distribution

        n, k = X_d.shape
        df = self.df_resid
        alpha = self.alpha

        # XtX and its inverse — on device
        XtX = X_d.T @ X_d
        try:
            XtX_inv = xp.linalg.inv(XtX)
        except _LINALG_ERRORS:
            XtX_inv = xp.linalg.pinv(XtX)

        if self.cov_type == 'nonrobust':
            cov_params = self._scale * XtX_inv
            bse_dev = xp.sqrt(xp.maximum(xp.diag(cov_params), 0.0))

        elif self.cov_type == 'robust':
            # HC1 sandwich — on device
            e2 = resid ** 2
            Xw = X_d * e2[:, None]
            meat = X_d.T @ Xw
            cov_params = XtX_inv @ meat @ XtX_inv
            if n > k:
                cov_params = cov_params * (n / (n - k))
            bse_dev = xp.sqrt(xp.maximum(xp.diag(cov_params), 0.0))

        else:  # clustered
            cluster_np = _to_numpy(cluster)
            # Validate cluster length matches fitted data
            if len(cluster_np) != X_d.shape[0]:
                raise ValueError(
                    f"cluster length ({len(cluster_np)}) does not match "
                    f"data length ({X_d.shape[0]})"
                )
            if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
                V = two_way_clustered_covariance(
                    X_d, resid, cluster_np[:, 0], cluster_np[:, 1], xp=xp
                )
            else:
                V = clustered_covariance(X_d, resid, cluster_np, xp=xp)
            bse_dev = xp.sqrt(xp.maximum(xp.diag(V), 0.0))

        # t-values — on device
        _eps = xp.finfo(xp.float64).tiny if hasattr(xp, 'finfo') else 2.2e-308
        tvalues_dev = coef / xp.maximum(bse_dev, _eps)
        abs_t = xp.abs(tvalues_dev)

        # p-values via backend-agnostic inference framework — on device
        if self.cov_type in ('nonrobust',):
            t_dist = get_distribution("t", backend=backend_name)
            pvalues_dev = 2.0 * t_dist.sf(abs_t, float(df))
            t_crit = float(t_dist.isf(xp.asarray([alpha / 2.0]), float(df))[0])
        else:
            norm_dist = get_distribution("norm", backend=backend_name)
            pvalues_dev = 2.0 * norm_dist.sf(abs_t)
            t_crit = float(norm_dist.isf(xp.asarray([alpha / 2.0]))[0])

        # Final transfer: only k-length vectors to CPU for storage
        self.bse_ = _to_numpy(bse_dev).ravel()
        self.tvalues_ = _to_numpy(tvalues_dev).ravel()
        self.pvalues_ = _to_numpy(pvalues_dev).ravel()

        coef_np = _to_numpy(coef).ravel()
        self.conf_int_ = np.column_stack([
            coef_np - t_crit * self.bse_,
            coef_np + t_crit * self.bse_,
        ])

        # Within R-squared — on device, single sync
        ss_res = _to_float_scalar(xp.sum(resid ** 2))
        y_d_mean = _to_float_scalar(xp.mean(y_d))
        ss_tot = _to_float_scalar(xp.sum((y_d - y_d_mean) ** 2))
        self.rsquared_within = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, X, entity_ids=None, time_ids=None):
        """Predict using the fitted model.

        If the model was fitted with entity/time effects and the
        corresponding identifiers are provided, the predictions include
        the estimated fixed effects.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Regressor matrix.
        entity_ids : array-like, shape (n,), optional
            Entity identifiers.  Required to include entity effects in
            the prediction.
        time_ids : array-like, shape (n,), optional
            Time-period identifiers.  Required to include time effects
            in the prediction.

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

        # Add entity effects via vectorized lookup
        if self._entity_effects_map and entity_ids is not None:
            ent_arr = np.asarray(entity_ids).ravel()
            ent_effects = np.vectorize(
                self._entity_effects_map.get, otypes=[np.float64]
            )(ent_arr, 0.0)
            y_pred = y_pred + ent_effects

        # Add time effects via vectorized lookup
        if self._time_effects_map and time_ids is not None:
            time_arr = np.asarray(time_ids).ravel()
            time_effects = np.vectorize(
                self._time_effects_map.get, otypes=[np.float64]
            )(time_arr, 0.0)
            y_pred = y_pred + time_effects

        return y_pred

    def summary(self):
        """Print and return a structured coefficient summary.

        Returns
        -------
        PanelSummary
            Dataclass with all model results.  Also prints a formatted
            table to stdout for interactive use.
        """
        self._check_is_fitted()

        k = len(self._params)
        feat_names = [f'x{i+1}' for i in range(k)]

        s = PanelSummary(
            model_type='PanelOLS',
            nobs=self.nobs,
            df_resid=self.df_resid,
            coef=self._params,
            bse=self.bse_,
            tvalues=self.tvalues_,
            pvalues=self.pvalues_,
            conf_int=self.conf_int_,
            feature_names=feat_names,
            rsquared_within=self.rsquared_within,
            cov_type=self.cov_type,
            entity_effects=self.entity_effects,
            time_effects=self.time_effects,
            alpha=self.alpha,
        )
        print(s)
        return s

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({
            'entity_effects': self.entity_effects,
            'time_effects': self.time_effects,
            'cov_type': self.cov_type,
            'alpha': self.alpha,
        })
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key in ('entity_effects', 'time_effects', 'cov_type', 'alpha'):
            if key in params:
                setattr(self, key, params.pop(key))
        super().set_params(**params)
        return self
