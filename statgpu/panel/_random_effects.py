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

import warnings
from typing import Optional, Union

import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _get_torch_device_str, _torch_dev, _to_float_scalar, _to_numpy, xp_astype, xp_zeros, xp_cholesky_solve

from statgpu.panel._utils import PanelSummary, within_transform, group_means, group_sizes


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
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha

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
        self._scale = None

    def fit(self, X, y, entity_ids=None, time_ids=None):
        """Fit the random effects model.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Regressor matrix.
        y : array-like, shape (n,)
            Outcome vector.
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
        self._backend_name = backend_name  # store for inference
        xp = backend.xp

        # Convert inputs
        y_arr = xp_astype(self._to_array(y, backend=backend_name).ravel(), xp.float64, xp)
        X_arr = xp_astype(self._to_array(X, backend=backend_name), xp.float64, xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        entity_arr = self._to_array(entity_ids, backend=backend_name).ravel()
        n, k = X_arr.shape
        self.nobs = n

        # Validate shapes
        if y_arr.shape[0] != n:
            raise ValueError(
                f"y has {y_arr.shape[0]} observations but X has {n} rows"
            )
        if entity_arr.shape[0] != n:
            raise ValueError(
                f"entity_ids has {entity_arr.shape[0]} observations but X has {n} rows"
            )

        # --- Step 1: Between estimation (group means) ---
        y_bar_i = group_means(y_arr, entity_arr, xp=xp)
        X_bar_i = xp.zeros_like(X_arr)
        for j in range(k):
            X_bar_i[:, j] = group_means(X_arr[:, j], entity_arr, xp=xp)

        # Extract unique group means for between estimation
        # Use first occurrence index to get one row per entity
        entity_np = _to_numpy(entity_arr).ravel()
        unique_entities, first_idx = np.unique(entity_np, return_index=True)
        n_groups = len(unique_entities)
        first_idx_dev = xp.asarray(first_idx, dtype=xp.int64)
        y_bar_unique = y_bar_i[first_idx_dev]
        X_bar_unique = X_bar_i[first_idx_dev]

        # Between OLS: beta_between = (X_bar'X_bar)^{-1} X_bar' y_bar
        XtX_b = X_bar_unique.T @ X_bar_unique
        Xty_b = X_bar_unique.T @ y_bar_unique
        try:
            beta_between = xp.linalg.solve(XtX_b, Xty_b)
        except _LINALG_ERRORS:
            beta_between = xp.linalg.pinv(XtX_b) @ Xty_b

        # Between residuals (using unique group means for correct RSS)
        resid_between = y_bar_unique - X_bar_unique @ beta_between
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
        except _LINALG_ERRORS:
            beta_within = xp.linalg.pinv(XtX_w) @ Xty_w

        resid_within = y_within - X_within @ beta_within
        rss_within = float(xp.sum(resid_within ** 2))

        # --- Step 3: Variance components ---
        unique_entities = xp.unique(entity_arr)
        n_entities = len(unique_entities)
        T_i = group_sizes(entity_arr, xp=xp)
        T_i_np = _to_numpy(T_i)  # needed for theta computation below

        # Harmonic mean of group sizes: one value per entity, not per observation.
        # T_i_np is per-observation (each entity's size repeated T_i times).
        # Get one size per entity via unique entity IDs + first occurrence.
        entity_np = _to_numpy(entity_arr).ravel()
        _, first_idx = np.unique(entity_np, return_index=True)
        per_entity_sizes = T_i_np[first_idx]
        T_bar = float(n_entities) / float(np.sum(1.0 / per_entity_sizes))

        # df for within residuals: n*T - k - (n_entities - 1)
        df_within = n - k - (n_entities - 1)
        if df_within <= 0:
            raise ValueError(
                f"Not enough observations for within df: n={n}, k={k}, "
                f"n_entities={n_entities}, df_within={df_within}"
            )

        sigma2_e = rss_within / df_within
        # Swamy-Arora: sigma2_a = max(0, (s_b^2 - sigma2_e) / T_bar)
        # where s_b^2 = RSS_between / (G - k) and T_bar is harmonic mean
        df_between = n_entities - k
        if df_between <= 0:
            warnings.warn(
                f"Between estimator under-identified: n_entities={n_entities} <= k={k}. "
                f"Variance component sigma2_a may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            df_between = max(df_between, 1)
        s_b_sq = rss_between / df_between
        sigma2_a_raw = (s_b_sq - sigma2_e) / T_bar
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

        # Weighted average of theta by number of entities at each group size
        entity_counts = {}
        for Ti in T_i_unique:
            entity_counts[Ti] = int(np.sum(T_i_np[first_idx] == Ti))
        total_entities = sum(entity_counts.values())
        self.theta_ = sum(
            theta_map[Ti] * entity_counts[Ti] / total_entities
            for Ti in T_i_unique
        )

        # Transformed variables: y* = y - theta * y_bar
        y_star = y_arr - theta_arr * y_bar_i
        X_star = xp.zeros_like(X_arr)
        for j in range(k):
            X_star[:, j] = X_arr[:, j] - theta_arr * X_bar_i[:, j]

        # --- Step 5: OLS on transformed data ---
        XtX_s = X_star.T @ X_star
        Xty_s = X_star.T @ y_star
        try:
            beta_gls = xp_cholesky_solve(XtX_s, Xty_s, xp)
        except _LINALG_ERRORS:
            beta_gls = xp.linalg.solve(XtX_s, Xty_s)

        resid_gls = y_star - X_star @ beta_gls
        df_resid = n - k
        self.df_resid = df_resid
        self._scale = _to_float_scalar(xp.sum(resid_gls ** 2)) / df_resid

        # --- Step 6: Inference — all on device ---
        self._compute_inference_on_device(xp, X_star, beta_gls, resid_gls)

        # Single transfer of final results
        self._params = _to_numpy(beta_gls).ravel()
        self.coef_ = self._params

        self._fitted = True
        return self

    def _compute_inference_on_device(self, xp, X, coef, resid):
        """Compute SE/t/p/CI with matrix ops on device, only final vectors to CPU."""
        from statgpu.inference._distributions_backend import get_distribution

        n, k = X.shape
        df = self.df_resid
        alpha = self.alpha

        # XtX_inv on device
        XtX = X.T @ X
        try:
            XtX_inv = xp.linalg.inv(XtX)
        except _LINALG_ERRORS:
            XtX_inv = xp.linalg.pinv(XtX)

        # cov_params = scale * (X'X)^{-1} on device
        cov_params = self._scale * XtX_inv
        bse_dev = xp.sqrt(xp.maximum(xp.diag(cov_params), 0.0))

        # t-values on device
        _eps = xp.finfo(xp.float64).tiny if hasattr(xp, 'finfo') else 2.2e-308
        tvalues_dev = coef / xp.maximum(bse_dev, _eps)
        abs_t = xp.abs(tvalues_dev)

        # p-values via backend-agnostic inference framework — on device
        t_dist = get_distribution("t", backend=self._backend_name)
        pvalues_dev = 2.0 * t_dist.sf(abs_t, float(df))
        t_crit = float(t_dist.isf(xp.asarray([alpha / 2.0]), float(df))[0])

        # Final transfer: only k-length vectors to CPU for storage
        bse_np = _to_numpy(bse_dev).ravel()
        tvalues_np = _to_numpy(tvalues_dev).ravel()
        coef_np = _to_numpy(coef).ravel()
        pvalues_np = _to_numpy(pvalues_dev).ravel()

        self.bse_ = bse_np
        self.tvalues_ = tvalues_np
        self.pvalues_ = pvalues_np
        self.conf_int_ = np.column_stack([
            coef_np - t_crit * bse_np,
            coef_np + t_crit * bse_np,
        ])

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
            model_type='RandomEffects',
            nobs=self.nobs,
            df_resid=self.df_resid,
            coef=self._params,
            bse=self.bse_,
            tvalues=self.tvalues_,
            pvalues=self.pvalues_,
            conf_int=self.conf_int_,
            feature_names=feat_names,
            variance_components=self.variance_components_,
            theta=self.theta_,
            alpha=self.alpha,
        )
        print(s)
        return s

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({
            'alpha': self.alpha,
        })
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        if 'alpha' in params:
            self.alpha = params.pop('alpha')
        super().set_params(**params)
        return self
