"""Pooled OLS panel data model with GPU acceleration."""

from __future__ import annotations

__all__ = ["PooledOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._utils import factorize_panel_labels


def _panel_lstsq(X, y, xp):
    """Return least-squares coefficients and the effective design rank."""
    if getattr(xp, "__name__", "") == "torch":
        params = xp.linalg.pinv(X) @ y
        rank = int(_to_float_scalar(xp.linalg.matrix_rank(X)))
        return params, rank
    try:
        result = xp.linalg.lstsq(X, y, rcond=None)
        params = result[0]
        rank = int(_to_float_scalar(result[2]))
        return params, rank
    except (TypeError, AttributeError, np.linalg.LinAlgError):
        params = xp.linalg.pinv(X) @ y
        rank = int(_to_float_scalar(xp.linalg.matrix_rank(X)))
        return params, rank


class PooledOLS(BasePanelModel):
    """Pooled OLS estimator for panel data.

    Runs OLS on the pooled (stacked) panel data without any demeaning
    or transformation. Supports the existing nonrobust, HC1 robust,
    clustered, and HAC covariance estimators.
    """

    def __init__(
        self,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = cov_type.lower()
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel
        if self.cov_type not in ("nonrobust", "robust", "clustered", "hac"):
            raise ValueError(
                "cov_type must be 'nonrobust', 'robust', 'clustered', or 'hac'"
            )

    def fit(self, X=None, y=None, cluster=None, time_index=None, formula=None, data=None):
        """Fit the pooled OLS model."""
        (
            y_data,
            X_data,
            _fe_eids,
            _fe_tids,
            _fe_entity,
            _fe_time,
            aligned,
        ) = self._panel_prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=True,
            side_arrays={"cluster": cluster, "time_index": time_index},
        )
        cluster = aligned["cluster"]
        time_index = aligned["time_index"]

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)

        # HAC depends on temporal ordering. Metadata may remain on CPU, while
        # the numerical arrays are reordered on their selected backend.
        if self._cov_type == "hac" and time_index is not None:
            time_values = np.asarray(_to_numpy(time_index))
            if time_values.ndim != 1 or time_values.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    "time_index must be one-dimensional with length n_samples"
                )
            order_np = np.argsort(time_values, kind="stable")
            order = xp_asarray(order_np, dtype=xp.int64, xp=xp, ref_arr=X_arr)
            X_arr = X_arr[order]
            y_arr = y_arr[order]

        n = X_arr.shape[0]
        ones = xp.ones((n, 1), dtype=xp.float64)
        if hasattr(X_arr, "is_cuda"):
            ones = ones.to(device=X_arr.device)
        X_arr = xp.concatenate([ones, X_arr], axis=1)

        n, _ = X_arr.shape
        params, rank = _panel_lstsq(X_arr, y_arr, xp)
        df_resid = n - rank
        if df_resid <= 0:
            raise ValueError(
                f"positive residual degrees of freedom required; n={n}, rank={rank}"
            )
        resid = y_arr - X_arr @ params
        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid

        cluster_for_cov = cluster
        if self._cov_type == "clustered":
            if cluster is None:
                raise ValueError("cluster is required for cov_type='clustered'")
            # Preserve the existing validation/factorization behavior before
            # delegating to the shared covariance registry.
            cluster_for_cov, _ = factorize_panel_labels(
                cluster,
                xp,
                ref_arr=X_arr,
                name="cluster",
                expected_n=n,
            )

        self._panel_store_ols_inference(
            X_arr,
            resid,
            params,
            scale=scale,
            df_resid=df_resid,
            backend=backend,
            cov_type=self._cov_type,
            cluster=cluster_for_cov,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            allowed=("nonrobust", "robust", "clustered", "hac"),
            hc1_correction=n / df_resid if self._cov_type == "robust" else None,
            distribution_df=df_resid,
            # PooledOLS historically used sqrt(diag(V)) without clipping.
            diag_floor=None,
        )

        y_mean = xp.mean(y_arr)
        ss_tot = _to_float_scalar(xp.sum((y_arr - y_mean) ** 2))
        ss_res = _to_float_scalar(xp.sum(resid * resid))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.nobs = n
        self.rank_ = rank
        self.df_resid = df_resid
        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return self._panel_predict_linear(
            X,
            model_has_intercept=True,
            add_intercept=True,
            return_numpy=True,
        )

    def summary(self):
        """Return a summary object."""
        return self._panel_summary(
            model_type="PooledOLS",
            cov_type=self._cov_type,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)
