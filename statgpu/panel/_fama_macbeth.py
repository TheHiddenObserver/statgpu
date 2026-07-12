"""Fama-MacBeth two-pass regression with backend-native core linear algebra."""

from __future__ import annotations

__all__ = ["FamaMacBeth"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import (
    _LINALG_ERRORS,
    _get_xp,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_ones,
)
from statgpu.covariance._empirical import _detect_backend
from statgpu.panel._utils import PanelSummary


def _stack(values, xp, axis=0):
    return xp.stack(values, dim=axis) if xp.__name__ == "torch" else xp.stack(values, axis=axis)


def _index_array(indices, xp, ref):
    return xp_asarray(
        np.asarray(indices, dtype=np.int64), dtype=xp.int64, xp=xp, ref_arr=ref
    )


def _finite_all(x, xp):
    return bool(_to_float_scalar(xp.all(xp.isfinite(x))))


class FamaMacBeth(BaseEstimator):
    """Fama-MacBeth two-pass regression estimator.

    Formula parsing and time-label factorization are CPU metadata operations.
    Cross-sectional regressions, coefficient aggregation, and HAC covariance are
    evaluated on the selected NumPy, CuPy, or Torch backend.
    """

    def __init__(
        self,
        cov_type: str = "newey-west",
        bandwidth: Optional[int] = None,
        alpha: float = 0.05,
        min_obs_per_period: int = 1,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = str(cov_type).lower()
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.min_obs_per_period = min_obs_per_period
        if self.cov_type not in ("nonrobust", "newey-west"):
            raise ValueError("cov_type must be 'nonrobust' or 'newey-west'")

    def _validate_parameters(self):
        if self.cov_type not in ("nonrobust", "newey-west"):
            raise ValueError("cov_type must be 'nonrobust' or 'newey-west'")
        if self.bandwidth is not None:
            if (
                isinstance(self.bandwidth, bool)
                or not isinstance(self.bandwidth, (int, np.integer))
                or int(self.bandwidth) < 0
            ):
                raise ValueError("bandwidth must be a non-negative integer or None")
        if not np.isfinite(float(self.alpha)) or not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        if (
            isinstance(self.min_obs_per_period, bool)
            or not isinstance(self.min_obs_per_period, (int, np.integer))
            or int(self.min_obs_per_period) < 1
        ):
            raise ValueError("min_obs_per_period must be a positive integer")

    def _prepare_backend_arrays(self, X, y):
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        ref = None
        if backend_name == "torch":
            import torch

            if isinstance(X, torch.Tensor):
                ref = X
            else:
                dev = self._get_compute_device()
                target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                ref = torch.empty(0, dtype=torch.float64, device=target)
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=ref)
        y_arr = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or int(X_arr.shape[0]) == 0 or int(X_arr.shape[1]) == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if int(y_arr.shape[0]) != int(X_arr.shape[0]):
            raise ValueError("X and y must have the same number of observations")
        if not _finite_all(X_arr, xp) or not _finite_all(y_arr, xp):
            raise ValueError("X and y must contain only finite values")
        return backend_name, xp, X_arr, y_arr

    def fit(self, X=None, y=None, time_ids=None, formula=None, data=None):
        self._validate_parameters()
        if time_ids is None:
            raise ValueError("time_ids is required for FamaMacBeth")

        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit

        (
            y_data,
            X_data,
            self._design_info,
            self._feature_names,
            self._formula_has_intercept,
            _fe_eids,
            _fe_tids,
            _fe_entity,
            _fe_time,
        ) = _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)
        if formula is not None:
            time_ids = _align_formula_side_array(
                time_ids, self._design_info, len(y_data), "time_ids"
            )

        backend_name, xp, X_arr, y_arr = self._prepare_backend_arrays(X_data, y_data)
        n_orig = int(X_arr.shape[0])
        tids_np = np.asarray(_to_numpy(time_ids)).ravel()
        if tids_np.shape[0] != n_orig:
            raise ValueError("time_ids must have one entry per observation")
        if np.any(np.asarray([x is None for x in tids_np], dtype=bool)):
            raise ValueError("time_ids must not contain missing values")

        _, time_codes = np.unique(tids_np, return_inverse=True)
        counts = np.bincount(time_codes)
        intercept = xp_ones((n_orig, 1), xp.float64, xp, X_arr)
        X_design = xp.cat([intercept, X_arr], dim=1) if xp.__name__ == "torch" else xp.concatenate([intercept, X_arr], axis=1)
        k = int(X_design.shape[1])

        betas_list = []
        for code, n_t in enumerate(counts):
            if int(n_t) < int(self.min_obs_per_period) or int(n_t) < k + 1:
                continue
            idx = _index_array(np.flatnonzero(time_codes == code), xp, X_design)
            X_t = X_design[idx]
            y_t = y_arr[idx]
            try:
                beta_t = xp.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)
            except _LINALG_ERRORS:
                beta_t = xp.linalg.pinv(X_t) @ y_t
            betas_list.append(beta_t)

        if not betas_list:
            raise ValueError("No time periods with enough observations")
        betas = _stack(betas_list, xp, axis=0)
        T = int(betas.shape[0])
        if T < 2:
            raise ValueError("FamaMacBeth requires at least 2 time periods after filtering")

        avg_beta = xp.mean(betas, axis=0)
        beta_centered = betas - avg_beta
        if self.cov_type == "nonrobust":
            covariance = (beta_centered.T @ beta_centered) / float(T - 1)
            cov_params = covariance / float(T)
        else:
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
            bandwidth = max(0, min(int(bandwidth), T - 1))
            long_run = beta_centered.T @ beta_centered / float(T)
            for lag in range(1, bandwidth + 1):
                weight = 1.0 - lag / float(bandwidth + 1)
                gamma_lag = beta_centered[lag:].T @ beta_centered[:-lag] / float(T)
                long_run = long_run + weight * (gamma_lag + gamma_lag.T)
            cov_params = long_run / float(T)

        diagonal = xp.diag(cov_params)
        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
        tvalues = avg_beta / bse
        df = T - 1

        from statgpu.inference._distributions_backend import get_distribution

        dist_name = "norm" if self.cov_type == "newey-west" else "t"
        distribution = get_distribution(dist_name, backend="numpy")
        pvalues_py = []
        for value in xp.abs(tvalues):
            scalar = _to_float_scalar(value)
            if dist_name == "t":
                pvalues_py.append(2.0 * _to_float_scalar(distribution.sf(scalar, df)))
            else:
                pvalues_py.append(2.0 * _to_float_scalar(distribution.sf(scalar)))
        pvalues = xp_asarray(pvalues_py, dtype=xp.float64, xp=xp, ref_arr=avg_beta)
        if dist_name == "t":
            critical = _to_float_scalar(distribution.isf(float(self.alpha) / 2.0, df))
        else:
            critical = _to_float_scalar(distribution.isf(float(self.alpha) / 2.0))
        conf_int = _stack(
            [avg_beta - critical * bse, avg_beta + critical * bse], xp, axis=1
        )

        self.coef_ = avg_beta
        self.bse_ = bse
        self.tvalues_ = tvalues
        self.pvalues_ = pvalues
        self.conf_int_ = conf_int
        self.betas_ = betas
        self.cov_params_ = cov_params
        self.nobs = n_orig
        self.n_periods = T
        self.df_resid = df
        self._backend_name = backend_name
        self._xp = xp
        self._fit_ref_ = X_arr
        self._fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict

        X_data = _formula_predict(
            X,
            getattr(self, "_design_info", None),
            getattr(self, "_formula_has_intercept", None),
            model_has_intercept=True,
        )
        xp = self._xp
        X_arr = xp_asarray(X_data, dtype=xp.float64, xp=xp, ref_arr=self._fit_ref_)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or int(X_arr.shape[1]) + 1 != int(self.coef_.shape[0]):
            raise ValueError("X has an incompatible feature count")
        intercept = xp_ones((int(X_arr.shape[0]), 1), xp.float64, xp, X_arr)
        X_design = xp.cat([intercept, X_arr], dim=1) if xp.__name__ == "torch" else xp.concatenate([intercept, X_arr], axis=1)
        return X_design @ self.coef_

    def summary(self):
        self._check_is_fitted()
        from statgpu.panel._formula import _get_feature_names

        feature_names = _get_feature_names(
            getattr(self, "_feature_names", None), len(self.coef_), prefix="x"
        )
        return PanelSummary(
            model_type="FamaMacBeth",
            cov_type=self.cov_type,
            coef=np.asarray(_to_numpy(self.coef_)),
            bse=np.asarray(_to_numpy(self.bse_)),
            tvalues=np.asarray(_to_numpy(self.tvalues_)),
            pvalues=np.asarray(_to_numpy(self.pvalues_)),
            conf_int=np.asarray(_to_numpy(self.conf_int_)),
            nobs=self.nobs,
            df_resid=self.df_resid,
            alpha=self.alpha,
            feature_names=feature_names,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            cov_type=self.cov_type,
            bandwidth=self.bandwidth,
            alpha=self.alpha,
            min_obs_per_period=self.min_obs_per_period,
        )
        return params

    def set_params(self, **params):
        for key in ["cov_type", "bandwidth", "alpha", "min_obs_per_period"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
