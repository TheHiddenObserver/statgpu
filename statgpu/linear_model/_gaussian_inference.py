"""Shared Gaussian linear-model inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from statgpu.backends import _to_numpy
from statgpu.inference._results import GaussianInferenceResult


@dataclass
class GaussianFitState:
    X_design: np.ndarray
    y: np.ndarray
    resid: np.ndarray
    scale: np.ndarray | float
    nobs: int
    df_resid: int
    params: np.ndarray


def validate_cov_type(cov_type: str) -> str:
    cov_type = str(cov_type).lower()
    if cov_type not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        raise ValueError(
            "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'"
        )
    return cov_type


def validate_hac_maxlags(hac_maxlags: Optional[int]) -> Optional[int]:
    if hac_maxlags is not None and int(hac_maxlags) < 0:
        raise ValueError("hac_maxlags must be a non-negative integer or None")
    return None if hac_maxlags is None else int(hac_maxlags)


def resolve_hac_maxlags(n_obs: int, hac_maxlags: Optional[int]) -> int:
    if n_obs <= 1:
        return 0
    if hac_maxlags is None:
        maxlags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    else:
        maxlags = int(hac_maxlags)
    return max(0, min(maxlags, n_obs - 1))


def build_gaussian_fit_state(X, y, coef, intercept, fit_intercept: bool) -> GaussianFitState:
    X_np = np.asarray(_to_numpy(X), dtype=float)
    y_np = np.asarray(_to_numpy(y), dtype=float)
    if y_np.ndim == 2 and y_np.shape[1] == 1:
        y_np = y_np.ravel()

    coef_np = np.asarray(coef, dtype=float)
    intercept_np = np.asarray(intercept, dtype=float)
    if fit_intercept:
        X_design = np.column_stack([np.ones(X_np.shape[0], dtype=X_np.dtype), X_np])
        if coef_np.ndim == 1:
            params = np.concatenate([[float(intercept_np)], coef_np])
        else:
            params = np.vstack([intercept_np.reshape(1, -1), coef_np])
    else:
        X_design = X_np
        params = coef_np.copy()

    y_pred = X_np @ coef_np
    if fit_intercept:
        y_pred = y_pred + intercept_np
    resid = y_np - y_pred
    nobs = X_design.shape[0]
    df_resid = nobs - X_design.shape[1]
    rss = np.sum(resid ** 2, axis=0)
    scale = rss / df_resid if df_resid > 0 else np.full_like(rss, np.nan, dtype=float)
    if np.ndim(scale) == 0:
        scale = float(scale)
    return GaussianFitState(
        X_design=X_design,
        y=y_np,
        resid=resid,
        scale=scale,
        nobs=nobs,
        df_resid=df_resid,
        params=params,
    )


def _hac_meat_numpy(scores: np.ndarray, maxlags: int) -> np.ndarray:
    n = scores.shape[0]
    meat = scores.T @ scores
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    return meat


def robust_covariance_numpy(
    X: np.ndarray,
    resid: np.ndarray,
    bread_inv: np.ndarray,
    cov_type: str,
    hac_maxlags: Optional[int] = None,
) -> np.ndarray:
    cov_type = validate_cov_type(cov_type)
    n, k = X.shape
    resid = np.asarray(resid, dtype=float)

    if cov_type == "hac":
        scores = X * resid[:, None]
        maxlags = resolve_hac_maxlags(n, hac_maxlags)
        meat = _hac_meat_numpy(scores, maxlags)
        return bread_inv @ meat @ bread_inv

    leverage = None
    if cov_type in ("hc2", "hc3"):
        leverage = np.sum(X * (X @ bread_inv), axis=1)
        leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)

    if cov_type == "hc2":
        omega = resid ** 2 / np.maximum(1.0 - leverage, 1e-12)
    elif cov_type == "hc3":
        omega = resid ** 2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    else:
        omega = resid ** 2

    meat = X.T @ (X * omega[:, None])
    if cov_type == "hc1" and n > k:
        meat *= n / (n - k)
    return bread_inv @ meat @ bread_inv


def compute_gaussian_inference(
    X_design,
    params,
    resid,
    scale,
    df_resid: int,
    cov_type: str,
    hac_maxlags: Optional[int] = None,
    ridge_alpha: float = 0.0,
    alpha: float = 0.05,
    ridge_penalize_intercept: Optional[bool] = None,
) -> Optional[GaussianInferenceResult]:
    if X_design is None or scale is None:
        return None
    scale_arr = np.asarray(scale, dtype=float)
    if np.any(np.isnan(scale_arr)):
        return None

    X = np.asarray(_to_numpy(X_design), dtype=float)
    params_arr = np.asarray(_to_numpy(params), dtype=float)
    resid_arr = np.asarray(_to_numpy(resid), dtype=float)
    n, k = X.shape
    XtX = X.T @ X
    penalty_diag = np.zeros(k, dtype=float)
    if ridge_alpha:
        penalty_diag[:] = float(ridge_alpha)
        if ridge_penalize_intercept is None:
            unpenalized_intercept = k > 0 and np.allclose(X[:, 0], X[0, 0])
        else:
            unpenalized_intercept = k > 0 and not bool(ridge_penalize_intercept)
        if unpenalized_intercept:
            penalty_diag[0] = 0.0
    bread = XtX + np.diag(penalty_diag)
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)

    if params_arr.ndim == 2:
        n_targets = params_arr.shape[1]
        bse_out = np.empty_like(params_arr)
        t_out = np.empty_like(params_arr)
        p_out = np.empty_like(params_arr)
        ci_out = np.empty((params_arr.shape[0], n_targets, 2), dtype=float)
        for j in range(n_targets):
            result = compute_gaussian_inference(
                X,
                params_arr[:, j],
                resid_arr[:, j],
                scale_arr.reshape(-1)[j],
                df_resid,
                cov_type,
                hac_maxlags=hac_maxlags,
                ridge_alpha=ridge_alpha,
                alpha=alpha,
                ridge_penalize_intercept=ridge_penalize_intercept,
            )
            if result is None:
                return None
            bse_out[:, j] = result.bse
            t_out[:, j] = result.tvalues
            p_out[:, j] = result.pvalues
            ci_out[:, j, :] = result.conf_int
        method = "classical" if validate_cov_type(cov_type) == "nonrobust" else "sandwich"
        distribution = "t" if validate_cov_type(cov_type) == "nonrobust" else "normal"
        return GaussianInferenceResult(
            params=params_arr,
            bse=bse_out,
            statistic=t_out,
            pvalues=p_out,
            conf_int=ci_out,
            cov_type=cov_type,
            distribution=distribution,
            df=df_resid,
            method=method,
            metadata={"ridge_alpha": float(ridge_alpha), "alpha": float(alpha)},
        )

    cov_type = validate_cov_type(cov_type)
    if cov_type == "nonrobust":
        cov_params = float(scale_arr) * bread_inv
        bse = np.sqrt(np.diag(cov_params))
        tvalues = params_arr / (bse + 1e-30)
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df_resid))
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
        conf_int = np.column_stack([
            params_arr - t_crit * bse,
            params_arr + t_crit * bse,
        ])
        return GaussianInferenceResult(
            params=params_arr,
            bse=bse,
            statistic=tvalues,
            pvalues=pvalues,
            conf_int=conf_int,
            cov_type=cov_type,
            distribution="t",
            df=df_resid,
            method="classical",
            metadata={"ridge_alpha": float(ridge_alpha), "alpha": float(alpha)},
        )

    cov_params = robust_covariance_numpy(
        X,
        resid_arr,
        bread_inv,
        cov_type,
        hac_maxlags=hac_maxlags,
    )
    bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
    tvalues = params_arr / (bse + 1e-30)
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    conf_int = np.column_stack([
        params_arr - z_crit * bse,
        params_arr + z_crit * bse,
    ])
    return GaussianInferenceResult(
        params=params_arr,
        bse=bse,
        statistic=tvalues,
        pvalues=pvalues,
        conf_int=conf_int,
        cov_type=cov_type,
        distribution="normal",
        df=df_resid,
        method="sandwich",
        metadata={"ridge_alpha": float(ridge_alpha), "alpha": float(alpha)},
    )
