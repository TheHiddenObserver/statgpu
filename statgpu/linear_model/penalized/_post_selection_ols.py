"""Backend-native post-selection OLS/WLS inference for sparse linear models."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model._gaussian_inference import (
    _as_backend_array,
    _compute_single_native,
    _diag,
    _inverse_or_pinv,
    _maximum,
    _namespace,
    _reference_inference,
    _sqrt,
    _stack,
    compute_gaussian_inference,
)

# Preserve the maintained pre-migration active-set boundary. An API cleanup
# should not silently change which solver-scale coefficients count as selected.
_POST_SELECTION_ACTIVE_TOL = 1e-15


def _selected_device(model, backend_name: str) -> str:
    selected = getattr(model, "_selected_backend_device", None)
    if backend_name == "numpy":
        return "cpu"
    label = str(selected or "")
    if backend_name == "cupy" and label.startswith("cuda:"):
        return label
    if backend_name == "torch" and (label == "cpu" or label.startswith("cuda:")):
        return label
    raise RuntimeError(
        "post_selection_ols is missing concrete executed-device provenance "
        f"for backend={backend_name!r}: {selected!r}"
    )


def _active_design(X_native, selected_idx, *, fit_intercept: bool, backend_name: str):
    n = int(X_native.shape[0])
    if backend_name == "torch":
        import torch

        index = torch.as_tensor(selected_idx, dtype=torch.long, device=X_native.device)
        features = X_native.index_select(1, index)
        if not fit_intercept:
            return features
        ones = torch.ones((n, 1), dtype=X_native.dtype, device=X_native.device)
        return torch.cat([ones, features], dim=1)

    if backend_name == "cupy":
        import cupy as cp

        with cp.cuda.Device(int(X_native.device.id)):
            index = cp.asarray(selected_idx, dtype=cp.int64)
            features = X_native[:, index]
            if not fit_intercept:
                return features
            ones = cp.ones((n, 1), dtype=X_native.dtype)
            return cp.concatenate([ones, features], axis=1)

    features = X_native[:, selected_idx]
    if not fit_intercept:
        return features
    return np.column_stack([np.ones(n, dtype=X_native.dtype), features])


def _matrix_rank(X_work, backend_name: str) -> int:
    """Return the active-design rank using the executed numerical backend."""
    if backend_name == "torch":
        import torch

        return int(torch.linalg.matrix_rank(X_work).item())
    if backend_name == "cupy":
        import cupy as cp

        return int(cp.linalg.matrix_rank(X_work).item())
    return int(np.linalg.matrix_rank(np.asarray(X_work)))


def _design_pinv(X_work, backend_name: str):
    """Compute a design-level Moore-Penrose inverse without squaring condition."""
    if backend_name == "torch":
        import torch

        return torch.linalg.pinv(X_work)
    if backend_name == "cupy":
        import cupy as cp

        return cp.linalg.pinv(X_work)
    return np.linalg.pinv(np.asarray(X_work))


def _classical_nonrobust_inference(
    X_work,
    params_native,
    scale_native,
    *,
    backend_name: str,
    selected_device: str,
    df_resid: int,
):
    """Compute the maintained classical Student-t post-selection report."""
    XtX = X_work.T @ X_work
    bread_inv = _inverse_or_pinv(XtX, backend_name)
    cov_params = scale_native * bread_inv
    bse_native = _sqrt(
        _maximum(_diag(cov_params, backend_name), 0.0, backend_name),
        backend_name,
    )
    statistic_native = params_native / (bse_native + 1e-30)
    pvalues_native, critical = _reference_inference(
        statistic_native,
        distribution="t",
        alpha=0.05,
        backend=backend_name,
        device=selected_device,
        df=df_resid,
    )
    conf_int_native = _stack(
        [
            params_native - critical * bse_native,
            params_native + critical * bse_native,
        ],
        backend_name,
        axis=1,
    )
    return (
        np.asarray(_to_numpy(bse_native), dtype=np.float64),
        np.asarray(_to_numpy(statistic_native), dtype=np.float64),
        np.asarray(_to_numpy(pvalues_native), dtype=np.float64),
        np.asarray(_to_numpy(conf_int_native), dtype=np.float64),
    )


def _rank_deficient_inference(
    X_work,
    params_native,
    resid_work,
    scale_native,
    *,
    backend_name: str,
    selected_device: str,
    df_resid: int,
    cov_type: str,
    hac_maxlags,
    design_pinv,
):
    """Use the design SVD for covariance as well as the refit coefficients."""
    XtX = X_work.T @ X_work
    # X^+ (X^+)^T == (X^T X)^+ mathematically, but computing it from the
    # design SVD avoids the condition-number squaring that broke exact
    # collinearity on maintained NumPy 1.26 environments.
    bread_inv = design_pinv @ design_pinv.T
    (
        bse_native,
        statistic_native,
        pvalues_native,
        conf_int_native,
        distribution,
        _method,
    ) = _compute_single_native(
        X_work,
        params_native,
        resid_work,
        scale_native,
        XtX=XtX,
        bread_inv=bread_inv,
        backend=backend_name,
        device=selected_device,
        df_resid=df_resid,
        cov_type=cov_type,
        hac_maxlags=hac_maxlags,
        ridge_alpha=0.0,
        alpha=0.05,
    )
    return (
        np.asarray(_to_numpy(bse_native), dtype=np.float64),
        np.asarray(_to_numpy(statistic_native), dtype=np.float64),
        np.asarray(_to_numpy(pvalues_native), dtype=np.float64),
        np.asarray(_to_numpy(conf_int_native), dtype=np.float64),
        str(distribution),
    )


def compute_post_selection_ols_inference(model, X, y, sample_weight=None):
    """Populate heuristic active-set OLS/WLS inference on the fitted backend.

    ``coef_`` and the estimator's generic fit diagnostics continue to describe
    the penalized prediction model.  The active-set refit has its own private
    diagnostic snapshot and supplies only the inference/reporting fields.
    """
    backend_name = str(getattr(model, "_selected_backend_name", "")).lower()
    if backend_name not in {"numpy", "cupy", "torch"}:
        raise RuntimeError(
            "post_selection_ols requires executed-backend provenance from the "
            "successful penalized fit; refusing to re-detect from raw input."
        )
    selected_device = _selected_device(model, backend_name)

    X_native = _as_backend_array(X, backend_name, device=selected_device)
    y_native = _as_backend_array(
        y, backend_name, like=X_native, device=selected_device
    ).reshape(-1)
    n = int(X_native.shape[0])
    p_full = int(X_native.shape[1])

    coef_penalized = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    if coef_penalized.shape[0] != p_full:
        raise RuntimeError(
            "post_selection_ols coefficient dimension does not match the fitted design."
        )
    selected_idx = np.flatnonzero(
        np.abs(coef_penalized) > _POST_SELECTION_ACTIVE_TOL
    )
    X_design = _active_design(
        X_native,
        selected_idx,
        fit_intercept=bool(model._effective_intercept),
        backend_name=backend_name,
    )

    xp = _namespace(backend_name)
    if sample_weight is None:
        X_work = X_design
        y_work = y_native
    else:
        sw_native = _as_backend_array(
            sample_weight, backend_name, like=X_native, device=selected_device
        ).reshape(-1)
        if int(sw_native.shape[0]) != n:
            raise ValueError("sample_weight must have length n_samples")
        sqrt_sw = xp.sqrt(sw_native)
        X_work = X_design * sqrt_sw.reshape(-1, 1)
        y_work = y_native * sqrt_sw

    k = int(X_work.shape[1])
    fit_rank = 0 if k == 0 else _matrix_rank(X_work, backend_name)
    if fit_rank >= n:
        raise ValueError(
            "post_selection_ols requires positive residual degrees of freedom; "
            f"selected design has effective rank {fit_rank} for {n} observations."
        )

    if k == 0:
        params_sel = np.empty((0,), dtype=np.float64)
        bse_sel = np.empty((0,), dtype=np.float64)
        stat_sel = np.empty((0,), dtype=np.float64)
        pvalues_sel = np.empty((0,), dtype=np.float64)
        ci_sel = np.empty((0, 2), dtype=np.float64)
        resid_native = y_native
        resid_work = y_work
        df_resid = n
        scale_native = xp.sum(resid_work * resid_work) / float(df_resid)
        resolved_cov_type = "nonrobust"
        resolved_distribution = "t"
        numerical_metadata = {
            "numerical_backend": backend_name,
            "numerical_device": selected_device,
            "reporting_backend": "numpy",
            "reporting_boundary": "post_numerical_inference",
        }
    else:
        XtX = X_work.T @ X_work
        Xty = X_work.T @ y_work
        rank_deficient = fit_rank < k
        design_pinv = _design_pinv(X_work, backend_name) if rank_deficient else None
        params_native = (
            design_pinv @ y_work
            if rank_deficient
            else _inverse_or_pinv(XtX, backend_name) @ Xty
        )
        resid_native = y_native - X_design @ params_native
        resid_work = y_work - X_work @ params_native
        df_resid = n - fit_rank
        scale_native = xp.sum(resid_work * resid_work) / float(df_resid)
        cov_type = str(getattr(model, "_cov_type", "nonrobust")).lower()
        hac_maxlags = getattr(model, "_hac_maxlags", None)

        if rank_deficient:
            (
                bse_sel,
                stat_sel,
                pvalues_sel,
                ci_sel,
                resolved_distribution,
            ) = _rank_deficient_inference(
                X_work,
                params_native,
                resid_work,
                scale_native,
                backend_name=backend_name,
                selected_device=selected_device,
                df_resid=df_resid,
                cov_type=cov_type,
                hac_maxlags=hac_maxlags,
                design_pinv=design_pinv,
            )
            params_sel = np.asarray(_to_numpy(params_native), dtype=np.float64)
            numerical_metadata = {
                "numerical_backend": backend_name,
                "numerical_device": selected_device,
                "reporting_backend": "numpy",
                "reporting_boundary": "post_numerical_inference",
            }
            resolved_cov_type = cov_type
        elif cov_type == "nonrobust":
            bse_sel, stat_sel, pvalues_sel, ci_sel = _classical_nonrobust_inference(
                X_work,
                params_native,
                scale_native,
                backend_name=backend_name,
                selected_device=selected_device,
                df_resid=df_resid,
            )
            params_sel = np.asarray(_to_numpy(params_native), dtype=np.float64)
            numerical_metadata = {
                "numerical_backend": backend_name,
                "numerical_device": selected_device,
                "reporting_backend": "numpy",
                "reporting_boundary": "post_numerical_inference",
            }
            resolved_cov_type = "nonrobust"
            resolved_distribution = "t"
        else:
            gaussian = compute_gaussian_inference(
                X_work,
                params_native,
                resid_work,
                scale_native,
                df_resid,
                cov_type,
                hac_maxlags=hac_maxlags,
                backend=backend_name,
                device=selected_device,
            )
            if gaussian is None:
                raise RuntimeError(
                    "post_selection_ols could not construct finite Gaussian inference "
                    "for the selected active set."
                )
            params_sel = np.asarray(gaussian.params, dtype=np.float64)
            bse_sel = np.asarray(gaussian.bse, dtype=np.float64)
            stat_sel = np.asarray(gaussian.statistic, dtype=np.float64)
            pvalues_sel = np.asarray(gaussian.pvalues, dtype=np.float64)
            ci_sel = np.asarray(gaussian.conf_int, dtype=np.float64)
            numerical_metadata = dict(gaussian.metadata)
            resolved_cov_type = gaussian.cov_type
            resolved_distribution = str(gaussian.distribution)

    full_dim = p_full + int(bool(model._effective_intercept))
    params = coef_penalized.copy()
    if model._effective_intercept:
        params = np.concatenate([[float(model.intercept_)], params])
    bse = np.zeros(full_dim, dtype=np.float64)
    statistic = np.zeros(full_dim, dtype=np.float64)
    pvalues = np.ones(full_dim, dtype=np.float64)
    conf_int = np.zeros((full_dim, 2), dtype=np.float64)

    if model._effective_intercept and k:
        params[0] = params_sel[0]
        bse[0] = bse_sel[0]
        statistic[0] = stat_sel[0]
        pvalues[0] = pvalues_sel[0]
        conf_int[0] = ci_sel[0]
        target = selected_idx + 1
        params[target] = params_sel[1:]
        bse[target] = bse_sel[1:]
        statistic[target] = stat_sel[1:]
        pvalues[target] = pvalues_sel[1:]
        conf_int[target] = ci_sel[1:]
    elif k:
        params[selected_idx] = params_sel
        bse[selected_idx] = bse_sel
        statistic[selected_idx] = stat_sel
        pvalues[selected_idx] = pvalues_sel
        conf_int[selected_idx] = ci_sel

    # Dedicated reporting snapshot for the auxiliary refit. Do not overwrite
    # _X_design/_y/_resid/_scale/_df_resid/_nobs: those generic fields describe
    # the penalized fitted estimator and drive rsquared/AIC/BIC/F diagnostics.
    post_X_design = np.asarray(_to_numpy(X_design), dtype=np.float64)
    post_y = np.asarray(_to_numpy(y_native), dtype=np.float64).reshape(-1)
    post_resid = np.asarray(_to_numpy(resid_native), dtype=np.float64).reshape(-1)
    post_scale = float(np.asarray(_to_numpy(scale_native), dtype=np.float64))
    model._post_selection_X_design = post_X_design
    model._post_selection_y = post_y
    model._post_selection_resid = post_resid
    model._post_selection_scale = post_scale
    model._post_selection_df_resid = int(df_resid)
    model._post_selection_nobs = n

    metadata = {
        **numerical_metadata,
        "heuristic_post_selection": True,
        "resolved_method": "post_selection_ols",
        "requested_method": str(getattr(model, "inference_method", "post_selection_ols")),
        "n_selected": int(selected_idx.size),
        "selected_feature_indices": selected_idx.tolist(),
        "active_set_tolerance": _POST_SELECTION_ACTIVE_TOL,
        "sample_weighted": sample_weight is not None,
        "inactive_inference_placeholders": True,
        "refit_parameter_count": int(k),
        "refit_rank": int(fit_rank),
        "refit_rank_deficient": bool(fit_rank < k),
        "refit_df_resid": int(df_resid),
        "refit_scale": post_scale,
        "refit_nobs": n,
    }
    statistic_name = "t" if resolved_distribution == "t" else "z"
    result = ParameterInferenceResult(
        method="post_selection_ols",
        feature_names=model._inference_feature_names(),
        params=params,
        bse=bse,
        statistic=statistic,
        statistic_name=statistic_name,
        pvalues=pvalues,
        conf_int=conf_int,
        cov_type=resolved_cov_type,
        distribution=resolved_distribution,
        df=float(df_resid) if resolved_distribution == "t" else None,
        metadata=metadata,
    )
    result.apply_to(model)
    return result


__all__ = ["compute_post_selection_ols_inference"]
