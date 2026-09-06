"""Backend-native post-selection OLS/WLS inference for sparse linear models."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model._gaussian_inference import (
    _as_backend_array,
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
# should not silently turn solver-scale numerical dust into selected variables.
_POST_SELECTION_ACTIVE_TOL = 1e-10


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


def _normal_nonrobust_inference(
    X_work,
    params_native,
    resid_work,
    scale_native,
    *,
    backend_name: str,
    selected_device: str,
):
    """Preserve the pre-migration normal/z post-selection reporting contract.

    The old unified ``cpu_ols`` / ``gpu_ols`` path performed an active-set
    unpenalized refit and reported z/normal inference.  The API migration keeps
    that statistical definition while moving the same numerical work to the
    fit-resolved NumPy/CuPy/Torch backend.
    """
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
        distribution="normal",
        alpha=0.05,
        backend=backend_name,
        device=selected_device,
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


def compute_post_selection_ols_inference(model, X, y, sample_weight=None):
    """Populate heuristic active-set OLS/WLS inference on the fitted backend.

    The sparse penalized fit chooses the active set. This function then refits an
    unpenalized least-squares model on exactly those columns and computes the
    requested covariance/reference-distribution inference there. The intervals
    remain a post-selection diagnostic; they do not account for data-driven
    active-set selection.

    Numerical work is performed on the backend/device recorded by the successful
    penalized fit. A NumPy reporting snapshot is taken only after parameter,
    covariance, reference-distribution, p-value, and confidence-interval work is
    complete.
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
    if k >= n and k > 0:
        raise ValueError(
            "post_selection_ols requires positive residual degrees of freedom; "
            f"selected design has {k} parameters for {n} observations."
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
        resolved_cov_type = str(getattr(model, "_cov_type", "nonrobust"))
        resolved_distribution = "normal"
        numerical_metadata = {
            "numerical_backend": backend_name,
            "numerical_device": selected_device,
            "reporting_backend": "numpy",
            "reporting_boundary": "post_numerical_inference",
        }
    else:
        XtX = X_work.T @ X_work
        Xty = X_work.T @ y_work
        params_native = _inverse_or_pinv(XtX, backend_name) @ Xty
        resid_native = y_native - X_design @ params_native
        resid_work = y_work - X_work @ params_native
        df_resid = n - k
        scale_native = xp.sum(resid_work * resid_work) / float(df_resid)
        cov_type = str(getattr(model, "_cov_type", "nonrobust")).lower()
        hac_maxlags = getattr(model, "_hac_maxlags", None)

        if cov_type == "nonrobust":
            bse_sel, stat_sel, pvalues_sel, ci_sel = _normal_nonrobust_inference(
                X_work,
                params_native,
                resid_work,
                scale_native,
                backend_name=backend_name,
                selected_device=selected_device,
            )
            params_sel = np.asarray(_to_numpy(params_native), dtype=np.float64)
            numerical_metadata = {
                "numerical_backend": backend_name,
                "numerical_device": selected_device,
                "reporting_backend": "numpy",
                "reporting_boundary": "post_numerical_inference",
            }
            resolved_cov_type = "nonrobust"
            resolved_distribution = "normal"
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
            resolved_distribution = "normal"

    full_dim = p_full + int(bool(model._effective_intercept))
    # Preserve the fitted penalized values for coordinates that were not
    # selected, matching the pre-migration reporting contract. Crucially, those
    # coordinates did not receive an OLS/WLS refit, so their inferential fields
    # are NaN rather than fake zero-variance [0, 0] intervals.
    params = coef_penalized.copy()
    if model._effective_intercept:
        params = np.concatenate([[float(model.intercept_)], params])
    bse = np.full(full_dim, np.nan, dtype=np.float64)
    statistic = np.full(full_dim, np.nan, dtype=np.float64)
    pvalues = np.full(full_dim, np.nan, dtype=np.float64)
    conf_int = np.full((full_dim, 2), np.nan, dtype=np.float64)

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

    # Reporting boundary: all numerical inference above has completed.
    model._X_design = np.asarray(_to_numpy(X_design), dtype=np.float64)
    model._y = np.asarray(_to_numpy(y_native), dtype=np.float64).reshape(-1)
    model._resid = np.asarray(_to_numpy(resid_native), dtype=np.float64).reshape(-1)
    model._scale = float(np.asarray(_to_numpy(scale_native), dtype=np.float64))
    model._df_resid = int(df_resid)
    model._nobs = n

    metadata = {
        **numerical_metadata,
        "heuristic_post_selection": True,
        "resolved_method": "post_selection_ols",
        "requested_method": str(getattr(model, "inference_method", "post_selection_ols")),
        "n_selected": int(selected_idx.size),
        "selected_feature_indices": selected_idx.tolist(),
        "active_set_tolerance": _POST_SELECTION_ACTIVE_TOL,
        "sample_weighted": sample_weight is not None,
        "compatibility_reference_distribution": "normal",
    }
    result = ParameterInferenceResult(
        method="post_selection_ols",
        feature_names=model._inference_feature_names(),
        params=params,
        bse=bse,
        statistic=statistic,
        statistic_name="z",
        pvalues=pvalues,
        conf_int=conf_int,
        cov_type=resolved_cov_type,
        distribution=resolved_distribution,
        df=None,
        metadata=metadata,
    )
    result.apply_to(model)
    return result


__all__ = ["compute_post_selection_ols_inference"]
