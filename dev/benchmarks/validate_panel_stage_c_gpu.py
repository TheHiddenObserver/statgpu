#!/usr/bin/env python3
"""Physical CuPy/Torch acceptance for Panel Tier-1 Stage C covariance.

This is a correctness/backend-provenance gate, not a timing benchmark. It
compares every newly supported covariance integration against NumPy on the same
aligned panel while proving that explicit CuPy/Torch CUDA requests really
execute without CPU fallback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.backends import _is_cupy_array, _is_torch_array, _to_numpy
from statgpu.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
    clustered_covariance,
    driscoll_kraay_covariance,
)
from statgpu.panel._covariance import ols_covariance


CORRECTNESS_SCHEMA_VERSION = 2


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dataset(seed=20260811, *, unbalanced=True):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 10, 8
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.4, size=n_entities), n_times)
    tau = np.tile(np.linspace(-0.2, 0.25, n_times), n_entities)
    y = 0.8 * X[:, 0] - 0.35 * X[:, 1] + alpha + 0.25 * tau
    y += rng.normal(scale=0.22, size=entity.size)
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 10, 19, 37, 58, 71]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    cluster_a = np.asarray([f"firm-{v}" for v in entity], dtype=object)
    cluster_b = np.asarray([f"period-{v}" for v in time], dtype=object)
    clusters = np.column_stack([cluster_a, cluster_b])
    return X.astype(np.float64), y.astype(np.float64), entity, time, clusters


def _ill_conditioned_inputs(seed=20260814):
    rng = np.random.default_rng(seed)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack(
        [np.ones(n), x, x + 1.0e-9 * rng.normal(size=n)]
    )
    y = X @ np.array([0.35, 0.8, -0.45]) + rng.normal(scale=0.2, size=n)
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params
    time = np.tile(np.arange(10), 5)
    if np.linalg.matrix_rank(X) != 3 or np.linalg.cond(X) <= 1.0e8:
        raise AssertionError("ill-conditioned physical fixture lost full-rank contract")
    return X, resid, time


def _rank_boundary_inputs(seed=20260815):
    """Return a controlled design just below the explicit numerical-rank cutoff."""
    rng = np.random.default_rng(seed)
    n, k = 100, 3
    q_left, _ = np.linalg.qr(rng.normal(size=(n, k)))
    q_right, _ = np.linalg.qr(rng.normal(size=(k, k)))
    singular_values = np.array([10.0, 1.0, 1.0e-13])
    X = q_left @ np.diag(singular_values) @ q_right.T
    resid = rng.normal(size=n)
    time = np.tile(np.arange(10), 10)
    clusters = np.repeat(np.arange(10), 10)
    observed = np.linalg.svd(X, compute_uv=False)
    cutoff = max(X.shape) * np.finfo(np.float64).eps * observed[0]
    if not observed[-1] < cutoff < observed[-2]:
        raise AssertionError("rank-boundary fixture no longer straddles the explicit cutoff")
    if np.linalg.matrix_rank(X) != 2:
        raise AssertionError("rank-boundary fixture must have numerical rank two")
    return X, resid, time, clusters


def _rank_deficient_estimator_inputs(seed=20260816):
    """Return an unbalanced exact-collinearity panel for estimator integration."""
    rng = np.random.default_rng(seed)
    n_entities, n_times = 15, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    keep = np.ones(entity.size, dtype=bool)
    keep[[1, 10, 19, 33, 46, 57]] = False
    entity = entity[keep]
    time = time[keep]
    x = rng.normal(size=entity.size)
    X = np.column_stack([x, 2.0 * x]).astype(np.float64)
    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)[keep]
    y = 0.4 + 0.7 * x + alpha + rng.normal(scale=0.2, size=entity.size)
    return X, y.astype(np.float64), entity, time


def _to_backend(X, y, entity, time, backend):
    if backend == "numpy":
        return X, y, entity, time
    if backend == "cupy":
        import cupy as cp
        return (
            cp.asarray(X),
            cp.asarray(y),
            cp.asarray(entity, dtype=cp.int64),
            cp.asarray(time, dtype=cp.int64),
        )
    if backend == "torch":
        import torch
        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            torch.as_tensor(entity, dtype=torch.int64, device="cuda"),
            torch.as_tensor(time, dtype=torch.int64, device="cuda"),
        )
    raise ValueError(backend)


def _device(backend):
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _backend_name(model):
    executed = getattr(model, "_backend_name", None)
    if executed is None:
        raise AssertionError("fit did not persist executed backend provenance")
    return executed


def _array(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _optional_array(value):
    return None if value is None else _array(value)


def _array_backend_name(value):
    if _is_cupy_array(value):
        return "cupy"
    if _is_torch_array(value):
        return "torch"
    return "numpy"


def _public_primitive_cases(X, y, entity, time, clusters, backend):
    X_design = np.column_stack([np.ones(len(y)), X])
    params = np.linalg.lstsq(X_design, y, rcond=None)[0]
    resid = y - X_design @ params
    Xb, rb, _eb, _tb = _to_backend(X_design, resid, entity, time, backend)

    X_ill, resid_ill, time_ill = _ill_conditioned_inputs()
    dummy_entity = np.arange(len(resid_ill), dtype=np.int64)
    X_ill_b, resid_ill_b, _dummy_b, time_ill_b = _to_backend(
        X_ill, resid_ill, dummy_entity, time_ill, backend
    )
    X_rank, resid_rank, time_rank, cluster_rank = _rank_boundary_inputs()
    rank_entity = np.arange(len(resid_rank), dtype=np.int64)
    X_rank_b, resid_rank_b, _rank_entity_b, time_rank_b = _to_backend(
        X_rank, resid_rank, rank_entity, time_rank, backend
    )
    return {
        "cluster_group_debias": clustered_covariance(
            Xb, rb, clusters[:, 0], group_debias=True
        ),
        "driscoll_kraay_qs": driscoll_kraay_covariance(
            Xb, rb, time, bandwidth=2, kernel="qs"
        ),
        "ill_conditioned_hc0": ols_covariance(
            X_ill_b, resid_ill_b, cov_type="hc0"
        ),
        "ill_conditioned_hc2": ols_covariance(
            X_ill_b, resid_ill_b, cov_type="hc2"
        ),
        "ill_conditioned_hc3": ols_covariance(
            X_ill_b, resid_ill_b, cov_type="hc3"
        ),
        "ill_conditioned_dk": driscoll_kraay_covariance(
            X_ill_b, resid_ill_b, time_ill_b, bandwidth=2, kernel="bartlett"
        ),
        "rank_boundary_nonrobust": ols_covariance(
            X_rank_b, resid_rank_b, cov_type="nonrobust", scale=1.0
        ),
        "rank_boundary_hc0": ols_covariance(
            X_rank_b, resid_rank_b, cov_type="hc0"
        ),
        "rank_boundary_hc2": ols_covariance(
            X_rank_b, resid_rank_b, cov_type="hc2"
        ),
        "rank_boundary_hc3": ols_covariance(
            X_rank_b, resid_rank_b, cov_type="hc3"
        ),
        "rank_boundary_cluster": clustered_covariance(
            X_rank_b, resid_rank_b, cluster_rank
        ),
        "rank_boundary_dk": driscoll_kraay_covariance(
            X_rank_b, resid_rank_b, time_rank_b, bandwidth=2, kernel="bartlett"
        ),
    }


def _fit_rank(model):
    """Return the estimator fit-space numerical rank for audit payloads."""
    fit = model.fit_statistics_
    metadata = getattr(fit, "metadata", {}) if fit is not None else {}
    diagnostic_df = metadata.get("diagnostic_df")
    if isinstance(diagnostic_df, dict) and "rank_x" in diagnostic_df:
        return int(diagnostic_df["rank_x"])
    if "diagnostic_rank" in metadata:
        return int(metadata["diagnostic_rank"])
    if hasattr(model, "rank_"):
        return int(model.rank_)
    return int(len(np.asarray(model.coef_).ravel()))


def _snapshot(model):
    fit = model.fit_statistics_
    fit_payload = {
        "rsquared_within": fit.rsquared_within,
        "rsquared_between": fit.rsquared_between,
        "rsquared_overall": fit.rsquared_overall,
        "rsquared_adj": fit.rsquared_adj,
        "f_statistic": fit.f_statistic,
        "f_pvalue": fit.f_pvalue,
        "f_df": None if fit.f_df is None else tuple(float(v) for v in fit.f_df),
    }
    return {
        "coef": _array(model.coef_).ravel(),
        "bse": None if model.bse_ is None else _array(model.bse_).ravel(),
        "tvalues": None if model.tvalues_ is None else _array(model.tvalues_).ravel(),
        "pvalues": None if model.pvalues_ is None else _array(model.pvalues_).ravel(),
        "conf_int": _optional_array(model.conf_int_),
        "covariance": _array(model._panel_cov_params_raw),
        "coefficient_inference_applicable": bool(
            getattr(model, "_coefficient_inference_available", True)
        ),
        "coefficient_inference_reason": getattr(
            model, "_coefficient_inference_reason", None
        ),
        "prediction": _optional_array(getattr(model, "_physical_prediction", None)),
        "prediction_backend": getattr(model, "_predict_backend_name", None),
        "prediction_contract": getattr(model, "_physical_prediction_contract", None),
        "nobs": int(model.nobs),
        "df_resid": int(model.df_resid),
        "fit_statistics": fit_payload,
        "covariance_metadata": dict(getattr(model, "_covariance_metadata", {})),
    }


def _fit_cases(X, y, entity, time, clusters, backend):
    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)
    device = _device(backend)
    cases = {}

    for cov in ("hc0", "hc2", "hc3"):
        cases[f"pooled_{cov}"] = PooledOLS(cov_type=cov, device=device).fit(
            Xb, yb, entity_ids=eb
        )
    cases["pooled_cluster_one_way"] = PooledOLS(
        cov_type="clustered", device=device
    ).fit(Xb, yb, cluster=clusters[:, 0], entity_ids=eb)
    cases["pooled_cluster_two_way_group_debias"] = PooledOLS(
        cov_type="clustered", group_debias=True, device=device
    ).fit(Xb, yb, cluster=clusters, entity_ids=eb)
    cases["pooled_dk_bartlett"] = PooledOLS(
        cov_type="dk", bandwidth=2, kernel="bartlett", device=device
    ).fit(Xb, yb, entity_ids=eb, time_index=time)
    cases["pooled_dk_qs"] = PooledOLS(
        cov_type="dk", bandwidth=2, kernel="qs", device=device
    ).fit(Xb, yb, entity_ids=eb, time_index=time)
    cases["pooled_legacy_hac"] = PooledOLS(
        cov_type="hac", bandwidth=2, device=device
    ).fit(Xb, yb, entity_ids=eb, time_index=time)

    for cov in ("hc0", "hc2", "hc3"):
        cases[f"panel_entity_{cov}"] = PanelOLS(
            entity_effects=True, cov_type=cov, device=device
        ).fit(Xb, yb, entity_ids=eb)
        if cov == "hc0":
            cases[f"panel_entity_{cov}"]._physical_prediction = cases[
                f"panel_entity_{cov}"
            ].predict(Xb[:8], entity_ids=eb[:8])
            cases[f"panel_entity_{cov}"]._physical_prediction_contract = (
                "entity_effect_prediction"
            )
    cases["panel_two_way_hc3"] = PanelOLS(
        entity_effects=True, time_effects=True, cov_type="hc3", device=device
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    cases["panel_two_way_hc3"]._physical_prediction = cases[
        "panel_two_way_hc3"
    ].predict(Xb[:8], entity_ids=eb[:8], time_ids=tb[:8])
    cases["panel_two_way_hc3"]._physical_prediction_contract = (
        "two_way_effect_prediction"
    )
    cases["panel_two_way_cluster_group_debias"] = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="clustered",
        group_debias=True,
        device=device,
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb, cluster=clusters)
    cases["panel_two_way_dk"] = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="dk",
        bandwidth=2,
        device=device,
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)

    Xc = np.column_stack([np.ones(len(y)), X])
    Xcb, ycb, ecb, tcb = _to_backend(Xc, y, entity, time, backend)
    for cov in ("robust", "hc0", "hc2", "hc3"):
        cases[f"random_effects_explicit_constant_{cov}"] = RandomEffects(
            cov_type=cov, device=device
        ).fit(Xcb, ycb, entity_ids=ecb)
        if cov == "hc0":
            # Deliberately omit the fitted explicit constant.  This exercises the
            # shared backend-native constant restoration path rather than only
            # predicting with the already-complete design matrix.
            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction = cases[
                f"random_effects_explicit_constant_{cov}"
            ].predict(Xb[:8])
            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction_contract = (
                "omitted_explicit_constant"
            )
    cases["random_effects_cluster_two_way"] = RandomEffects(
        cov_type="clustered", group_debias=True, device=device
    ).fit(Xcb, ycb, entity_ids=ecb, cluster=clusters)
    cases["random_effects_dk"] = RandomEffects(
        cov_type="dk", bandwidth=2, kernel="parzen", device=device
    ).fit(Xcb, ycb, entity_ids=ecb, time_ids=tcb)

    for cov in ("hc0", "hc2", "hc3"):
        cases[f"between_{cov}"] = BetweenOLS(cov_type=cov, device=device).fit(
            Xb, yb, entity_ids=eb
        )
        cases[f"first_difference_{cov}"] = FirstDifferenceOLS(
            cov_type=cov, device=device
        ).fit(Xb, yb, entity_ids=eb, time_ids=tb)

    X_rd, y_rd, entity_rd, time_rd = _rank_deficient_estimator_inputs()
    X_rd_b, y_rd_b, entity_rd_b, time_rd_b = _to_backend(
        X_rd, y_rd, entity_rd, time_rd, backend
    )
    X_rd_re = np.column_stack([np.ones(len(y_rd)), X_rd])
    X_rd_re_b, y_rd_re_b, entity_rd_re_b, _time_rd_re_b = _to_backend(
        X_rd_re, y_rd, entity_rd, time_rd, backend
    )
    for cov in ("nonrobust", "robust"):
        cases[f"panel_entity_rank_deficient_{cov}"] = PanelOLS(
            entity_effects=True, cov_type=cov, device=device
        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b)
        cases[f"between_rank_deficient_{cov}"] = BetweenOLS(
            cov_type=cov, device=device
        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b)
        cases[f"first_difference_rank_deficient_{cov}"] = FirstDifferenceOLS(
            cov_type=cov, device=device
        ).fit(X_rd_b, y_rd_b, entity_ids=entity_rd_b, time_ids=time_rd_b)
        cases[f"random_effects_rank_deficient_{cov}"] = RandomEffects(
            cov_type=cov, device=device
        ).fit(X_rd_re_b, y_rd_re_b, entity_ids=entity_rd_re_b)

    X_rank, y_rank, time_rank, _cluster_rank = _rank_boundary_inputs()
    rank_entity = np.arange(len(y_rank), dtype=np.int64)
    X_rank_b, y_rank_b, _rank_entity_b, time_rank_b = _to_backend(
        X_rank, y_rank, rank_entity, time_rank, backend
    )
    cases["panel_rank_boundary_dk"] = PanelOLS(
        cov_type="dk", bandwidth=2, device=device
    ).fit(X_rank_b, y_rank_b, time_ids=time_rank_b)

    return cases


def _level_constant_contract_audit(backend, *, rtol=5e-6, atol=5e-7):
    """Audit no-FE PanelOLS level-intercept inference against PooledOLS."""
    rng = np.random.default_rng(20260822)
    n = 96
    x = rng.normal(size=n).astype(np.float64)
    y = (1.15 + 0.65 * x + rng.normal(scale=0.15, size=n)).astype(np.float64)
    X_full = np.column_stack([np.ones(n), x]).astype(np.float64)
    X_slope = x[:, None]
    dummy_entity = np.arange(n, dtype=np.int64)
    dummy_time = np.arange(n, dtype=np.int64)
    X_full_b, yb, _eb, _tb = _to_backend(
        X_full, y, dummy_entity, dummy_time, backend
    )
    X_slope_b, _yb2, _eb2, _tb2 = _to_backend(
        X_slope, y, dummy_entity, dummy_time, backend
    )
    device = _device(backend)
    panel = PanelOLS(cov_type="nonrobust", device=device).fit(X_full_b, yb)
    pooled = PooledOLS(cov_type="nonrobust", device=device).fit(X_slope_b, yb)
    if _backend_name(panel) != backend or _backend_name(pooled) != backend:
        raise AssertionError("level-constant audit fit backend provenance drifted")

    panel_prediction = panel.predict(X_full_b[:12])
    pooled_prediction = pooled.predict(X_slope_b[:12])
    if getattr(panel, "_predict_backend_name", None) != backend:
        raise AssertionError("level-constant audit prediction backend provenance drifted")
    if panel._predict_constant_index != 0 or float(panel._predict_constant_value) != 1.0:
        raise AssertionError("level-constant audit did not retain the fitted constant")

    numeric = {
        "coef": (_array(panel.coef_), _array(pooled.coef_)),
        "bse": (_array(panel.bse_), _array(pooled.bse_)),
        "prediction": (_array(panel_prediction), _array(pooled_prediction)),
    }
    differences = {}
    for field, (actual, expected) in numeric.items():
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=field)
        differences[field] = _max_abs(actual, expected)

    panel_fit = panel.fit_statistics_
    pooled_fit = pooled.fit_statistics_
    for field in ("rsquared_overall", "rsquared_adj", "f_statistic", "f_pvalue"):
        actual = getattr(panel_fit, field)
        expected = getattr(pooled_fit, field)
        differences[field] = _scalar_diff(
            actual, expected, rtol=rtol, atol=atol, label=f"level_constant.{field}"
        )
    if panel_fit.f_df != pooled_fit.f_df:
        raise AssertionError(f"level_constant.f_df: {panel_fit.f_df!r} != {pooled_fit.f_df!r}")
    if int(panel.df_resid) != int(pooled.df_resid):
        raise AssertionError("level_constant residual df differs from PooledOLS")
    diagnostic_df = panel_fit.metadata.get("diagnostic_df", {})
    if int(diagnostic_df.get("df_total", -1)) != n - 1:
        raise AssertionError("level_constant adjusted-R2 total df did not account for intercept")

    return {
        "status": "success",
        "executed_backend": backend,
        "prediction_backend": getattr(panel, "_predict_backend_name", None),
        "constant_index": int(panel._predict_constant_index),
        "constant_value": float(panel._predict_constant_value),
        "df_resid": int(panel.df_resid),
        "f_df": list(panel_fit.f_df) if panel_fit.f_df is not None else None,
        "max_abs_differences_vs_pooled": differences,
        "values": {
            "coef": _array(panel.coef_).tolist(),
            "bse": _array(panel.bse_).tolist(),
            "prediction": _array(panel_prediction).tolist(),
            "rsquared_overall": float(panel_fit.rsquared_overall),
            "rsquared_adj": float(panel_fit.rsquared_adj),
            "f_statistic": float(panel_fit.f_statistic),
            "f_pvalue": float(panel_fit.f_pvalue),
        },
    }


def _disconnected_two_way_prediction_audit(backend):
    """Exercise disconnected two-way prediction identifiability on one backend."""
    rng = np.random.default_rng(20260820)
    entity = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    time = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X = rng.normal(size=(entity.size, 1)).astype(np.float64)
    alpha = np.array([0.5, -0.2, 1.1, -0.7], dtype=np.float64)
    tau = np.array([0.25, -0.15, 0.6, -0.4], dtype=np.float64)
    y = (0.8 * X[:, 0] + alpha[entity] + tau[time]).astype(np.float64)
    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)
    model = PanelOLS(
        entity_effects=True, time_effects=True, cov_type="hc0", device=_device(backend)
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    executed = _backend_name(model)
    if executed != backend:
        raise AssertionError(
            f"disconnected prediction audit requested {backend}, executed {executed}"
        )

    observed = model.predict(Xb, entity_ids=eb, time_ids=tb)
    same_component = model.predict(
        Xb[:1], entity_ids=np.array([1]), time_ids=np.array([1])
    )

    def guarded(label, **kwargs):
        try:
            model.predict(Xb[:1], **kwargs)
        except ValueError as exc:
            if "disconnected incidence graph" not in str(exc):
                raise AssertionError(
                    f"{label}: wrong disconnected-prediction failure: {exc}"
                ) from exc
            return True
        raise AssertionError(f"{label}: disconnected prediction did not fail closed")

    guards = {
        "cross_component": guarded(
            "cross_component", entity_ids=np.array([0]), time_ids=np.array([2])
        ),
        "entity_only": guarded("entity_only", entity_ids=np.array([0])),
        "time_only": guarded("time_only", time_ids=np.array([0])),
        "known_entity_unknown_time": guarded(
            "known_entity_unknown_time",
            entity_ids=np.array([0]),
            time_ids=np.array([99]),
        ),
        "unknown_entity_known_time": guarded(
            "unknown_entity_known_time",
            entity_ids=np.array([99]),
            time_ids=np.array([0]),
        ),
    }
    both_unseen = model.predict(
        Xb[:1], entity_ids=np.array([98]), time_ids=np.array([99])
    )
    prediction_backend = getattr(model, "_predict_backend_name", None)
    if prediction_backend != backend:
        raise AssertionError(
            "disconnected prediction audit did not persist requested prediction backend: "
            f"{prediction_backend!r} != {backend!r}"
        )
    return {
        "executed_backend": executed,
        "prediction_backend": prediction_backend,
        "observed": _array(observed),
        "same_component": _array(same_component),
        "both_unseen": _array(both_unseen),
        "guards": guards,
    }


def _max_abs(actual, expected):
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _scalar_diff(actual, expected, *, rtol, atol, label):
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{label}: expected None, got {actual}")
        return 0.0
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)
    return float(abs(float(actual) - float(expected)))


def _compare(reference, candidate, *, rtol, atol, label):
    differences = {}
    for field in ("coef", "bse", "tvalues", "pvalues", "conf_int", "covariance", "prediction"):
        actual = candidate[field]
        expected = reference[field]
        if expected is None or actual is None:
            if expected is not None or actual is not None:
                raise AssertionError(
                    f"{label}.{field}: None contract differs between candidate and reference"
                )
            differences[field] = 0.0
            continue
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=f"{label}.{field}"
        )
        differences[field] = _max_abs(actual, expected)
    for field in (
        "coefficient_inference_applicable",
        "coefficient_inference_reason",
        "prediction_backend",
        "prediction_contract",
    ):
        actual = candidate[field]
        expected = reference[field]
        if field == "prediction_backend" and expected == "numpy" and actual is not None:
            # NumPy reference predicts on NumPy; GPU candidates must persist the
            # requested execution backend instead of matching the reference label.
            continue
        if actual != expected:
            raise AssertionError(f"{label}.{field}: {actual!r} != {expected!r}")
        differences[field] = 0.0
    for field in ("nobs", "df_resid"):
        if candidate[field] != reference[field]:
            raise AssertionError(f"{label}.{field}: {candidate[field]} != {reference[field]}")
        differences[field] = 0.0
    for field, expected in reference["fit_statistics"].items():
        actual = candidate["fit_statistics"][field]
        if field == "f_df":
            if expected is None:
                if actual is not None:
                    raise AssertionError(f"{label}.f_df expected None")
            else:
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        else:
            differences[f"fit_statistics.{field}"] = _scalar_diff(
                actual, expected, rtol=rtol, atol=atol, label=f"{label}.fit_statistics.{field}"
            )
    ref_meta = reference["covariance_metadata"]
    cand_meta = candidate["covariance_metadata"]
    if set(cand_meta) != set(ref_meta):
        raise AssertionError(
            f"{label}.covariance_metadata keys mismatch: "
            f"{sorted(cand_meta)} != {sorted(ref_meta)}"
        )
    for key, expected in ref_meta.items():
        actual = cand_meta[key]
        metric = f"covariance_metadata.{key}"
        if isinstance(expected, float):
            differences[metric] = _scalar_diff(
                actual, expected, rtol=rtol, atol=atol, label=f"{label}.{metric}"
            )
        elif isinstance(expected, list) and any(isinstance(v, float) for v in expected):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"{label}.{metric}")
            differences[metric] = _max_abs(
                np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
            )
        elif actual != expected:
            raise AssertionError(
                f"{label}.{metric}: {actual!r} != {expected!r}"
            )
        else:
            differences[metric] = 0.0
    return differences


def _environment(backends):
    gpu = None
    if "torch" in backends:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Torch backend requested but CUDA is unavailable")
        gpu = torch.cuda.get_device_name(0)
    elif "cupy" in backends:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy backend requested but CUDA is unavailable")
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": {
            name: (
                _version("cupy")
                or _version("cupy-cuda11x")
                or _version("cupy-cuda12x")
                if name == "cupy"
                else _version(name)
            )
            for name in ("statgpu", "numpy", "scipy", "cupy", "torch")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--rtol", type=float, default=5e-6)
    parser.add_argument("--atol", type=float, default=5e-7)
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    if not backends or any(item not in {"cupy", "torch"} for item in backends):
        raise ValueError("--backends must contain cupy and/or torch")
    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    dirty = _git_status_porcelain()
    if dirty.strip():
        raise RuntimeError("physical acceptance requires a clean working tree:\n" + dirty)

    X, y, entity, time, clusters = _dataset()
    reference_models = _fit_cases(X, y, entity, time, clusters, "numpy")
    reference = {name: _snapshot(model) for name, model in reference_models.items()}
    prediction_reference = _disconnected_two_way_prediction_audit("numpy")
    level_constant_reference = _level_constant_contract_audit(
        "numpy", rtol=args.rtol, atol=args.atol
    )
    primitive_reference = {
        name: _array(value)
        for name, value in _public_primitive_cases(
            X, y, entity, time, clusters, "numpy"
        ).items()
    }
    required_public_primitives = {
        "cluster_group_debias",
        "driscoll_kraay_qs",
        "ill_conditioned_hc0",
        "ill_conditioned_hc2",
        "ill_conditioned_hc3",
        "ill_conditioned_dk",
        "rank_boundary_nonrobust",
        "rank_boundary_hc0",
        "rank_boundary_hc2",
        "rank_boundary_hc3",
        "rank_boundary_cluster",
        "rank_boundary_dk",
    }
    if set(primitive_reference) != required_public_primitives:
        raise AssertionError("NumPy public primitive acceptance matrix drifted")

    results = {}
    for backend in backends:
        models = _fit_cases(X, y, entity, time, clusters, backend)
        payload = {
            "status": "success",
            "requested_backend": backend,
            "cases": {},
            "public_primitives": {},
            "prediction_contracts": {},
            "level_constant_contract": {},
        }
        if set(models) != set(reference):
            raise AssertionError(f"{backend}: physical case set differs from NumPy reference")
        for name, model in models.items():
            executed = _backend_name(model)
            if executed != backend:
                raise AssertionError(f"{name}: requested {backend}, executed {executed}")
            snapshot = _snapshot(model)
            differences = _compare(
                reference[name], snapshot, rtol=args.rtol, atol=args.atol, label=name
            )
            fit_rank = _fit_rank(model)
            parameter_count = int(snapshot["coef"].size)
            if fit_rank < parameter_count:
                if snapshot["coefficient_inference_applicable"]:
                    raise AssertionError(
                        f"{name}: rank-deficient fit published coordinate inference"
                    )
                reason = snapshot["coefficient_inference_reason"]
                if not reason or "rank deficient" not in reason:
                    raise AssertionError(
                        f"{name}: rank-deficient inference reason is not auditable"
                    )
            if name in {
                "panel_entity_hc0",
                "panel_two_way_hc3",
                "random_effects_explicit_constant_hc0",
            } and snapshot["prediction_backend"] != backend:
                raise AssertionError(
                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"
                )
            payload["cases"][name] = {
                "status": "success",
                "executed_backend": executed,
                "max_abs_differences": differences,
                "covariance_metadata": snapshot["covariance_metadata"],
                "fit_rank": fit_rank,
                "parameter_count": parameter_count,
                "coefficient_inference_applicable": snapshot[
                    "coefficient_inference_applicable"
                ],
                "coefficient_inference_reason": snapshot[
                    "coefficient_inference_reason"
                ],
                "prediction_backend": snapshot["prediction_backend"],
                "prediction_contract": snapshot["prediction_contract"],
            }

        prediction_audit = _disconnected_two_way_prediction_audit(backend)
        if prediction_audit["executed_backend"] != backend:
            raise AssertionError("disconnected prediction fit backend provenance drifted")
        if prediction_audit["prediction_backend"] != backend:
            raise AssertionError("disconnected prediction execution backend provenance drifted")
        if not all(prediction_audit["guards"].values()):
            raise AssertionError("disconnected prediction guard audit did not fail closed")
        prediction_diffs = {}
        for field in ("observed", "same_component", "both_unseen"):
            np.testing.assert_allclose(
                prediction_audit[field],
                prediction_reference[field],
                rtol=args.rtol,
                atol=args.atol,
                err_msg=f"two_way_disconnected_prediction.{field}",
            )
            prediction_diffs[field] = _max_abs(
                prediction_audit[field], prediction_reference[field]
            )
        payload["prediction_contracts"]["two_way_disconnected"] = {
            "status": "success",
            "executed_backend": prediction_audit["executed_backend"],
            "prediction_backend": prediction_audit["prediction_backend"],
            "guards": dict(prediction_audit["guards"]),
            "max_abs_differences": prediction_diffs,
        }

        level_constant_audit = _level_constant_contract_audit(
            backend, rtol=args.rtol, atol=args.atol
        )
        if level_constant_audit["executed_backend"] != backend:
            raise AssertionError("level-constant fit backend provenance drifted")
        if level_constant_audit["prediction_backend"] != backend:
            raise AssertionError("level-constant prediction backend provenance drifted")
        if level_constant_audit["constant_index"] != level_constant_reference["constant_index"]:
            raise AssertionError("level-constant fitted constant index drifted")
        np.testing.assert_allclose(
            level_constant_audit["constant_value"],
            level_constant_reference["constant_value"],
            rtol=0,
            atol=0,
        )
        numpy_diffs = {}
        for field in ("coef", "bse", "prediction"):
            actual = np.asarray(level_constant_audit["values"][field], dtype=np.float64)
            expected = np.asarray(level_constant_reference["values"][field], dtype=np.float64)
            np.testing.assert_allclose(
                actual, expected, rtol=args.rtol, atol=args.atol,
                err_msg=f"level_constant.numpy.{field}"
            )
            numpy_diffs[field] = _max_abs(actual, expected)
        for field in ("rsquared_overall", "rsquared_adj", "f_statistic", "f_pvalue"):
            numpy_diffs[field] = _scalar_diff(
                level_constant_audit["values"][field],
                level_constant_reference["values"][field],
                rtol=args.rtol,
                atol=args.atol,
                label=f"level_constant.numpy.{field}",
            )
        if level_constant_audit["f_df"] != level_constant_reference["f_df"]:
            raise AssertionError("level-constant F degrees of freedom drifted from NumPy")
        level_constant_audit["max_abs_differences_vs_numpy"] = numpy_diffs
        payload["level_constant_contract"] = level_constant_audit

        primitive_values = _public_primitive_cases(
            X, y, entity, time, clusters, backend
        )
        if set(primitive_values) != required_public_primitives:
            raise AssertionError(
                f"{backend}: public primitive acceptance matrix drifted"
            )
        for name, value in primitive_values.items():
            executed = _array_backend_name(value)
            if executed != backend:
                raise AssertionError(
                    f"public primitive {name}: requested {backend}, executed {executed}"
                )
            actual = _array(value)
            np.testing.assert_allclose(
                actual,
                primitive_reference[name],
                rtol=args.rtol,
                atol=args.atol,
                err_msg=f"public primitive {name}",
            )
            payload["public_primitives"][name] = {
                "status": "success",
                "executed_backend": executed,
                "max_abs_difference": _max_abs(actual, primitive_reference[name]),
            }
        results[backend] = payload

    required_cases = {
        "pooled_hc0", "pooled_hc2", "pooled_hc3",
        "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
        "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
        "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3", "panel_two_way_hc3",
        "panel_two_way_cluster_group_debias", "panel_two_way_dk",
        "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
        "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
        "random_effects_cluster_two_way", "random_effects_dk",
        "between_hc0", "between_hc2", "between_hc3",
        "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
        "panel_entity_rank_deficient_nonrobust", "panel_entity_rank_deficient_robust",
        "between_rank_deficient_nonrobust", "between_rank_deficient_robust",
        "first_difference_rank_deficient_nonrobust", "first_difference_rank_deficient_robust",
        "random_effects_rank_deficient_nonrobust", "random_effects_rank_deficient_robust",
        "panel_rank_boundary_dk",
    }
    if set(reference) != required_cases:
        missing = sorted(required_cases - set(reference))
        unexpected = sorted(set(reference) - required_cases)
        raise AssertionError(
            "NumPy reference Stage-C physical matrix drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(reference) != 35:
        raise AssertionError(f"expected 35 Stage-C physical cases, got {len(reference)}")
    for backend, payload in results.items():
        if set(payload["cases"]) != required_cases:
            missing = sorted(required_cases - set(payload["cases"]))
            unexpected = sorted(set(payload["cases"]) - required_cases)
            raise AssertionError(
                f"{backend}: Stage-C physical matrix drifted: "
                f"missing={missing}, unexpected={unexpected}"
            )

    output = {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "working_tree_clean": True,
        "status": "success",
        "environment": _environment(backends),
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "dataset": {
            "nobs": int(len(y)),
            "n_entities": int(len(np.unique(entity))),
            "n_times": int(len(np.unique(time))),
            "n_features": int(X.shape[1]),
            "unbalanced": True,
        },
        "case_count_per_backend": len(reference),
        "public_primitive_count_per_backend": len(required_public_primitives),
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"PASS — Panel Stage C physical GPU validation: {args.out}")


if __name__ == "__main__":
    main()
