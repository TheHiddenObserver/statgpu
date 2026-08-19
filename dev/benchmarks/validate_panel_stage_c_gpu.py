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
from statgpu.panel._covariance import _grouped_score_sums, hac_covariance, ols_covariance, two_way_clustered_covariance
from statgpu.panel._diagnostic_context import (
    bp_lm_from_residuals,
    pooling_f_from_level_arrays,
)
from statgpu.panel._diagnostics import (
    _classical_model_f,
    _hausman_quadratic,
    _scaled_group_means,
    _scaled_mean,
)
from statgpu.panel._linalg import (
    panel_lstsq,
    panel_lstsq_batched,
    panel_lstsq_gram_certified_batched,
    panel_matrix_rank,
)
from statgpu.panel._reductions import stable_reduction_flags
from statgpu.panel._utils import (
    _recover_two_way_effects,
    _zero_safe_statistic_ratio,
    demean_variables,
    within_transform,
)


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


def _projection_created_dynamic_range_audit(backend):
    """Exercise a stability flag that appears only after the first FE projection."""
    upper = np.nextafter(1.0, 2.0)
    y_np = np.asarray([1.0, -1.0, upper, 1.0, -1.0, 1.0], dtype=np.float64)
    X_np = np.asarray([0.3, -0.5, 0.7, 0.2, -0.8, 0.6], dtype=np.float64)[:, None]
    entity = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    X, y, entity_b, time_b = _to_backend(X_np, y_np, entity, time, backend)
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
    elif backend == "torch":
        import torch
        xp = torch
    else:
        raise ValueError(backend)

    raw_flag = bool(stable_reduction_flags(y, xp)[0])
    entity_projected = within_transform(y, entity_b, xp=xp)
    projected_flag = bool(stable_reduction_flags(entity_projected, xp)[0])
    if raw_flag or not projected_flag:
        raise AssertionError(
            f"{backend}: projected-risk fixture did not transition False -> True: "
            f"raw={raw_flag}, projected={projected_flag}"
        )

    y_d, X_d = demean_variables(
        y, X, entity_b, time_b, xp=xp, max_iter=200, tol=1.0e-12
    )
    y_ref, X_ref = demean_variables(
        y_np, X_np, entity, time, xp=np, max_iter=200, tol=1.0e-12
    )
    y_actual = _array(y_d)
    X_actual = _array(X_d)
    np.testing.assert_allclose(
        y_actual, y_ref, rtol=0.0, atol=2.0e-15,
        err_msg=f"{backend}: projection-created y stability path",
    )
    np.testing.assert_allclose(
        X_actual, X_ref, rtol=2.0e-14, atol=2.0e-15,
        err_msg=f"{backend}: projection-created X stability path",
    )
    for codes in (entity, time):
        for level in np.unique(codes):
            if abs(float(np.mean(y_actual[codes == level]))) > 2.0e-15:
                raise AssertionError(f"{backend}: projected-risk y group mean did not converge")
    return {
        "status": "success",
        "backend": backend,
        "raw_stability_flag": raw_flag,
        "post_entity_stability_flag": projected_flag,
        "max_abs_y_vs_numpy": _max_abs(y_actual, y_ref),
        "max_abs_X_vs_numpy": _max_abs(X_actual, X_ref),
    }



def _fixed_effect_recovery_cancellation_audit(backend):
    """Keep public FE prediction maps on cancellation-safe group means."""
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
    elif backend == "torch":
        import torch
        xp = torch
    else:
        raise ValueError(backend)

    amplitude = float(2.0 ** 55)
    entity = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    time = np.arange(6, dtype=np.int64)
    X_np = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0], dtype=np.float64)[:, None]
    y_np = np.asarray([amplitude, 1.0, -amplitude, 0.0, 0.0, 0.0], dtype=np.float64)
    X, y, entity_b, _time_b = _to_backend(X_np, y_np, entity, time, backend)
    model = PanelOLS(entity_effects=True, cov_type="hc0", device=_device(backend)).fit(
        X, y, entity_ids=entity_b
    )
    prediction = _array(model.predict(X, entity_ids=entity_b))
    expected = np.asarray([1.0 / 3.0] * 3 + [0.0] * 3, dtype=np.float64)
    np.testing.assert_allclose(
        prediction, expected, rtol=0.0, atol=2e-15,
        err_msg=f"{backend}: one-way FE prediction cancellation tail",
    )
    if _backend_name(model) != backend or getattr(model, "_predict_backend_name", None) != backend:
        raise AssertionError(f"{backend}: FE prediction backend provenance drifted")

    level = float(2.0 ** 50)
    values_np = np.asarray(
        [1.5 * level, 0.5 * level, level + 1.0, level - 1.0,
         0.5 * level, 1.5 * level],
        dtype=np.float64,
    )
    entity2 = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    dummy_X = np.arange(6, dtype=np.float64)[:, None]
    X2, values, entity2_b, time2_b = _to_backend(
        dummy_X, values_np, entity2, time2, backend
    )
    reference, _ = demean_variables(
        values, X2, entity2_b, time2_b, xp=xp, max_iter=200, tol=1e-12
    )
    entity_effect, time_effect = _recover_two_way_effects(
        values, entity2_b, time2_b, xp, max_iter=200, tol=1e-12
    )
    recovered = values - entity_effect[entity2_b] - time_effect[time2_b]
    recovered_np = _array(recovered)
    reference_np = _array(reference)
    np.testing.assert_allclose(
        recovered_np[2:4], reference_np[2:4], rtol=0.0, atol=2e-12,
        err_msg=f"{backend}: two-way FE recovery projection-created risk",
    )
    return {
        "status": "success",
        "backend": backend,
        "prediction_backend": getattr(model, "_predict_backend_name", None),
        "max_abs_prediction_error": _max_abs(prediction, expected),
        "max_abs_two_way_low_order_error": _max_abs(
            recovered_np[2:4], reference_np[2:4]
        ),
    }

def _nonfinite_covariance_guard_audit(backend):
    X_np = np.column_stack([np.ones(6), np.arange(6.0)])
    resid_np = np.linspace(-0.3, 0.4, 6)
    resid_np[2] = np.nan
    c1 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    time = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    dummy = np.arange(6, dtype=np.int64)
    X, resid, _entity, _time = _to_backend(X_np, resid_np, dummy, time, backend)
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
    elif backend == "torch":
        import torch
        xp = torch
    else:
        raise ValueError(backend)

    calls = {
        "cluster": lambda: clustered_covariance(X, resid, c1, xp=xp),
        "two_way": lambda: two_way_clustered_covariance(X, resid, c1, c2, xp=xp),
        "hac": lambda: hac_covariance(X, resid, bandwidth=1, xp=xp),
        "dk": lambda: driscoll_kraay_covariance(X, resid, time, bandwidth=1, xp=xp),
    }
    guards = {}
    for name, call in calls.items():
        try:
            call()
        except ValueError as exc:
            if "X and resid must contain only finite values" not in str(exc):
                raise AssertionError(f"{backend}: {name} raised wrong nonfinite error: {exc}") from exc
            guards[name] = True
        else:
            raise AssertionError(f"{backend}: {name} accepted a NaN residual")
    return {"status": "success", "backend": backend, "guards": guards}


def _cancellation_safe_mean_audit(backend):
    values = np.asarray([1.0e154, -1.0e154, 1.0e-170], dtype=np.float64)
    grouped = np.asarray(
        [1.0e154, -1.0e154, 1.0e-170, 1.0e-320, 1.0e-320], dtype=np.float64
    )
    groups = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    dummy = np.arange(values.size, dtype=np.float64)[:, None]
    entity = np.arange(values.size, dtype=np.int64)
    time = np.arange(values.size, dtype=np.int64)
    _dummy_b, values_b, _eb, _tb = _to_backend(dummy, values, entity, time, backend)
    dummy_g = np.arange(grouped.size, dtype=np.float64)[:, None]
    entity_g = np.arange(grouped.size, dtype=np.int64)
    time_g = np.arange(grouped.size, dtype=np.int64)
    _dg, grouped_b, groups_b, _tg = _to_backend(dummy_g, grouped, groups, time_g, backend)
    xp = __import__("torch") if backend == "torch" else __import__("cupy")
    mean = float(_array(_scaled_mean(values_b, xp)))
    group_result = _array(_scaled_group_means(grouped_b, groups_b, xp))
    expected_mean = 1.0e-170 / 3.0
    expected_group = np.asarray(
        [expected_mean] * 3 + [1.0e-320, 1.0e-320], dtype=np.float64
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=3e-11, atol=0.0)
    np.testing.assert_allclose(group_result, expected_group, rtol=3e-11, atol=0.0)
    return {"status": "success", "backend": backend, "mean": mean}


def _diagnostic_scale_audit(backend):
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
    elif backend == "torch":
        import torch
        xp = torch
    else:
        raise ValueError(backend)

    # Pooling F: naive column/scalar means overflow after this common scaling,
    # while the centered regression and scale-invariant statistic are finite.
    n = 24
    t = np.linspace(-1.0, 1.0, n)
    X = np.column_stack(
        [1.15 + 0.18 * t, 0.95 - 0.11 * t + 0.03 * t * t]
    ).astype(np.float64)
    y = (
        1.05
        + 0.42 * t
        - 0.08 * t * t
        + 0.025 * np.sin(np.arange(n))
    ).astype(np.float64)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    beta, _ = panel_lstsq(Xc, yc, np)
    effects_resid = 0.55 * (yc - Xc @ beta)
    dummy = np.arange(n, dtype=np.int64)

    def _pooling(scale):
        Xb, yb, _eb, _tb = _to_backend(
            scale * X, scale * y, dummy, dummy, backend
        )
        _X2, effects_b, _e2, _t2 = _to_backend(
            scale * X, scale * effects_resid, dummy, dummy, backend
        )
        return pooling_f_from_level_arrays(
            yb,
            Xb,
            xp=xp,
            rss_effects=0.0,
            df_resid_effects=n - 6,
            has_constant=False,
            resid_effects=effects_b,
        )

    pooling_reference = _pooling(1.0)
    pooling_large = _pooling(1.0e307)
    if not pooling_reference.applicable or not pooling_large.applicable:
        raise AssertionError(f"{backend}: pooling-F scale audit became inapplicable")
    np.testing.assert_allclose(
        pooling_large.statistic, pooling_reference.statistic,
        rtol=5e-8, atol=1e-10,
    )
    np.testing.assert_allclose(
        pooling_large.pvalue, pooling_reference.pvalue,
        rtol=5e-8, atol=1e-12,
    )

    # Classical model F: use a subnormal response scale so direct backend scalar
    # division is exercised. The statistic must remain invariant to response units.
    x = np.linspace(-1.0, 1.0, 12)
    Xf = np.column_stack([np.ones(x.size), x]).astype(np.float64)
    yf = (
        0.8
        + 0.45 * x
        + np.asarray(
            [0.08, -0.04, 0.03, -0.06, 0.05, -0.02,
             0.01, 0.04, -0.03, 0.02, -0.01, 0.05],
            dtype=np.float64,
        )
    )
    entity_f = np.arange(x.size, dtype=np.int64)

    def _model_f(scale):
        Xb, yb, _eb, _tb = _to_backend(
            Xf, scale * yf, entity_f, entity_f, backend
        )
        params, rank = panel_lstsq(Xb, yb, xp)
        result = _classical_model_f(
            yb, Xb, params, xp=xp,
            df_resid=x.size - int(rank), has_constant=True,
        )
        if result[0] is None or not np.isfinite(result[0]):
            raise AssertionError(f"{backend}: classical-F scale audit is not finite")
        return result

    model_f_reference = _model_f(1.0)
    model_f_tiny = _model_f(1.0e-310)
    np.testing.assert_allclose(
        model_f_tiny[0], model_f_reference[0], rtol=5e-4, atol=5e-6
    )
    np.testing.assert_allclose(
        model_f_tiny[1], model_f_reference[1], rtol=5e-4, atol=5e-8
    )

    # Baltagi-Li BP-LM is also response-scale invariant and uses grouped backend
    # reductions, so verify the subnormal path on the requested physical backend.
    groups = np.repeat(np.arange(5), 4).astype(np.int64)
    pattern = np.asarray(
        [1.0, -0.4, 0.6, -0.3, 0.8, -0.2, 0.5, -0.7, 1.1, -0.5,
         0.3, -0.1, 0.7, -0.6, 0.4, -0.2, 0.9, -0.3, 0.2, -0.4],
        dtype=np.float64,
    )
    dummy_x = np.arange(pattern.size, dtype=np.float64)[:, None]
    dummy_time = np.arange(pattern.size, dtype=np.int64)

    def _bp(scale):
        _xb, resid_b, groups_b, _tb = _to_backend(
            dummy_x, scale * pattern, groups, dummy_time, backend
        )
        result = bp_lm_from_residuals(resid_b, groups_b, xp=xp)
        if not result.applicable or not np.isfinite(result.statistic):
            raise AssertionError(f"{backend}: BP-LM scale audit is not finite/applicable")
        return result

    bp_reference = _bp(1.0)
    bp_tiny = _bp(1.0e-310)
    np.testing.assert_allclose(
        bp_tiny.statistic, bp_reference.statistic, rtol=5e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        bp_tiny.pvalue, bp_reference.pvalue, rtol=5e-8, atol=1e-12
    )

    return {
        "status": "success",
        "backend": backend,
        "pooling_f_statistic": float(pooling_large.statistic),
        "pooling_f_pvalue": float(pooling_large.pvalue),
        "classical_f_statistic": float(model_f_tiny[0]),
        "classical_f_pvalue": float(model_f_tiny[1]),
        "bp_lm_statistic": float(bp_tiny.statistic),
        "bp_lm_pvalue": float(bp_tiny.pvalue),
    }


def _covariance_extreme_scale_audit(backend):
    amplitude = 1.4e154
    X_np = np.ones((4, 1), dtype=np.float64)
    resid_np = np.asarray([amplitude, amplitude, -amplitude, -amplitude])
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    score_amplitude = 1.6e308
    scores_np = np.asarray(
        [[score_amplitude], [score_amplitude], [-score_amplitude], [-score_amplitude], [1.0], [-1.0]],
        dtype=np.float64,
    )
    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    n = 16
    influence_amplitude = 3.0e153
    X_hac_np = np.ones((n, 1), dtype=np.float64)
    resid_hac_np = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    time = np.arange(n, dtype=np.int64)

    lag_n = 7
    lag_bandwidth = 4
    lag_influence_sq = 2.0e307
    lag_influence_amp = float(np.sqrt(lag_influence_sq))
    lag_signs = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    X_lag_np = np.ones((lag_n, 1), dtype=np.float64)
    resid_lag_np = lag_n * lag_influence_amp * lag_signs
    lag_time = np.arange(lag_n, dtype=np.int64)

    pregram_n = 10
    pregram_sq = 1.0e308
    pregram_amp = float(np.sqrt(pregram_sq))
    pregram_signs = np.where(np.arange(pregram_n) % 2 == 0, 1.0, -1.0)
    X_pregram_np = np.ones((pregram_n, 1), dtype=np.float64)
    resid_pregram_np = pregram_n * pregram_amp * pregram_signs
    pregram_time = np.arange(pregram_n, dtype=np.int64)

    component_n = 4
    component_sq = 5.0e307
    component_amp = float(np.sqrt(component_sq))
    X_component_np = np.ones((component_n, 1), dtype=np.float64)
    resid_component_np = component_n * component_amp * np.asarray([1.0, -1.0, 1.0, -1.0])
    component_unique = np.arange(component_n, dtype=np.int64)
    component_pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)

    tiny_design_value = 1.0e-320
    X_tiny_cluster_np = np.ones((4, 1), dtype=np.float64) * tiny_design_value
    resid_tiny_cluster_np = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)

    X_mixed_np = np.ones((3, 1), dtype=np.float64)
    resid_mixed_np = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    mixed_coarse = np.asarray([0, 0, 1], dtype=np.int64)
    mixed_unique = np.asarray([0, 1, 2], dtype=np.int64)
    mixed_time = np.asarray([0, 0, 1], dtype=np.int64)
    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)

    nonnested_n = 4
    nonnested_amplitude = 1.0e154
    nonnested_small = 1.0e-154
    X_nonnested_np = np.full((nonnested_n, 1), 0.5, dtype=np.float64)
    nonnested_scores = np.asarray(
        [-nonnested_amplitude, nonnested_small,
         nonnested_amplitude, -nonnested_small],
        dtype=np.float64,
    )
    resid_nonnested_np = 2.0 * nonnested_scores
    nonnested_safe_scores = np.asarray(
        [-1.0e150, 1.0e-150, 1.0e150, -1.0e-150], dtype=np.float64
    )
    resid_nonnested_safe_np = 2.0 * nonnested_safe_scores
    nonnested_debias_scores = np.asarray(
        [-nonnested_amplitude, nonnested_amplitude,
         nonnested_amplitude, nonnested_small], dtype=np.float64
    )
    resid_nonnested_debias_np = 2.0 * nonnested_debias_scores
    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    nonnested_cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)

    if backend == "numpy":
        xp = np
        X, resid = X_np, resid_np
        scores = scores_np
        X_hac, resid_hac = X_hac_np, resid_hac_np
        X_lag, resid_lag = X_lag_np, resid_lag_np
        X_pregram, resid_pregram = X_pregram_np, resid_pregram_np
        X_component, resid_component = X_component_np, resid_component_np
        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np
        X_mixed, resid_mixed = X_mixed_np, resid_mixed_np
        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np
        resid_nonnested_safe = resid_nonnested_safe_np
        resid_nonnested_debias = resid_nonnested_debias_np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        X, resid = cp.asarray(X_np), cp.asarray(resid_np)
        scores = cp.asarray(scores_np)
        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)
        X_lag, resid_lag = cp.asarray(X_lag_np), cp.asarray(resid_lag_np)
        X_pregram, resid_pregram = cp.asarray(X_pregram_np), cp.asarray(resid_pregram_np)
        X_component, resid_component = cp.asarray(X_component_np), cp.asarray(resid_component_np)
        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)
        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)
        X_mixed, resid_mixed = cp.asarray(X_mixed_np), cp.asarray(resid_mixed_np)
        X_nonnested = cp.asarray(X_nonnested_np)
        resid_nonnested = cp.asarray(resid_nonnested_np)
        resid_nonnested_safe = cp.asarray(resid_nonnested_safe_np)
        resid_nonnested_debias = cp.asarray(resid_nonnested_debias_np)
    elif backend == "torch":
        import torch
        xp = torch
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        resid = torch.as_tensor(resid_np, dtype=torch.float64, device="cuda")
        scores = torch.as_tensor(scores_np, dtype=torch.float64, device="cuda")
        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")
        resid_hac = torch.as_tensor(resid_hac_np, dtype=torch.float64, device="cuda")
        X_lag = torch.as_tensor(X_lag_np, dtype=torch.float64, device="cuda")
        resid_lag = torch.as_tensor(resid_lag_np, dtype=torch.float64, device="cuda")
        X_pregram = torch.as_tensor(X_pregram_np, dtype=torch.float64, device="cuda")
        resid_pregram = torch.as_tensor(resid_pregram_np, dtype=torch.float64, device="cuda")
        X_component = torch.as_tensor(X_component_np, dtype=torch.float64, device="cuda")
        resid_component = torch.as_tensor(resid_component_np, dtype=torch.float64, device="cuda")
        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")
        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")
        X_mixed = torch.as_tensor(X_mixed_np, dtype=torch.float64, device="cuda")
        resid_mixed = torch.as_tensor(resid_mixed_np, dtype=torch.float64, device="cuda")
        X_nonnested = torch.as_tensor(
            X_nonnested_np, dtype=torch.float64, device="cuda"
        )
        resid_nonnested = torch.as_tensor(
            resid_nonnested_np, dtype=torch.float64, device="cuda"
        )
        resid_nonnested_safe = torch.as_tensor(
            resid_nonnested_safe_np, dtype=torch.float64, device="cuda"
        )
        resid_nonnested_debias = torch.as_tensor(
            resid_nonnested_debias_np, dtype=torch.float64, device="cuda"
        )
    else:
        raise ValueError(backend)

    expected_cluster = np.asarray([[0.5 * amplitude * amplitude]])
    expected_hac = np.asarray([[influence_amplitude ** 2]])
    one_way = _array(clustered_covariance(X, resid, groups, xp=xp))
    two_way = _array(two_way_clustered_covariance(X, resid, groups, groups, xp=xp))
    cancellation = _array(
        _grouped_score_sums(scores, cancel_groups, n_groups=2, xp=xp)
    )
    hac = _array(hac_covariance(X_hac, resid_hac, bandwidth=1, xp=xp))
    dk = _array(driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=xp))
    lag_hac = _array(hac_covariance(X_lag, resid_lag, bandwidth=lag_bandwidth, xp=xp))
    lag_dk = _array(
        driscoll_kraay_covariance(
            X_lag, resid_lag, lag_time, bandwidth=lag_bandwidth, kernel="bartlett", xp=xp
        )
    )
    pregram_hac = _array(hac_covariance(X_pregram, resid_pregram, bandwidth=1, xp=xp))
    pregram_dk = _array(
        driscoll_kraay_covariance(
            X_pregram, resid_pregram, pregram_time, bandwidth=1, xp=xp
        )
    )
    component_reference = _array(
        clustered_covariance(X_component, resid_component, component_pairs, xp=xp)
    )
    component_two_way = _array(
        two_way_clustered_covariance(
            X_component,
            resid_component,
            component_unique,
            component_pairs,
            xp=xp,
        )
    )
    tiny_design_cluster = _array(
        clustered_covariance(
            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp
        )
    )
    mixed_cluster = _array(clustered_covariance(X_mixed, resid_mixed, mixed_coarse, xp=xp))
    mixed_two_way = _array(two_way_clustered_covariance(
        X_mixed, resid_mixed, mixed_unique, mixed_coarse, xp=xp
    ))
    mixed_two_way_permuted = _array(two_way_clustered_covariance(
        X_mixed, resid_mixed, mixed_nonmonotone_coarse, mixed_unique, xp=xp
    ))
    mixed_dk = _array(driscoll_kraay_covariance(
        X_mixed, resid_mixed, mixed_time, bandwidth=0, xp=xp
    ))
    nonnested_two_way = _array(two_way_clustered_covariance(
        X_nonnested, resid_nonnested, nonnested_cluster1, nonnested_cluster2, xp=xp
    ))
    nonnested_two_way_safe = _array(two_way_clustered_covariance(
        X_nonnested, resid_nonnested_safe, nonnested_cluster1, nonnested_cluster2, xp=xp
    ))
    nonnested_two_way_group_debias = _array(two_way_clustered_covariance(
        X_nonnested, resid_nonnested_debias, nonnested_cluster1, nonnested_cluster2,
        xp=xp, group_debias=True
    ))
    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster), ("mixed_cluster", mixed_cluster), ("mixed_two_way", mixed_two_way), ("mixed_two_way_permuted", mixed_two_way_permuted), ("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way), ("nonnested_two_way_safe", nonnested_two_way_safe), ("nonnested_two_way_group_debias", nonnested_two_way_group_debias)):
        if not np.all(np.isfinite(value)):
            raise AssertionError(f"{backend}: {name} produced non-finite covariance")
    np.testing.assert_allclose(one_way, expected_cluster, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(two_way, expected_cluster, rtol=8e-13, atol=0.0)
    np.testing.assert_array_equal(cancellation, np.zeros((2, 1)))
    np.testing.assert_allclose(hac, expected_hac, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(dk, expected_hac * (n / (n - 1.0)), rtol=8e-13, atol=0.0)
    expected_lag_hac = np.asarray([[5.4 * lag_influence_sq]], dtype=np.float64)
    np.testing.assert_allclose(lag_hac, expected_lag_hac, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(
        lag_dk,
        expected_lag_hac * (lag_n / (lag_n - 1.0)),
        rtol=8e-13,
        atol=0.0,
    )
    expected_pregram_hac = np.asarray([[pregram_sq]], dtype=np.float64)
    np.testing.assert_allclose(pregram_hac, expected_pregram_hac, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(
        pregram_dk,
        expected_pregram_hac * (pregram_n / (pregram_n - 1.0)),
        rtol=8e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        component_two_way, component_reference, rtol=8e-13, atol=0.0
    )
    np.testing.assert_allclose(
        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(mixed_cluster, np.asarray([[1.0e-200]]), rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(mixed_two_way, mixed_cluster, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(mixed_two_way_permuted, mixed_cluster, rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)
    np.testing.assert_allclose(
        nonnested_two_way, np.asarray([[-4.0]]), rtol=2e-12, atol=0.0
    )
    np.testing.assert_allclose(
        nonnested_two_way_safe, np.asarray([[-4.0]]), rtol=5e-13, atol=0.0
    )
    np.testing.assert_allclose(
        nonnested_two_way_group_debias, np.asarray([[6.0]]), rtol=5e-13, atol=0.0
    )
    return {
        "status": "success",
        "backend": backend,
        "one_way": one_way.tolist(),
        "two_way": two_way.tolist(),
        "group_cancellation": cancellation.tolist(),
        "hac": hac.tolist(),
        "driscoll_kraay": dk.tolist(),
        "lag_accumulator_hac": lag_hac.tolist(),
        "lag_accumulator_driscoll_kraay": lag_dk.tolist(),
        "pregram_hac": pregram_hac.tolist(),
        "pregram_driscoll_kraay": pregram_dk.tolist(),
        "two_way_component_cancellation": component_two_way.tolist(),
        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),
        "mixed_cluster": mixed_cluster.tolist(),
        "mixed_two_way": mixed_two_way.tolist(),
        "mixed_two_way_permuted": mixed_two_way_permuted.tolist(),
        "mixed_driscoll_kraay": mixed_dk.tolist(),
        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),
        "nonnested_two_way_safe_gram_cancellation": nonnested_two_way_safe.tolist(),
        "nonnested_two_way_group_debias_cancellation": nonnested_two_way_group_debias.tolist(),
    }



def _multiscale_grouping_audit(backend):
    scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0, 1.0], dtype=np.float64
    )
    groups = np.asarray([0, 0, 0, 1], dtype=np.int64)
    X_np = np.full((4, 1), 0.5, dtype=np.float64)
    resid_np = 2.0 * scores_np
    dummy = np.arange(4, dtype=np.int64)
    X, resid, _entity, _time = _to_backend(
        X_np, resid_np, dummy, dummy, backend
    )

    deep_scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0, 2, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 0, 0, 1, 0, 1, 2, 3], dtype=np.int64)
    X_deep_np = np.full((8, 1), 0.5, dtype=np.float64)
    deep_dummy = np.arange(8, dtype=np.int64)
    X_deep, deep_scores, _entity2, _time2 = _to_backend(
        X_deep_np,
        4.0 * deep_scores_np,
        deep_dummy,
        deep_dummy,
        backend,
    )

    if backend == "numpy":
        xp = np
        scores = scores_np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        scores = cp.asarray(scores_np)
    elif backend == "torch":
        import torch
        xp = torch
        scores = torch.as_tensor(
            scores_np, dtype=torch.float64, device="cuda"
        )
    else:
        raise ValueError(backend)

    grouped = _array(
        _grouped_score_sums(
            scores[:, None], groups, n_groups=2, xp=xp
        )
    )
    one_way = _array(
        clustered_covariance(X, resid, groups, xp=xp)
    )
    dk = _array(
        driscoll_kraay_covariance(
            X, resid, groups, bandwidth=0, xp=xp
        )
    )
    deep_two_way = _array(
        two_way_clustered_covariance(
            X_deep,
            deep_scores,
            cluster1,
            cluster2,
            xp=xp,
        )
    )

    unsafe_amplitude = 1.0e200
    unsafe_low1 = 1.0e108
    unsafe_low2 = unsafe_low1 * (1.0 + 1.0e-3)
    unsafe_scores_np = np.asarray(
        [
            -unsafe_amplitude, unsafe_low1, unsafe_amplitude, -unsafe_low1,
            -unsafe_amplitude, -unsafe_low2, unsafe_amplitude, unsafe_low2,
        ],
        dtype=np.float64,
    )
    unsafe_c1 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    unsafe_c2 = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    unsafe_X_np = np.full((8, 1), 0.5, dtype=np.float64)
    unsafe_dummy = np.arange(8, dtype=np.int64)
    unsafe_X, unsafe_resid, _ue, _ut = _to_backend(
        unsafe_X_np,
        4.0 * unsafe_scores_np,
        unsafe_dummy,
        unsafe_dummy,
        backend,
    )
    unsafe_two_way = _array(
        two_way_clustered_covariance(
            unsafe_X,
            unsafe_resid,
            unsafe_c1,
            unsafe_c2,
            xp=xp,
        )
    )
    unsafe_expected = np.asarray(
        [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
        dtype=np.float64,
    )

    tier_amplitude = 2.0 ** 660
    tier_middle = 2.0 ** 600
    tier_tiny = 2.0 ** 350
    tier_scores_np = np.asarray(
        [
            -tier_amplitude, tier_middle, tier_tiny,
            tier_amplitude, -tier_middle, -tier_tiny,
            -tier_amplitude, -tier_middle, tier_amplitude, tier_middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    tier_c1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    tier_c2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    tier_X_np = np.full((16, 1), 0.5, dtype=np.float64)
    tier_dummy = np.arange(16, dtype=np.int64)
    tier_X, tier_resid, _te, _tt = _to_backend(
        tier_X_np,
        8.0 * tier_scores_np,
        tier_dummy,
        tier_dummy,
        backend,
    )
    tier_two_way = _array(
        two_way_clustered_covariance(
            tier_X, tier_resid, tier_c1, tier_c2, xp=xp
        )
    )
    tier_expected = np.asarray(
        [[-4.0 * tier_amplitude * tier_tiny]], dtype=np.float64
    )

    np.testing.assert_array_equal(
        grouped, np.asarray([[1.0], [1.0]])
    )
    np.testing.assert_allclose(
        one_way, np.asarray([[2.0]]), rtol=8e-13, atol=0.0
    )
    np.testing.assert_allclose(
        dk, np.asarray([[8.0 / 3.0]]), rtol=8e-13, atol=0.0
    )
    np.testing.assert_allclose(
        deep_two_way, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        unsafe_two_way, unsafe_expected, rtol=4e-12, atol=0.0
    )
    np.testing.assert_allclose(
        tier_two_way, tier_expected, rtol=3e-12, atol=0.0
    )
    return {
        "status": "success",
        "backend": backend,
        "grouped": grouped.tolist(),
        "one_way": one_way.tolist(),
        "driscoll_kraay": dk.tolist(),
        "deep_two_way": deep_two_way.tolist(),
        "unsafe_cross_two_way": unsafe_two_way.tolist(),
        "third_tier_two_way": tier_two_way.tolist(),
    }

def _hausman_scale_audit(backend):
    results = {}
    for label, variance, difference in (
        ("large", 1.0e308, np.sqrt(1.0e308)),
        ("subnormal", 1.0e-320, np.sqrt(1.0e-320)),
    ):
        result = _hausman_quadratic([difference], [[variance]])
        if not result.applicable:
            raise AssertionError(
                f"{backend}: Hausman {label} scale became inapplicable: {result.reason}"
            )
        np.testing.assert_allclose(
            result.statistic, 1.0, rtol=3e-12, atol=0.0,
            err_msg=f"{backend}: Hausman {label} scale statistic",
        )
        if not np.isfinite(result.pvalue):
            raise AssertionError(f"{backend}: Hausman {label} p-value is non-finite")
        results[label] = {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "df": float(result.df),
        }
    np.testing.assert_allclose(
        results["large"]["pvalue"], results["subnormal"]["pvalue"],
        rtol=3e-12, atol=0.0,
    )
    singular = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e200]),
        np.diag(np.asarray([1.0e308, 0.0])),
    )
    if singular.applicable or not singular.reason or (
        "outside the identified covariance-difference range" not in singular.reason
    ):
        raise AssertionError(
            f"{backend}: large singular Hausman range guard failed: {singular}"
        )
    dense = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e154]),
        np.full((2, 2), 1.0e308, dtype=np.float64),
    )
    dense_basis = np.full(4, 0.5, dtype=np.float64)
    dense_range = _hausman_quadratic(
        np.asarray(
            [
                1.0e308 + 1.0e300,
                1.0e308 - 1.0e300,
                1.0e308,
                1.0e308,
            ],
            dtype=np.float64,
        ),
        1.0e308 * np.outer(dense_basis, dense_basis),
    )
    if dense_range.applicable or not dense_range.reason or (
        "outside the identified covariance-difference range"
        not in dense_range.reason
    ):
        raise AssertionError(
            f"{backend}: dense Hausman range projection failed closed"
        )
    if not dense.applicable:
        raise AssertionError(
            f"{backend}: dense large Hausman scale became inapplicable: {dense.reason}"
        )
    np.testing.assert_allclose(
        dense.statistic, 1.0, rtol=5e-13, atol=0.0,
        err_msg=f"{backend}: dense large Hausman statistic",
    )
    return {
        "status": "success",
        "backend": backend,
        "cases": results,
        "large_singular_range_rejected": True,
        "dense_large_statistic": float(dense.statistic),
        "dense_projection_range_rejected": True,
    }


def _zero_variance_inference_audit(backend):
    tiny = np.nextafter(0.0, 1.0)
    params_np = np.asarray([0.0, tiny, -tiny], dtype=np.float64)
    bse_np = np.zeros(3, dtype=np.float64)
    regular_params_np = np.asarray([2.0e-20, -3.0e-20], dtype=np.float64)
    regular_bse_np = np.asarray([1.0e-20, 1.0e-20], dtype=np.float64)
    if backend == "numpy":
        xp = np
        params, bse = params_np, bse_np
        regular_params, regular_bse = regular_params_np, regular_bse_np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        params, bse = cp.asarray(params_np), cp.asarray(bse_np)
        regular_params, regular_bse = cp.asarray(regular_params_np), cp.asarray(regular_bse_np)
    elif backend == "torch":
        import torch
        xp = torch
        params = torch.as_tensor(params_np, dtype=torch.float64, device="cuda")
        bse = torch.as_tensor(bse_np, dtype=torch.float64, device="cuda")
        regular_params = torch.as_tensor(regular_params_np, dtype=torch.float64, device="cuda")
        regular_bse = torch.as_tensor(regular_bse_np, dtype=torch.float64, device="cuda")
    else:
        raise ValueError(backend)

    exact = _array(_zero_safe_statistic_ratio(params, bse, xp))
    regular = _array(_zero_safe_statistic_ratio(regular_params, regular_bse, xp))
    if exact[0] != 0.0 or not np.isposinf(exact[1]) or not np.isneginf(exact[2]):
        raise AssertionError(f"{backend}: exact-zero inference semantics drifted: {exact}")
    np.testing.assert_allclose(regular, np.asarray([2.0, -3.0]), rtol=2.0e-15, atol=0.0)
    return {
        "status": "success",
        "backend": backend,
        "zero_coefficient_statistic": float(exact[0]),
        "positive_zero_variance_is_inf": bool(np.isposinf(exact[1])),
        "negative_zero_variance_is_inf": bool(np.isneginf(exact[2])),
        "tiny_positive_bse_statistics": regular.tolist(),
    }


def _tiny_design_lstsq_audit(backend):
    tiny = 1.0e-320
    X = np.eye(2, dtype=np.float64) * tiny
    y = np.asarray([tiny, 2.0 * tiny], dtype=np.float64)
    entity = np.arange(2, dtype=np.int64)
    time = np.arange(2, dtype=np.int64)
    Xb, yb, _eb, _tb = _to_backend(X, y, entity, time, backend)
    if backend == "torch":
        import torch
        params, ranks = panel_lstsq_batched(Xb[None, ...], yb[None, ...], torch)
        rank = int(_array(ranks)[0])
        params_np = _array(params)[0]
    else:
        import cupy as cp
        params, rank = panel_lstsq(Xb, yb, cp)
        rank = int(rank)
        params_np = _array(params)
    matrix_rank = int(panel_matrix_rank(Xb, torch if backend == "torch" else cp))
    if rank != 2 or matrix_rank != rank:
        raise AssertionError(
            f"{backend}: tiny-design rank policy drifted: solver={rank}, matrix_rank={matrix_rank}"
        )
    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)
    return {
        "status": "success", "backend": backend,
        "rank": rank, "matrix_rank": matrix_rank, "params": params_np.tolist(),
    }


def _gram_overflow_certificate_audit(backend):
    X = np.eye(2, dtype=np.float64) * 1.0e200
    y = np.asarray([1.0e200, 2.0e200], dtype=np.float64)
    entity = np.arange(2, dtype=np.int64)
    time = np.arange(2, dtype=np.int64)
    Xb, yb, _eb, _tb = _to_backend(X, y, entity, time, backend)
    if backend == "torch":
        import torch
        _candidate, certified = panel_lstsq_gram_certified_batched(
            Xb[None, ...], yb[None, ...], torch
        )
        params, ranks = panel_lstsq_batched(Xb[None, ...], yb[None, ...], torch)
        rank = int(_array(ranks)[0])
        params_np = _array(params)[0]
    else:
        import cupy as cp
        _candidate, certified = panel_lstsq_gram_certified_batched(
            Xb[None, ...], yb[None, ...], cp
        )
        params, rank = panel_lstsq(Xb, yb, cp)
        rank = int(rank)
        params_np = _array(params)
    if bool(_array(certified)[0]):
        raise AssertionError(f"{backend}: non-finite Gram batch was incorrectly certified")
    if rank != 2:
        raise AssertionError(f"{backend}: Gram-overflow SVD fallback rank drifted to {rank}")
    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)
    return {"status": "success", "backend": backend, "rank": rank, "params": params_np.tolist()}


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


def _connected_two_way_prediction_audit(backend):
    """Audit two-way normalization guards on a connected incidence graph."""
    rng = np.random.default_rng(20260824)
    # Use a connected, overidentified fixture. The former six-edge cycle was
    # saturated once standard FE nuisance rank was counted correctly (df=0).
    entity = np.repeat(np.arange(3, dtype=np.int64), 3)
    time = np.tile(np.arange(3, dtype=np.int64), 3)
    X = rng.normal(size=(entity.size, 1)).astype(np.float64)
    alpha = np.array([0.4, -0.3, 0.8], dtype=np.float64)
    tau = np.array([0.25, -0.15, 0.45], dtype=np.float64)
    y = 0.75 * X[:, 0] + alpha[entity] + tau[time]
    y = (y + rng.normal(scale=0.02, size=entity.size)).astype(np.float64)
    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)
    model = PanelOLS(
        entity_effects=True, time_effects=True, cov_type="hc0", device=_device(backend)
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    if _backend_name(model) != backend:
        raise AssertionError("connected prediction audit fit backend provenance drifted")
    if int(model.df_resid) <= 0:
        raise AssertionError("connected prediction audit fixture has no residual df")
    diagnostic = model.fit_statistics_.metadata.get("diagnostic_df", {})
    if int(diagnostic.get("incidence_components", -1)) != 1:
        raise AssertionError("connected prediction audit fixture is not connected")
    both_known = model.predict(
        Xb[:1], entity_ids=np.array([0]), time_ids=np.array([1])
    )
    def guarded(label, **kwargs):
        try:
            model.predict(Xb[:1], **kwargs)
        except ValueError as exc:
            if "both entity and time labels are known" not in str(exc):
                raise AssertionError(f"{label}: wrong two-way prediction failure: {exc}") from exc
            return True
        raise AssertionError(f"{label}: two-way partial-label prediction did not fail closed")
    guards = {
        "entity_only": guarded("entity_only", entity_ids=np.array([0])),
        "time_only": guarded("time_only", time_ids=np.array([0])),
        "known_entity_unknown_time": guarded(
            "known_entity_unknown_time", entity_ids=np.array([0]), time_ids=np.array([99])
        ),
        "unknown_entity_known_time": guarded(
            "unknown_entity_known_time", entity_ids=np.array([99]), time_ids=np.array([0])
        ),
    }
    both_unseen = model.predict(
        Xb[:1], entity_ids=np.array([98]), time_ids=np.array([99])
    )
    prediction_backend = getattr(model, "_predict_backend_name", None)
    if prediction_backend != backend:
        raise AssertionError("connected prediction audit backend provenance drifted")
    return {
        "status": "success",
        "executed_backend": backend,
        "prediction_backend": prediction_backend,
        "both_known": _array(both_known),
        "both_unseen": _array(both_unseen),
        "guards": guards,
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
            if "both entity and time labels are known" not in str(exc):
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
    connected_prediction_reference = _connected_two_way_prediction_audit("numpy")
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

        connected_audit = _connected_two_way_prediction_audit(backend)
        if connected_audit["executed_backend"] != backend or connected_audit["prediction_backend"] != backend:
            raise AssertionError("connected prediction backend provenance drifted")
        if not all(connected_audit["guards"].values()):
            raise AssertionError("connected prediction partial-label guards did not fail closed")
        connected_diffs = {}
        for field in ("both_known", "both_unseen"):
            np.testing.assert_allclose(
                connected_audit[field], connected_prediction_reference[field],
                rtol=args.rtol, atol=args.atol,
                err_msg=f"two_way_connected_prediction.{field}",
            )
            connected_diffs[field] = _max_abs(
                connected_audit[field], connected_prediction_reference[field]
            )
        payload["prediction_contracts"]["two_way_connected_partial_labels"] = {
            "status": "success",
            "executed_backend": connected_audit["executed_backend"],
            "prediction_backend": connected_audit["prediction_backend"],
            "guards": dict(connected_audit["guards"]),
            "max_abs_differences": connected_diffs,
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
        diagnostic_scale = _diagnostic_scale_audit(backend)
        diagnostic_reference = _diagnostic_scale_audit("numpy")
        diagnostic_diffs = {}
        for field in (
            "pooling_f_statistic", "pooling_f_pvalue",
            "classical_f_statistic", "classical_f_pvalue",
            "bp_lm_statistic", "bp_lm_pvalue",
        ):
            tolerance = 5e-4 if field.startswith("classical_f") else max(args.rtol, 5e-8)
            np.testing.assert_allclose(
                diagnostic_scale[field], diagnostic_reference[field],
                rtol=tolerance, atol=max(args.atol, 5e-8),
                err_msg=f"diagnostic_scale.{field}",
            )
            diagnostic_diffs[field] = abs(
                float(diagnostic_scale[field]) - float(diagnostic_reference[field])
            )
        diagnostic_scale["max_abs_differences_vs_numpy"] = diagnostic_diffs
        payload["numerical_primitives"] = {
            "tiny_design_lstsq": _tiny_design_lstsq_audit(backend),
            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),
            "covariance_extreme_scale": _covariance_extreme_scale_audit(backend),
            "multiscale_grouping": _multiscale_grouping_audit(backend),
            "hausman_scale": _hausman_scale_audit(backend),
            "zero_variance_inference": _zero_variance_inference_audit(backend),
            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),
            "projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend),
            "fixed_effect_recovery_cancellation": _fixed_effect_recovery_cancellation_audit(backend),
            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),
            "diagnostic_scale_reductions": diagnostic_scale,
        }

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
