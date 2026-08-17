#!/usr/bin/env python3
"""Exact-head physical GPU validation for the PR126 Fama-MacBeth review fixes.

This runner verifies correctness/backend provenance for chronology, formula,
rank, exactly identified periods, no-intercept behavior, both covariance modes, and the standard inference
result surface. It also records synchronized timing for the rank-revealing
retained-period solve against the same-workload NumPy baseline. Timing is audit
evidence only; no universal speedup claim is derived from it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from statgpu.backends import _is_cupy_array, _is_torch_array, _to_numpy
from statgpu.panel import FamaMacBeth

SCHEMA_VERSION = 3
_REQUIRED_BACKENDS = {"cupy", "torch"}
_SNAPSHOT_KEYS = (
    "coef",
    "betas",
    "bse",
    "tvalues",
    "pvalues",
    "conf_int",
    "cov_params",
    "prediction",
)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device(backend: str) -> str:
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _arrays(X, y, backend: str):
    if backend == "numpy":
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(X, dtype=cp.float64), cp.asarray(y, dtype=cp.float64)
    if backend == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        )
    raise ValueError(backend)


def _sync(backend: str):
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch

        torch.cuda.synchronize()


def _validate_acceptance_backends(backends):
    normalized = [value.strip() for value in backends if value.strip()]
    if len(normalized) != 2 or set(normalized) != _REQUIRED_BACKENDS:
        raise ValueError(
            "physical acceptance requires exactly both GPU backends: cupy,torch"
        )
    return normalized


def _chronology_fixture():
    x_period = np.asarray([-2.0, -0.75, 0.25, 1.25, 2.5])
    period_params = ((0.2, 0.4), (1.1, -0.8), (-0.7, 0.3))
    x = np.tile(x_period, len(period_params))
    y = np.concatenate(
        [intercept + slope * x_period for intercept, slope in period_params]
    )
    labels = np.repeat(np.asarray(["t1", "t2", "t10"], dtype=object), x_period.size)
    numeric = np.repeat(np.arange(len(period_params)), x_period.size)
    return x[:, None], y, labels, numeric


def _ordered(labels):
    import pandas as pd

    return pd.Categorical(labels, categories=["t1", "t2", "t10"], ordered=True)


def _public_array(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _array_backend_name(value):
    if _is_cupy_array(value):
        return "cupy"
    if _is_torch_array(value):
        return "torch"
    return "numpy"


def _assert_backend_native_value(value, expected_backend: str, label: str):
    actual_backend = _array_backend_name(value)
    if actual_backend != expected_backend:
        raise AssertionError(
            f"{label}: expected backend {expected_backend}, got {actual_backend}"
        )
    if expected_backend == "torch":
        device = getattr(value, "device", None)
        if device is None or str(device).split(":", 1)[0] != "cuda":
            raise AssertionError(f"{label}: Torch output is not CUDA-resident: {device}")


def _assert_backend_native_outputs(model):
    expected = model._backend_name
    for name in (
        "coef_",
        "betas_",
        "bse_",
        "tvalues_",
        "pvalues_",
        "conf_int_",
        "cov_params_",
    ):
        _assert_backend_native_value(getattr(model, name), expected, name)
    inference_backend = getattr(model, "_inference_backend_name", None)
    if inference_backend != expected:
        raise AssertionError(
            f"inference backend provenance mismatch: {inference_backend} != {expected}"
        )
    result = getattr(model, "_inference_result", None)
    metadata_backend = None if result is None else result.metadata.get("inference_backend")
    if metadata_backend != expected:
        raise AssertionError(
            f"inference result metadata backend mismatch: {metadata_backend} != {expected}"
        )


def _inference_descriptor(model):
    result = getattr(model, "_inference_result", None)
    if result is None:
        raise AssertionError("FamaMacBeth did not publish _inference_result")
    if result.__class__.__name__ != "ParameterInferenceResult":
        raise AssertionError(
            f"unexpected inference result type: {result.__class__.__name__}"
        )

    public_internal = (
        (model.coef_, getattr(model, "_params", None), "_params"),
        (model.bse_, getattr(model, "_bse", None), "_bse"),
        (model.tvalues_, getattr(model, "_tvalues", None), "_tvalues"),
        (model.tvalues_, getattr(model, "_zvalues", None), "_zvalues"),
        (model.pvalues_, getattr(model, "_pvalues", None), "_pvalues"),
        (model.conf_int_, getattr(model, "_conf_int", None), "_conf_int"),
    )
    for public, internal, name in public_internal:
        if internal is None:
            raise AssertionError(f"standard inference alias missing: {name}")
        np.testing.assert_allclose(
            np.asarray(internal, dtype=np.float64),
            _public_array(public),
            rtol=0.0,
            atol=0.0,
        )

    np.testing.assert_allclose(result.params, model._params, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.bse, model._bse, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.statistic, model._tvalues, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.pvalues, model._pvalues, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.conf_int, model._conf_int, rtol=0.0, atol=0.0)

    feature_names = None if result.feature_names is None else list(result.feature_names)
    return {
        "result_type": result.__class__.__name__,
        "method": result.method,
        "statistic_name": result.statistic_name,
        "distribution": result.distribution,
        "df": result.df,
        "cov_type": result.cov_type,
        "feature_names": feature_names,
        "covariance_source": result.metadata.get("covariance_source"),
        "n_periods": result.metadata.get("n_periods"),
        "effective_bandwidth": result.metadata.get("effective_bandwidth"),
        "inference_backend": result.metadata.get("inference_backend"),
    }


def _assert_inference_descriptors(reference, actual):
    semantic_keys = set(reference).union(actual) - {"inference_backend"}
    reference_semantic = {key: reference.get(key) for key in semantic_keys}
    actual_semantic = {key: actual.get(key) for key in semantic_keys}
    if actual_semantic != reference_semantic:
        raise AssertionError(
            "standard inference descriptor mismatch: "
            f"actual={actual_semantic}, reference={reference_semantic}"
        )
    return actual


def _snapshot(model, prediction_input):
    # Validate the common inference container and private aliases every time a
    # numerical snapshot is taken; public arrays must remain on the fit backend.
    _inference_descriptor(model)
    _assert_backend_native_outputs(model)
    fit_ref = getattr(model, "_fit_ref_", None)
    if fit_ref is None or tuple(fit_ref.shape) != (0,):
        raise AssertionError(
            "FamaMacBeth must retain only a zero-length prediction device anchor"
        )
    prediction = model.predict(prediction_input)
    _assert_backend_native_value(prediction, model._backend_name, "prediction")
    return {
        "coef": _public_array(model.coef_),
        "betas": _public_array(model.betas_),
        "bse": _public_array(model.bse_),
        "tvalues": _public_array(model.tvalues_),
        "pvalues": _public_array(model.pvalues_),
        "conf_int": _public_array(model.conf_int_),
        "cov_params": _public_array(model.cov_params_),
        "prediction": _public_array(prediction),
    }


def _max_abs(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _assert_snapshot(reference, actual, *, rtol=5e-6, atol=5e-7):
    if set(reference) != set(_SNAPSHOT_KEYS) or set(actual) != set(_SNAPSHOT_KEYS):
        raise AssertionError("focused Fama-MacBeth snapshot schema is incomplete")
    diffs = {}
    for key in _SNAPSHOT_KEYS:
        np.testing.assert_allclose(actual[key], reference[key], rtol=rtol, atol=atol)
        diffs[key] = _max_abs(actual[key], reference[key])
    return diffs


def _chronology_case(backend: str):
    X, y, labels, numeric = _chronology_fixture()
    ordered = _ordered(labels)
    ref = FamaMacBeth(bandwidth=1, device="cpu").fit(X, y, time_ids=numeric)
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(bandwidth=1, device=_device(backend)).fit(
        Xb, yb, time_ids=ordered
    )
    lexical = FamaMacBeth(bandwidth=1, device="cpu").fit(
        X, y, time_ids=np.asarray(ordered, dtype=object)
    )
    if np.allclose(
        _public_array(actual.cov_params_),
        _public_array(lexical.cov_params_),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("chronology negative control lost power")
    prediction_X = X[:3]
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(ref), _inference_descriptor(actual)
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(ref, prediction_X), _snapshot(actual, prediction_X)
        ),
    }


def _formula_case(backend: str):
    import pandas as pd

    X, y, labels, numeric = _chronology_fixture()
    ordered = _ordered(labels)
    x = X[:, 0].copy()
    x[1] = np.nan
    data = pd.DataFrame({"y": y, "x": x})
    ref = FamaMacBeth(bandwidth=1, device="cpu").fit(
        formula="y ~ x", data=data, time_ids=numeric
    )
    actual = FamaMacBeth(bandwidth=1, device=_device(backend)).fit(
        formula="y ~ x", data=data, time_ids=ordered
    )
    prediction_data = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(ref), _inference_descriptor(actual)
    )
    if inference_result["feature_names"] != ["Intercept", "x"]:
        raise AssertionError(
            f"formula inference feature names drifted: {inference_result['feature_names']}"
        )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(ref, prediction_data), _snapshot(actual, prediction_data)
        ),
    }


def _nonrobust_case(backend: str):
    X, y, _labels, numeric = _chronology_fixture()
    ref = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(
        X, y, time_ids=numeric
    )
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(cov_type="nonrobust", device=_device(backend)).fit(
        Xb, yb, time_ids=numeric
    )
    prediction_X = X[:3]
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(ref), _inference_descriptor(actual)
    )
    if inference_result["statistic_name"] != "t":
        raise AssertionError("nonrobust inference must use t statistics")
    if inference_result["distribution"] != "t":
        raise AssertionError("nonrobust inference must use the t distribution")
    if inference_result["df"] != float(actual.n_periods - 1):
        raise AssertionError("nonrobust inference df must equal T-1")
    if inference_result["effective_bandwidth"] is not None:
        raise AssertionError("nonrobust inference must not report a Newey-West bandwidth")
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(ref, prediction_X), _snapshot(actual, prediction_X)
        ),
    }


def _rank_fixture():
    x = np.concatenate(
        [
            np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.ones(5),
            np.asarray([-1.5, -0.5, 0.5, 1.5, 2.5]),
        ]
    )
    time_ids = np.repeat(np.arange(3), 5)
    y = 0.5 + 0.8 * x + np.repeat(np.asarray([0.0, 0.4, -0.3]), 5)
    return x[:, None], y, time_ids


def _rank_rejection(backend: str):
    X, y, time_ids = _rank_fixture()
    Xb, yb = _arrays(X, y, backend)
    try:
        FamaMacBeth(device=_device(backend)).fit(Xb, yb, time_ids=time_ids)
    except ValueError as exc:
        if "rank deficient" not in str(exc):
            raise
        return True
    raise AssertionError("rank-deficient retained period was not rejected")


def _exact_period_fixture():
    # Automatic intercept plus two slopes gives k=3. The first period is a
    # full-rank 3x3 solve, while later periods are overidentified. This directly
    # exercises the n_t == k eligibility boundary on the requested backend.
    blocks = [
        np.asarray([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]),
        np.asarray([[-2.0, 0.5], [-0.5, -1.0], [0.4, 1.5], [1.3, -0.2], [2.2, 0.8]]),
        np.asarray([[-1.5, -0.7], [-0.3, 0.9], [0.6, -1.2], [1.4, 0.4], [2.5, 1.1]]),
    ]
    period_params = np.asarray(
        [[0.2, 1.1, -0.4], [1.0, -0.6, 0.8], [-0.7, 0.3, 1.2]],
        dtype=np.float64,
    )
    ys = []
    for X_t, beta_t in zip(blocks, period_params):
        design = np.column_stack([np.ones(X_t.shape[0]), X_t])
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise AssertionError("exact-period physical fixture lost full rank")
        ys.append(design @ beta_t)
    X = np.vstack(blocks).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    time_ids = np.concatenate(
        [np.full(block.shape[0], i, dtype=np.int64) for i, block in enumerate(blocks)]
    )
    return X, y, time_ids, period_params


def _exact_period_case(backend: str):
    X, y, time_ids, expected_betas = _exact_period_fixture()
    reference = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device="cpu"
    ).fit(X, y, time_ids=time_ids)
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device=_device(backend)
    ).fit(Xb, yb, time_ids=time_ids)
    if reference.n_periods != 3 or actual.n_periods != 3:
        raise AssertionError(
            f"exactly identified period was filtered: reference={reference.n_periods}, "
            f"actual={actual.n_periods}"
        )
    np.testing.assert_allclose(
        _public_array(reference.betas_), expected_betas, rtol=2e-12, atol=2e-13
    )
    _assert_backend_native_outputs(actual)
    prediction_X = X[:3]
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(reference), _inference_descriptor(actual)
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "period_observation_counts": [3, 5, 5],
        "design_columns_including_intercept": 3,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, prediction_X), _snapshot(actual, prediction_X)
        ),
    }


def _explicit_device_cross_container_case(backend: str):
    X, y, time_ids, _expected_betas = _exact_period_fixture()
    reference = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device="cpu"
    ).fit(X, y, time_ids=time_ids)
    if backend == "cupy":
        import torch

        foreign_X = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        foreign_y = torch.as_tensor(y, dtype=torch.float64, device="cuda")
    elif backend == "torch":
        import cupy as cp

        foreign_X = cp.asarray(X, dtype=cp.float64)
        foreign_y = cp.asarray(y, dtype=cp.float64)
    else:
        raise ValueError("cross-container physical case requires cupy or torch")

    actual = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device=_device(backend)
    ).fit(foreign_X, foreign_y, time_ids=time_ids)
    if actual._backend_name != backend or actual._inference_backend_name != backend:
        raise AssertionError(
            f"explicit {backend} request was overridden by foreign input container: "
            f"fit={actual._backend_name}, inference={actual._inference_backend_name}"
        )
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(reference), _inference_descriptor(actual)
    )
    return {
        "status": "success",
        "foreign_input_backend": "torch" if backend == "cupy" else "cupy",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, X[:3]), _snapshot(actual, foreign_X[:3])
        ),
    }


def _numeric_stability_case(backend: str):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    n_periods = 4
    X = np.tile(x_period, n_periods)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time_ids = np.repeat(np.arange(n_periods), x_period.size)

    reference = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(bandwidth=0, device=_device(backend)).fit(
        Xb, yb, time_ids=time_ids
    )
    if actual._backend_name != backend:
        raise AssertionError(
            f"numeric stability case requested {backend}, executed {actual._backend_name}"
        )
    if int(actual._period_svd_fallbacks) != n_periods:
        raise AssertionError(
            "non-finite Gram RHS must force every retained period to SVD fallback: "
            f"fallbacks={actual._period_svd_fallbacks}, periods={n_periods}"
        )
    for label, value in (
        ("betas", actual.betas_),
        ("coef", actual.coef_),
        ("cov_params", actual.cov_params_),
    ):
        if not np.all(np.isfinite(_public_array(value))):
            raise AssertionError(f"numeric stability {label} contains non-finite values")
    np.testing.assert_allclose(
        _public_array(actual.betas_)[:, 0],
        _public_array(reference.betas_)[:, 0],
        rtol=5e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _public_array(actual.coef_)[0],
        _public_array(reference.coef_)[0],
        rtol=5e-13,
        atol=0.0,
    )

    # Separately exercise the scaled coefficient covariance at a magnitude
    # where naive beta' beta overflows but the final covariance is representable.
    slopes = np.asarray([-1.0e154, 0.0, 1.0e154])
    X_cov = np.tile(x_period, slopes.size)[:, None]
    y_cov = np.concatenate([slope * x_period for slope in slopes])
    time_cov = np.repeat(np.arange(slopes.size), x_period.size)
    ref_cov = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X_cov, y_cov, time_ids=time_cov
    )
    X_cov_b, y_cov_b = _arrays(X_cov, y_cov, backend)
    actual_cov = FamaMacBeth(bandwidth=0, device=_device(backend)).fit(
        X_cov_b, y_cov_b, time_ids=time_cov
    )
    if not np.all(np.isfinite(_public_array(actual_cov.cov_params_))):
        raise AssertionError("scaled coefficient covariance is non-finite")
    np.testing.assert_allclose(
        _public_array(actual_cov.cov_params_),
        _public_array(ref_cov.cov_params_),
        rtol=2e-11,
        atol=0.0,
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "gram_rhs_overflow_svd_fallbacks": int(actual._period_svd_fallbacks),
        "n_periods": n_periods,
        "common_intercept": float(_public_array(actual.coef_)[0]),
        "scaled_covariance_slope_variance": float(
            _public_array(actual_cov.cov_params_)[1, 1]
        ),
    }


def _square_rank_rejection(backend: str):
    X, y, time_ids, _expected_betas = _exact_period_fixture()
    mask = time_ids == 0
    X[mask, 1] = 2.0 * X[mask, 0]
    Xb, yb = _arrays(X, y, backend)
    try:
        FamaMacBeth(device=_device(backend)).fit(Xb, yb, time_ids=time_ids)
    except ValueError as exc:
        if "rank deficient" not in str(exc):
            raise
        return True
    raise AssertionError(
        "square rank-deficient n_t == k period was filtered or accepted instead of rejected"
    )


def _no_intercept_rejections(backend: str):
    import pandas as pd

    X, y, _labels, numeric = _chronology_fixture()
    data = pd.DataFrame({"y": y, "x": X[:, 0]})
    result = {}
    for formula in ("y ~ 0 + x", "y ~ x - 1"):
        try:
            FamaMacBeth(device=_device(backend)).fit(
                formula=formula, data=data, time_ids=numeric
            )
        except ValueError as exc:
            if "no-intercept formulas are not supported" not in str(exc):
                raise
            result[formula] = True
        else:
            raise AssertionError(f"no-intercept formula was accepted: {formula}")
    return result


def _timing_fixture():
    rng = np.random.default_rng(20260816)
    n_times, per_period, p = 64, 128, 4
    time_ids = np.repeat(np.arange(n_times), per_period)
    X = rng.normal(size=(n_times * per_period, p))
    beta = np.asarray([0.7, -0.4, 0.25, 0.9])
    period_shift = np.repeat(rng.normal(scale=0.3, size=n_times), per_period)
    y = 0.5 + X @ beta + period_shift + rng.normal(scale=0.4, size=X.shape[0])
    return X.astype(np.float64), y.astype(np.float64), time_ids


def _timed_fit(X, y, time_ids, backend: str, warmup: int, repeats: int):
    Xb, yb = _arrays(X, y, backend)
    device = _device(backend)
    for _ in range(warmup):
        FamaMacBeth(bandwidth=2, device=device).fit(Xb, yb, time_ids=time_ids)
        _sync(backend)

    samples = []
    last = None
    for _ in range(repeats):
        _sync(backend)
        start = time.perf_counter()
        last = FamaMacBeth(bandwidth=2, device=device).fit(
            Xb, yb, time_ids=time_ids
        )
        _sync(backend)
        samples.append(time.perf_counter() - start)

    if last is None or last._backend_name != backend:
        raise AssertionError(
            f"requested {backend}, executed {getattr(last, '_backend_name', None)}"
        )
    if last._inference_backend_name != backend:
        raise AssertionError(
            f"requested {backend} inference, executed {last._inference_backend_name}"
        )
    _assert_backend_native_outputs(last)
    return last, samples


def _timing_case(backend: str, warmup: int, repeats: int):
    X, y, time_ids = _timing_fixture()
    reference, numpy_samples = _timed_fit(
        X, y, time_ids, "numpy", warmup=warmup, repeats=repeats
    )
    if backend == "numpy":
        candidate = reference
        backend_samples = list(numpy_samples)
    else:
        candidate, backend_samples = _timed_fit(
            X, y, time_ids, backend, warmup=warmup, repeats=repeats
        )

    prediction_X = X[:16]
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(reference), _inference_descriptor(candidate)
    )
    numerical_differences = _assert_snapshot(
        _snapshot(reference, prediction_X), _snapshot(candidate, prediction_X)
    )
    numpy_median = float(statistics.median(numpy_samples))
    backend_median = float(statistics.median(backend_samples))
    ratio = backend_median / numpy_median

    return {
        "status": "success",
        "executed_backend": candidate._backend_name,
        "inference_backend": candidate._inference_backend_name,
        "inference_result": inference_result,
        "n_times": 64,
        "observations_per_period": 128,
        "n_features": 4,
        "warmup": warmup,
        "repeats": repeats,
        "numpy_baseline": {
            "samples_seconds": numpy_samples,
            "median_seconds": numpy_median,
        },
        "backend_timing": {
            "samples_seconds": backend_samples,
            "median_seconds": backend_median,
        },
        "backend_over_numpy_median_ratio": float(ratio),
        "max_abs_differences_vs_numpy": numerical_differences,
        "optimization_notes": {
            "period_solver": (
                "one rank-revealing SVD per retained period; panel_lstsq uses the "
                "SVD factors directly without materializing an unused covariance bread"
            ),
            "rank_cutoff": (
                "singular-value cutoff remains on the active backend; only the final "
                "integer rank is extracted for fail-closed Python control flow"
            ),
            "distribution_inference": (
                "p-values and critical values use the selected NumPy/CuPy/Torch "
                "inference backend directly; GPU fits do not round-trip the statistic "
                "vector through NumPy/SciPy for distribution evaluation"
            ),
            "remaining_structure": (
                "retained periods are processed serially in Python, so small "
                "per-period regressions can remain launch/synchronization dominated"
            ),
            "interpretation": (
                "ratio > 1 means the requested backend is slower than NumPy on this "
                "fixture; no universal GPU speedup claim is made"
            ),
        },
    }


def _environment(backends):
    gpu_by_backend = {}
    cupy_version = None
    if "cupy" in backends:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy CUDA is unavailable")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        gpu_by_backend["cupy"] = name.decode() if isinstance(name, bytes) else name
        cupy_version = cp.__version__
    if "torch" in backends:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is unavailable")
        gpu_by_backend["torch"] = torch.cuda.get_device_name(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "patsy": _version("patsy"),
        "cupy": cupy_version,
        "torch": _version("torch"),
        "gpu_by_backend": gpu_by_backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    clean_before = _git_clean()
    if not clean_before:
        raise RuntimeError("physical acceptance requires a clean working tree")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    backends = _validate_acceptance_backends(args.backends.split(","))
    results = {}
    for backend in backends:
        chronology = _chronology_case(backend)
        formula = _formula_case(backend)
        nonrobust = _nonrobust_case(backend)
        if any(
            case["executed_backend"] != backend
            or case["inference_backend"] != backend
            for case in (chronology, formula, nonrobust)
        ):
            raise AssertionError(f"{backend}: fit/inference backend provenance mismatch")
        results[backend] = {
            "status": "success",
            "executed_backend": backend,
            "inference_backend": backend,
            "array_ordered_categorical": chronology,
            "formula_ordered_categorical_alignment": formula,
            "nonrobust_inference": nonrobust,
            "rank_deficient_retained_period_rejected": _rank_rejection(backend),
            "exactly_identified_full_rank_period": _exact_period_case(backend),
            "explicit_device_overrides_foreign_input_container": _explicit_device_cross_container_case(backend),
            "numeric_stability_and_gram_fallback": _numeric_stability_case(backend),
            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),
            "no_intercept_formula_rejections": _no_intercept_rejections(backend),
            "performance": _timing_case(backend, args.warmup, args.repeats),
        }

    clean_after_checks = _git_clean()
    if not clean_after_checks:
        raise RuntimeError("working tree changed during physical validation")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "required_backends": sorted(_REQUIRED_BACKENDS),
        "validated_backends": backends,
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after_checks": clean_after_checks,
        "status": "success",
        "validation_tier": "remote-full",
        "environment": _environment(backends),
        "timing_claim": (
            "same-workload synchronized NumPy/GPU timing for audit only; "
            "no universal speedup claim"
        ),
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — focused Fama-MacBeth GPU validation: {args.out}")


if __name__ == "__main__":
    main()
