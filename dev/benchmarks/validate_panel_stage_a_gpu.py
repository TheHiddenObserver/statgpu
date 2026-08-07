#!/usr/bin/env python3
"""Physical CuPy/Torch validation for Panel P1 Stage A (Issue #93 / PR #119).

This is a correctness/backend acceptance script, not a performance benchmark.
It runs the behavior-preserving Stage-A panel refactor on deterministic panel
data, compares CuPy/Torch results with the NumPy reference, and records exact
source/environment provenance.
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

from statgpu.backends import _to_numpy
from statgpu.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dataset():
    rng = np.random.default_rng(20260807)
    n_entities, n_times = 8, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    entity_effect = np.repeat(rng.normal(scale=0.6, size=n_entities), n_times)
    time_effect = np.tile(np.linspace(-0.25, 0.30, n_times), n_entities)
    y = (
        0.4
        + 1.15 * X[:, 0]
        - 0.65 * X[:, 1]
        + entity_effect
        + 0.20 * time_effect
        + rng.normal(scale=0.16, size=entity.size)
    )
    return X.astype(np.float64), y.astype(np.float64), entity, time


def _to_backend(X, y, backend):
    if backend == "numpy":
        return X, y
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(X), cp.asarray(y)
    if backend == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
        )
    raise ValueError(backend)


def _device_arg(backend):
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _backend_name(model):
    if isinstance(model, FamaMacBeth):
        return model._backend_name
    return model._get_backend(backend="auto").name


def _array(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _snapshot(model, prediction):
    payload = {
        "coef": _array(model.coef_).ravel(),
        "bse": _array(model.bse_).ravel(),
        "tvalues": _array(model.tvalues_).ravel(),
        "pvalues": _array(model.pvalues_).ravel(),
        "conf_int": _array(model.conf_int_),
        "prediction": _array(prediction).ravel(),
        "df_resid": int(model.df_resid),
        "nobs": int(model.nobs),
    }
    if hasattr(model, "rsquared"):
        payload["rsquared"] = float(model.rsquared)
    if hasattr(model, "rsquared_within") and model.rsquared_within is not None:
        payload["rsquared_within"] = float(model.rsquared_within)
    if hasattr(model, "theta_") and model.theta_ is not None:
        payload["theta"] = float(model.theta_)
    if hasattr(model, "variance_components_") and model.variance_components_ is not None:
        payload["variance_components"] = {
            key: float(value)
            for key, value in model.variance_components_.items()
        }
    if hasattr(model, "n_periods"):
        payload["n_periods"] = int(model.n_periods)
    return payload


def _cases(X, y, entity, time, backend):
    Xb, yb = _to_backend(X, y, backend)
    device = _device_arg(backend)
    two_way = np.column_stack([entity, time])

    cases = {}

    model = PooledOLS(cov_type="nonrobust", device=device).fit(Xb, yb)
    cases["pooled_nonrobust"] = (model, model.predict(X[:5]))

    model = PooledOLS(cov_type="robust", device=device).fit(Xb, yb)
    cases["pooled_robust"] = (model, model.predict(X[:5]))

    model = PooledOLS(cov_type="clustered", device=device).fit(
        Xb, yb, cluster=entity
    )
    cases["pooled_clustered"] = (model, model.predict(X[:5]))

    model = PooledOLS(cov_type="hac", bandwidth=2, device=device).fit(
        Xb, yb, time_index=time
    )
    cases["pooled_hac"] = (model, model.predict(X[:5]))

    model = BetweenOLS(cov_type="robust", device=device).fit(
        Xb, yb, entity_ids=entity
    )
    cases["between_robust"] = (model, model.predict(X[:5]))

    model = FirstDifferenceOLS(cov_type="robust", device=device).fit(
        Xb, yb, entity_ids=entity, time_ids=time
    )
    cases["first_difference_robust"] = (model, model.predict(X[:5]))

    model = PanelOLS(entity_effects=True, cov_type="robust", device=device).fit(
        Xb, yb, entity_ids=entity
    )
    cases["panel_entity_robust"] = (
        model,
        model.predict(X[:5], entity_ids=entity[:5]),
    )

    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="clustered",
        device=device,
    ).fit(
        Xb,
        yb,
        entity_ids=entity,
        time_ids=time,
        cluster=two_way,
    )
    cases["panel_two_way_clustered"] = (
        model,
        model.predict(X[:5], entity_ids=entity[:5], time_ids=time[:5]),
    )

    model = RandomEffects(device=device).fit(Xb, yb, entity_ids=entity)
    cases["random_effects"] = (model, model.predict(X[:5]))

    model = FamaMacBeth(cov_type="newey-west", bandwidth=2, device=device).fit(
        Xb, yb, time_ids=time
    )
    cases["fama_macbeth_newey_west"] = (model, model.predict(X[:5]))

    return cases


def _compare(reference, candidate, *, rtol, atol):
    differences = {}
    for key in ("coef", "bse", "tvalues", "pvalues", "conf_int", "prediction"):
        np.testing.assert_allclose(
            candidate[key], reference[key], rtol=rtol, atol=atol,
            err_msg=f"mismatch in {key}",
        )
        differences[key] = float(
            np.max(np.abs(np.asarray(candidate[key]) - np.asarray(reference[key])))
        )
    for key in ("df_resid", "nobs", "n_periods"):
        if key in reference:
            assert candidate[key] == reference[key], (key, candidate[key], reference[key])
    for key in ("rsquared", "rsquared_within", "theta"):
        if key in reference:
            np.testing.assert_allclose(candidate[key], reference[key], rtol=rtol, atol=atol)
            differences[key] = float(abs(candidate[key] - reference[key]))
    if "variance_components" in reference:
        for name, value in reference["variance_components"].items():
            np.testing.assert_allclose(
                candidate["variance_components"][name], value, rtol=rtol, atol=atol
            )
            differences[f"variance_components.{name}"] = float(
                abs(candidate["variance_components"][name] - value)
            )
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
            name: _version(name)
            for name in ("statgpu", "numpy", "scipy", "cupy", "torch")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backends", default="cupy,torch",
        help="Comma-separated physical GPU backends to validate against NumPy.",
    )
    parser.add_argument("--expected-sha", default=None)
    parser.add_argument("--rtol", type=float, default=5e-6)
    parser.add_argument("--atol", type=float, default=5e-7)
    args = parser.parse_args()

    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or any(value not in {"cupy", "torch"} for value in backends):
        raise ValueError("--backends must contain cupy and/or torch")

    sha = _git_sha()
    if args.expected_sha is not None and sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")

    X, y, entity, time = _dataset()
    reference_models = _cases(X, y, entity, time, "numpy")
    references = {
        name: _snapshot(model, prediction)
        for name, (model, prediction) in reference_models.items()
    }

    results = {}
    for backend in backends:
        backend_models = _cases(X, y, entity, time, backend)
        backend_results = {}
        for name, (model, prediction) in backend_models.items():
            actual_backend = _backend_name(model)
            if actual_backend != backend:
                raise AssertionError(
                    f"{name}: requested {backend}, executed {actual_backend}"
                )
            snapshot = _snapshot(model, prediction)
            differences = _compare(
                references[name], snapshot, rtol=args.rtol, atol=args.atol
            )
            backend_results[name] = {
                "status": "success",
                "executed_backend": actual_backend,
                "max_abs_differences": differences,
            }
        results[backend] = backend_results

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "status": "success",
        "environment": _environment(backends),
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Panel Stage A physical GPU validation: {args.out}")


if __name__ == "__main__":
    main()
