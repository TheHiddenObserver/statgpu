#!/usr/bin/env python3
"""Focused physical GPU gate for disconnected two-way PanelOLS rank handling.

This complements ``validate_panel_stage_b_gpu.py`` after the PR #122
Ready-for-review fix that moved the component-aware ``N + T - C`` rank ahead
of the fixed-effects residual-df feasibility gate.  The fixture is deliberately
chosen so the historical count gives residual df 0 while the correct
component-aware rank gives residual df 1.
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
from statgpu.panel import PanelOLS


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fixture():
    # Two disconnected 2x2 blocks plus one singleton cell.
    # N = T = 5, C = 3, n = 9, k = 1.
    # Historical nuisance count: (N - 1) + (T - 1) = 8 -> df = 0.
    # Correct effect rank: N + T - C = 7 -> df = 1.
    entity = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 2, 3, 2, 3, 4], dtype=np.int64)
    X = np.asarray(
        [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0],
        dtype=np.float64,
    ).reshape(-1, 1)
    y = np.asarray(
        [1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0, 0.0],
        dtype=np.float64,
    )
    return X, y, entity, time


def _to_backend_arrays(X, y, entity, time, backend):
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


def _device_arg(backend):
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _backend_name(model):
    return model._get_backend(backend="auto").name


def _array(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _test_result(result):
    return {
        "applicable": bool(result.applicable),
        "reason": result.reason,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
        "distribution": result.distribution,
    }


def _snapshot(model):
    fit = model.fit_statistics_
    metadata = fit.metadata
    diagnostic_df = metadata["diagnostic_df"]
    return {
        "coef": _array(model.coef_).ravel().tolist(),
        "bse": _array(model.bse_).ravel().tolist(),
        "tvalues": _array(model.tvalues_).ravel().tolist(),
        "pvalues": _array(model.pvalues_).ravel().tolist(),
        "conf_int": _array(model.conf_int_).tolist(),
        "nobs": int(model.nobs),
        "df_resid": int(model.df_resid),
        "rsquared_within": float(model.rsquared_within),
        "fit_statistics": {
            "rsquared_within": fit.rsquared_within,
            "rsquared_between": fit.rsquared_between,
            "rsquared_overall": fit.rsquared_overall,
            "rsquared_adj": fit.rsquared_adj,
            "f_statistic": fit.f_statistic,
            "f_pvalue": fit.f_pvalue,
            "f_df": None if fit.f_df is None else list(fit.f_df),
        },
        "diagnostic_df": {
            "effect_rank": int(diagnostic_df["effect_rank"]),
            "incidence_components": diagnostic_df["incidence_components"],
            "rank_x": int(diagnostic_df["rank_x"]),
            "df_resid": int(diagnostic_df["df_resid"]),
            "df_total": int(diagnostic_df["df_total"]),
        },
        "legacy_df_resid": int(metadata["legacy_df_resid"]),
        "public_df_resid_basis": metadata["public_df_resid_basis"],
        "pooling_f": _test_result(model.pooling_f_test()),
    }


def _fit(backend):
    X, y, entity, time = _fixture()
    Xb, yb, eb, tb = _to_backend_arrays(X, y, entity, time, backend)
    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="nonrobust",
        device=_device_arg(backend),
    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
    return model, _snapshot(model)


def _assert_structural_contract(snapshot, *, label):
    if snapshot["legacy_df_resid"] != 0:
        raise AssertionError(f"{label}: expected legacy df 0")
    if snapshot["public_df_resid_basis"] != "component-aware":
        raise AssertionError(f"{label}: component-aware public df not used")
    if snapshot["df_resid"] != 1:
        raise AssertionError(f"{label}: expected public df_resid=1")
    diagnostic = snapshot["diagnostic_df"]
    expected = {
        "effect_rank": 7,
        "incidence_components": 3,
        "rank_x": 1,
        "df_resid": 1,
    }
    for name, value in expected.items():
        if diagnostic[name] != value:
            raise AssertionError(
                f"{label}: diagnostic_df[{name!r}]={diagnostic[name]!r} != {value!r}"
            )


def _compare(reference, candidate, *, rtol, atol, label):
    differences = {}
    for field in ("coef", "bse", "tvalues", "pvalues", "conf_int"):
        actual = np.asarray(candidate[field], dtype=np.float64)
        expected = np.asarray(reference[field], dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"{label}.{field}",
        )
        differences[field] = float(np.max(np.abs(actual - expected)))

    for field in ("nobs", "df_resid", "legacy_df_resid"):
        if candidate[field] != reference[field]:
            raise AssertionError(
                f"{label}.{field}: {candidate[field]!r} != {reference[field]!r}"
            )

    if candidate["public_df_resid_basis"] != reference["public_df_resid_basis"]:
        raise AssertionError(f"{label}.public_df_resid_basis mismatch")
    if candidate["diagnostic_df"] != reference["diagnostic_df"]:
        raise AssertionError(f"{label}.diagnostic_df mismatch")

    for field, expected in reference["fit_statistics"].items():
        actual = candidate["fit_statistics"][field]
        if expected is None:
            if actual is not None:
                raise AssertionError(f"{label}.fit_statistics.{field} expected None")
        elif field == "f_df":
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                err_msg=f"{label}.fit_statistics.{field}",
            )
            differences[f"fit_statistics.{field}"] = float(abs(actual - expected))

    ref_pool = reference["pooling_f"]
    cand_pool = candidate["pooling_f"]
    for field in ("applicable", "reason", "df", "distribution"):
        if cand_pool[field] != ref_pool[field]:
            raise AssertionError(f"{label}.pooling_f.{field} mismatch")
    for field in ("statistic", "pvalue"):
        expected = ref_pool[field]
        actual = cand_pool[field]
        if expected is None:
            if actual is not None:
                raise AssertionError(f"{label}.pooling_f.{field} expected None")
        else:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            differences[f"pooling_f.{field}"] = float(abs(actual - expected))
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
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--rtol", type=float, default=5e-6)
    parser.add_argument("--atol", type=float, default=5e-7)
    args = parser.parse_args()

    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or any(value not in {"cupy", "torch"} for value in backends):
        raise ValueError("--backends must contain cupy and/or torch")

    sha = _git_sha()
    clean = _git_status_porcelain() == ""
    if sha != args.expected_sha:
        raise RuntimeError(f"HEAD {sha} != --expected-sha {args.expected_sha}")
    if not clean:
        raise RuntimeError("working tree must be clean for physical validation")

    reference_model, reference = _fit("numpy")
    if _backend_name(reference_model) != "numpy":
        raise RuntimeError("NumPy reference did not execute on NumPy")
    _assert_structural_contract(reference, label="numpy")

    result = {
        "schema_version": 1,
        "validation": "panel_stage_b_disconnected_two_way_fe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "working_tree_clean": clean,
        "status": "success",
        "protocol": {
            "reference_backend": "numpy",
            "requested_backends": backends,
            "rtol": args.rtol,
            "atol": args.atol,
            "timing_collected": False,
            "fixture": "two disconnected 2x2 blocks plus one singleton",
            "expected_rank_contract": {
                "nobs": 9,
                "n_entities": 5,
                "n_times": 5,
                "incidence_components": 3,
                "effect_rank": 7,
                "rank_x": 1,
                "legacy_df_resid": 0,
                "component_aware_df_resid": 1,
            },
        },
        "environment": _environment(backends),
        "reference": reference,
        "backend_results": {},
    }

    for backend in backends:
        try:
            model, snapshot = _fit(backend)
            executed_backend = _backend_name(model)
            if executed_backend != backend:
                raise AssertionError(
                    f"requested {backend}, executed backend {executed_backend!r}"
                )
            _assert_structural_contract(snapshot, label=backend)
            differences = _compare(
                reference,
                snapshot,
                rtol=args.rtol,
                atol=args.atol,
                label=backend,
            )
            result["backend_results"][backend] = {
                "status": "success",
                "executed_backend": executed_backend,
                "snapshot": snapshot,
                "differences_vs_numpy": differences,
            }
        except Exception as exc:
            result["status"] = "failed"
            result["backend_results"][backend] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
