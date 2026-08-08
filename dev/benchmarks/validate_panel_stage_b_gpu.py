#!/usr/bin/env python3
"""Physical CuPy/Torch acceptance for Panel Tier-1 Stage B (Issue #93).

This is a correctness/backend-provenance gate, not a performance benchmark.
It validates the new parameter-based fit statistics and specification tests on
balanced and unbalanced panels against the NumPy implementation while proving
that requested CuPy/Torch CUDA backends actually execute. It also rechecks the
maintained coefficient-inference outputs so Stage-B integration cannot regress
the Stage-A bse/t/p/CI/df contracts.
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dataset(seed: int, *, unbalanced: bool):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 9, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    entity_effect = np.repeat(np.linspace(-0.7, 0.8, n_entities), n_times)
    time_effect = np.tile(np.linspace(-0.22, 0.27, n_times), n_entities)
    y = (
        0.9 * X[:, 0]
        - 0.4 * X[:, 1]
        + entity_effect
        + 0.25 * time_effect
        + rng.normal(scale=0.18, size=entity.size)
    )
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 8, 17, 31, 44]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X.astype(np.float64), y.astype(np.float64), entity, time


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
    if isinstance(model, FamaMacBeth):
        return model._backend_name
    return model._get_backend(backend="auto").name


def _array(value):
    return np.asarray(_to_numpy(value), dtype=np.float64)


def _fit_stats(model):
    result = model.fit_statistics_
    payload = {
        "rsquared_within": result.rsquared_within,
        "rsquared_between": result.rsquared_between,
        "rsquared_overall": result.rsquared_overall,
        "rsquared_adj": result.rsquared_adj,
        "f_statistic": result.f_statistic,
        "f_pvalue": result.f_pvalue,
        "f_df": None if result.f_df is None else tuple(float(x) for x in result.f_df),
    }
    return payload


def _test_result(result):
    return {
        "applicable": bool(result.applicable),
        "reason": result.reason,
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
        "distribution": result.distribution,
    }


def _model_snapshot(model):
    payload = {
        "coef": _array(model.coef_).ravel(),
        "bse": _array(model.bse_).ravel(),
        "tvalues": _array(model.tvalues_).ravel(),
        "pvalues": _array(model.pvalues_).ravel(),
        "conf_int": _array(model.conf_int_),
        "nobs": int(model.nobs),
        "df_resid": int(model.df_resid),
        "fit_statistics": _fit_stats(model),
    }
    covariance = getattr(model, "_panel_cov_params", None)
    if covariance is not None:
        payload["diagnostic_covariance"] = _array(covariance)
    if hasattr(model, "pooling_f_test") and isinstance(model, PanelOLS):
        payload["pooling_f"] = _test_result(model.pooling_f_test())
    if hasattr(model, "breusch_pagan_lm_test") and isinstance(model, PooledOLS):
        payload["bp_lm"] = _test_result(model.breusch_pagan_lm_test())
    if isinstance(model, RandomEffects):
        meta = model.fit_statistics_.metadata
        model_f_meta = meta.get("model_f", {})
        payload["random_effects_diagnostic_contract"] = {
            "has_explicit_constant": bool(meta.get("has_explicit_constant")),
            "constant_column_index": meta.get("constant_column_index"),
            "restricted_rank": int(meta.get("restricted_rank", 0)),
            "model_f_rank_restricted": int(model_f_meta.get("rank_restricted", 0)),
            "model_f_restricted_design_supplied": bool(
                model_f_meta.get("restricted_design_supplied", False)
            ),
        }
    return payload


def _fit_cases(X, y, entity, time, backend, *, unbalanced):
    Xb, yb, eb, tb = _to_backend_arrays(X, y, entity, time, backend)
    device = _device_arg(backend)
    suffix = "unbalanced" if unbalanced else "balanced"
    cases = {}

    pooled = PooledOLS(device=device).fit(Xb, yb, entity_ids=eb)
    cases[f"pooled_{suffix}"] = pooled

    scrambled_time_np = (3 * time + 2 * entity + 1) % 11
    if backend == "numpy":
        scrambled_time = scrambled_time_np
    elif backend == "cupy":
        import cupy as cp

        scrambled_time = cp.asarray(scrambled_time_np, dtype=cp.int64)
    else:
        import torch

        scrambled_time = torch.as_tensor(
            scrambled_time_np, dtype=torch.int64, device="cuda"
        )
    pooled_hac = PooledOLS(cov_type="hac", bandwidth=2, device=device).fit(
        Xb,
        yb,
        time_index=scrambled_time,
        entity_ids=eb,
    )
    cases[f"pooled_hac_unsorted_{suffix}"] = pooled_hac

    between = BetweenOLS(cov_type="robust", device=device).fit(
        Xb, yb, entity_ids=eb
    )
    cases[f"between_{suffix}"] = between

    first_diff = FirstDifferenceOLS(cov_type="robust", device=device).fit(
        Xb, yb, entity_ids=eb, time_ids=tb
    )
    cases[f"first_difference_{suffix}"] = first_diff

    fe = PanelOLS(entity_effects=True, cov_type="nonrobust", device=device).fit(
        Xb, yb, entity_ids=eb
    )
    cases[f"panel_entity_{suffix}"] = fe

    re = RandomEffects(device=device).fit(Xb, yb, entity_ids=eb)
    cases[f"random_effects_{suffix}"] = re

    # Exercise the explicit-constant RandomEffects branch on the physical GPU.
    # This is intentionally a separate case so the no-intercept Stage-A path
    # remains independently frozen by the ordinary RandomEffects case above.
    X_constant = np.column_stack([np.ones(X.shape[0]), X[:, 0]])
    Xcb, ycb, ecb, _ = _to_backend_arrays(
        X_constant, y, entity, time, backend
    )
    re_constant = RandomEffects(device=device).fit(Xcb, ycb, entity_ids=ecb)
    cases[f"random_effects_explicit_constant_{suffix}"] = re_constant

    fmb = FamaMacBeth(cov_type="newey-west", bandwidth=2, device=device).fit(
        Xb,
        yb,
        time_ids=tb,
        entity_ids=eb,
    )
    cases[f"fama_macbeth_{suffix}"] = fmb

    diagnostics = {
        f"hausman_{suffix}": _test_result(fe.hausman_test(re)),
    }

    if not unbalanced:
        two_way = PanelOLS(
            entity_effects=True,
            time_effects=True,
            cov_type="nonrobust",
            device=device,
        ).fit(Xb, yb, entity_ids=eb, time_ids=tb)
        cases["panel_two_way_balanced"] = two_way

    return cases, diagnostics


def _scalar_diff(actual, expected, *, rtol, atol, label):
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{label}: expected None, got {actual}")
        return 0.0
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)
    return float(abs(float(actual) - float(expected)))


def _compare_test_result(reference, candidate, *, rtol, atol, label):
    if candidate["applicable"] != reference["applicable"]:
        raise AssertionError(
            f"{label}: applicability {candidate['applicable']} != {reference['applicable']}"
        )
    if candidate["distribution"] != reference["distribution"]:
        raise AssertionError(f"{label}: distribution mismatch")
    if candidate["reason"] != reference["reason"]:
        raise AssertionError(
            f"{label}: reason {candidate['reason']!r} != {reference['reason']!r}"
        )
    differences = {}
    for field in ("statistic", "pvalue"):
        differences[field] = _scalar_diff(
            candidate[field],
            reference[field],
            rtol=rtol,
            atol=atol,
            label=f"{label}.{field}",
        )
    ref_df = reference["df"]
    cand_df = candidate["df"]
    if ref_df is None:
        if cand_df is not None:
            raise AssertionError(f"{label}.df expected None")
    elif isinstance(ref_df, (tuple, list)):
        np.testing.assert_allclose(cand_df, ref_df, rtol=0, atol=0)
    else:
        np.testing.assert_allclose(cand_df, ref_df, rtol=0, atol=0)
    return differences


def _max_abs_difference(actual, expected):
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _compare_model(reference, candidate, *, rtol, atol, label):
    differences = {}
    for field in ("coef", "bse", "tvalues", "pvalues", "conf_int"):
        np.testing.assert_allclose(
            candidate[field],
            reference[field],
            rtol=rtol,
            atol=atol,
            err_msg=f"{label}.{field}",
        )
        differences[field] = _max_abs_difference(
            candidate[field], reference[field]
        )

    for field in ("nobs", "df_resid"):
        if int(candidate[field]) != int(reference[field]):
            raise AssertionError(
                f"{label}.{field}: {candidate[field]} != {reference[field]}"
            )
        differences[field] = 0.0

    for field, expected in reference["fit_statistics"].items():
        actual = candidate["fit_statistics"][field]
        if field == "f_df":
            if expected is None:
                if actual is not None:
                    raise AssertionError(f"{label}.fit_statistics.f_df expected None")
            else:
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
            continue
        differences[f"fit_statistics.{field}"] = _scalar_diff(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            label=f"{label}.fit_statistics.{field}",
        )

    if "diagnostic_covariance" in reference:
        np.testing.assert_allclose(
            candidate["diagnostic_covariance"],
            reference["diagnostic_covariance"],
            rtol=rtol,
            atol=atol,
            err_msg=f"{label}.diagnostic_covariance",
        )
        differences["diagnostic_covariance"] = _max_abs_difference(
            candidate["diagnostic_covariance"],
            reference["diagnostic_covariance"],
        )

    if "random_effects_diagnostic_contract" in reference:
        if candidate.get("random_effects_diagnostic_contract") != reference[
            "random_effects_diagnostic_contract"
        ]:
            raise AssertionError(
                f"{label}.random_effects_diagnostic_contract mismatch: "
                f"{candidate.get('random_effects_diagnostic_contract')} != "
                f"{reference['random_effects_diagnostic_contract']}"
            )
        differences["random_effects_diagnostic_contract"] = 0.0

    for test_name in ("pooling_f", "bp_lm"):
        if test_name in reference:
            nested = _compare_test_result(
                reference[test_name],
                candidate[test_name],
                rtol=rtol,
                atol=atol,
                label=f"{label}.{test_name}",
            )
            differences.update(
                {f"{test_name}.{name}": value for name, value in nested.items()}
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
    parser.add_argument("--backends", default="cupy,torch")
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--rtol", type=float, default=5e-6)
    parser.add_argument("--atol", type=float, default=5e-7)
    args = parser.parse_args()

    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or any(value not in {"cupy", "torch"} for value in backends):
        raise ValueError("--backends must contain cupy and/or torch")

    sha = _git_sha()
    if sha != args.expected_sha:
        raise RuntimeError(f"wrong source head: {sha} != {args.expected_sha}")
    dirty = _git_status_porcelain()
    if dirty.strip():
        raise RuntimeError(
            "physical acceptance requires a clean working tree; uncommitted changes:\n"
            + dirty
        )

    datasets = {
        "balanced": _dataset(20260808, unbalanced=False),
        "unbalanced": _dataset(20260809, unbalanced=True),
    }
    reference_models = {}
    reference_diagnostics = {}
    for name, (X, y, entity, time) in datasets.items():
        models, diagnostics = _fit_cases(
            X, y, entity, time, "numpy", unbalanced=(name == "unbalanced")
        )
        reference_models.update(
            {case: _model_snapshot(model) for case, model in models.items()}
        )
        reference_diagnostics.update(diagnostics)

    results = {}
    for backend in backends:
        backend_payload = {"models": {}, "diagnostics": {}}
        for name, (X, y, entity, time) in datasets.items():
            models, diagnostics = _fit_cases(
                X, y, entity, time, backend, unbalanced=(name == "unbalanced")
            )
            for case, model in models.items():
                actual_backend = _backend_name(model)
                if actual_backend != backend:
                    raise AssertionError(
                        f"{case}: requested {backend}, executed {actual_backend}"
                    )
                snapshot = _model_snapshot(model)
                differences = _compare_model(
                    reference_models[case],
                    snapshot,
                    rtol=args.rtol,
                    atol=args.atol,
                    label=case,
                )
                backend_payload["models"][case] = {
                    "status": "success",
                    "executed_backend": actual_backend,
                    "max_abs_differences": differences,
                }
            for case, result in diagnostics.items():
                differences = _compare_test_result(
                    reference_diagnostics[case],
                    result,
                    rtol=args.rtol,
                    atol=args.atol,
                    label=case,
                )
                backend_payload["diagnostics"][case] = {
                    "status": "success",
                    "max_abs_differences": differences,
                    "applicable": result["applicable"],
                    "reason": result["reason"],
                }
        results[backend] = backend_payload

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": sha,
        "working_tree_clean": True,
        "status": "success",
        "environment": _environment(backends),
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "datasets": {
            name: {"nobs": int(len(values[1]))}
            for name, values in datasets.items()
        },
        "backends": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"PASS — Panel Stage B physical GPU validation: {args.out}")


if __name__ == "__main__":
    main()
