#!/usr/bin/env python3
"""Collect raw PR79 accuracy evidence without dropping failed runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.runners.common import (
    make_case_id,
    make_method_config_id,
    make_raw_run,
    record_environment,
    safe_run,
    synchronized_time,
)


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "configs" / "expected_accuracy_manifest.json"
SUPPORTED_BACKENDS = {"numpy", "cupy", "torch"}


class RepositoryIntegrityError(RuntimeError):
    """Raised when canonical evidence cannot be tied to a clean Git tree."""


def _git_snapshot() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_project_root,
            text=True,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_project_root,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "git_sha": "unknown",
            "worktree_clean": False,
            "dirty_entries": [],
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }
    dirty_entries = [line for line in status.splitlines() if line]
    return {
        "git_sha": sha,
        "worktree_clean": not dirty_entries,
        "dirty_entries": dirty_entries,
        "inspection_error": None,
    }


def _repository_provenance(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    allow_dirty: bool,
) -> Dict[str, Any]:
    sha_unchanged = (
        initial.get("git_sha") == final.get("git_sha")
        and initial.get("git_sha") != "unknown"
    )
    snapshots_clean = (
        initial.get("worktree_clean") is True
        and final.get("worktree_clean") is True
    )
    return {
        "schema_version": "pr79-repository-provenance-1.0",
        "inspection": "git-status-porcelain-v1",
        "allow_dirty_requested": bool(allow_dirty),
        "sha_unchanged_during_collection": sha_unchanged,
        "canonical_eligible": snapshots_clean and sha_unchanged and not allow_dirty,
        "initial": dict(initial),
        "final": dict(final),
    }


def _require_collectable_snapshot(
    snapshot: Mapping[str, Any], *, allow_dirty: bool, phase: str
) -> None:
    if snapshot.get("worktree_clean") is True or allow_dirty:
        return
    error = snapshot.get("inspection_error")
    if error:
        detail = f"Git inspection failed: {error}"
    else:
        entries = list(snapshot.get("dirty_entries", []))
        preview = ", ".join(entries[:5])
        suffix = "" if len(entries) <= 5 else f" (+{len(entries) - 5} more)"
        detail = f"dirty entries: {preview}{suffix}"
    raise RepositoryIntegrityError(
        f"refusing canonical PR79 evidence from a non-clean repository "
        f"({phase}; {detail}); use --allow-dirty only for non-canonical local development"
    )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="full", help="manifest configuration name")
    parser.add_argument(
        "--backend",
        action="append",
        help="backend to collect (repeat or use a comma-separated value)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "permit local collection from a dirty tree; the artifact is marked "
            "non-canonical and cannot pass aggregation"
        ),
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_backends(
    requested: Optional[List[str]], configured: Iterable[str]
) -> List[str]:
    configured_list = list(configured)
    if not requested:
        return configured_list
    selected: List[str] = []
    for value in requested:
        selected.extend(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - SUPPORTED_BACKENDS)
    if unknown:
        raise ValueError("unsupported backend(s): " + ", ".join(unknown))
    disallowed = sorted(set(selected) - set(configured_list))
    if disallowed:
        raise ValueError(
            "backend(s) not declared by this manifest configuration: "
            + ", ".join(disallowed)
        )
    return list(dict.fromkeys(selected))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _case_specs() -> Dict[str, Dict[str, Any]]:
    from dev.benchmarks.pr79.generators.linear import (
        case_params_linear,
        case_params_linear_rank_def,
        case_params_linear_weighted,
        generate_linear_full_rank,
        generate_linear_rank_deficient,
        generate_linear_weighted,
    )
    from dev.benchmarks.pr79.generators.panel import (
        case_params_pooled,
        case_params_pooled_rank_def,
        generate_pooled_balanced,
        generate_pooled_rank_def,
    )
    from dev.benchmarks.pr79.generators.survival import (
        case_params_coxph_entry,
        case_params_coxph_no_ties,
        case_params_coxph_penalized,
        case_params_coxph_small_ties,
        generate_coxph_entry,
        generate_coxph_no_ties,
        generate_coxph_penalized,
        generate_coxph_small_ties,
    )

    return {
        "linear-fr": {
            "model_id": "LinearRegression",
            "case_params": case_params_linear,
            "generate": lambda: generate_linear_full_rank(1000, 10, 42),
            "cov_type": "nonrobust",
        },
        "linear-rd": {
            "model_id": "LinearRegression",
            "case_params": case_params_linear_rank_def,
            "generate": lambda: generate_linear_rank_deficient(200, 6, 42),
            "cov_type": "nonrobust",
            "rank_deficient": True,
        },
        "linear-wt": {
            "model_id": "LinearRegression",
            "case_params": case_params_linear_weighted,
            "generate": lambda: generate_linear_weighted(500, 8, 42),
            "cov_type": "nonrobust",
            "weighted": True,
        },
        "linear-rd-hc1": {
            "model_id": "LinearRegression",
            "case_params": case_params_linear_rank_def,
            "generate": lambda: generate_linear_rank_deficient(200, 6, 42),
            "cov_type": "hc1",
            "rank_deficient": True,
        },
        "cox-no-ties": {
            "model_id": "CoxPH",
            "case_params": case_params_coxph_no_ties,
            "generate": lambda: generate_coxph_no_ties(200, 4, 42),
        },
        "cox-small-ties": {
            "model_id": "CoxPH",
            "case_params": case_params_coxph_small_ties,
            "generate": lambda: generate_coxph_small_ties(300, 4, 42, 3),
        },
        "cox-entry": {
            "model_id": "CoxPH",
            "case_params": case_params_coxph_entry,
            "generate": lambda: generate_coxph_entry(200, 4, 42),
        },
        "cox-pen": {
            "model_id": "CoxPH",
            "case_params": case_params_coxph_penalized,
            "generate": lambda: generate_coxph_penalized(100, 8, 42),
        },
        "pooled-bal": {
            "model_id": "PooledOLS",
            "case_params": case_params_pooled,
            "generate": lambda: generate_pooled_balanced(30, 5, 3, 42),
            "cov_type": "nonrobust",
        },
        "pooled-rd": {
            "model_id": "PooledOLS",
            "case_params": case_params_pooled_rank_def,
            "generate": lambda: generate_pooled_rank_def(20, 5, 4, 45),
            "cov_type": "nonrobust",
            "rank_deficient": True,
        },
    }


def _prepare_case(label: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    generated = spec["generate"]()
    model_id = spec["model_id"]
    parameters = spec["case_params"]()
    if model_id == "LinearRegression":
        X, y = generated[0], generated[1]
        sample_weight = generated[3] if spec.get("weighted") else None
        inputs = {"X": X, "y": y, "sample_weight": sample_weight}
    elif model_id == "CoxPH":
        X, time, event = generated[0], generated[1], generated[2]
        entry = generated[3] if parameters.get("entry") else None
        inputs = {"X": X, "time": time, "event": event, "entry": entry}
    elif model_id == "PooledOLS":
        X, y, entity, time_index = generated[:4]
        inputs = {
            "X": X,
            "y": y,
            "entity": entity,
            "time_index": time_index,
            "cluster": None,
        }
    else:
        raise ValueError(f"unsupported model in case {label}: {model_id}")
    return {
        "case_label": label,
        "case_id": make_case_id(parameters),
        "model_id": model_id,
        "parameters": parameters,
        "inputs": inputs,
    }


def _method_config(
    spec: Mapping[str, Any], case: Mapping[str, Any], backend: str
) -> Dict[str, Any]:
    model_id = spec["model_id"]
    config: Dict[str, Any] = {"model_id": model_id, "backend": backend}
    if model_id == "LinearRegression":
        config.update({
            "cov_type": spec.get("cov_type", "nonrobust"),
            "compute_inference": True,
            "weighted": bool(spec.get("weighted")),
            "rank_deficient": bool(spec.get("rank_deficient")),
        })
    elif model_id == "CoxPH":
        parameters = case["parameters"]
        config.update({
            "ties": parameters.get("ties", "efron"),
            "penalty": float(parameters.get("penalty", 0.0)),
            "entry": bool(parameters.get("entry")),
            "cov_type": "nonrobust",
            "compute_inference": True,
        })
    else:
        config.update({
            "cov_type": spec.get("cov_type", "nonrobust"),
            "rank_deficient": bool(spec.get("rank_deficient")),
        })
    return config


def _run_case(
    spec: Mapping[str, Any],
    case: Mapping[str, Any],
    backend: str,
    warmup: int,
    iterations: int,
) -> List[Dict[str, Any]]:
    inputs = case["inputs"]
    model_id = spec["model_id"]
    if model_id == "LinearRegression":
        return _bench_linear(
            inputs["X"],
            inputs["y"],
            backend,
            inputs.get("sample_weight"),
            warmup,
            iterations,
            spec.get("cov_type", "nonrobust"),
        )
    if model_id == "CoxPH":
        parameters = case["parameters"]
        return _bench_coxph(
            inputs["X"],
            inputs["time"],
            inputs["event"],
            backend,
            inputs.get("entry"),
            float(parameters.get("penalty", 0.0)),
            warmup,
            iterations,
        )
    return _bench_pooled(
        inputs["X"],
        inputs["y"],
        inputs["entity"],
        inputs["time_index"],
        inputs.get("cluster"),
        backend,
        warmup,
        iterations,
    )


def collect_accuracy(
    *,
    config_name: str,
    manifest: Mapping[str, Any],
    backends: Optional[List[str]] = None,
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    initial_snapshot = _git_snapshot()
    _require_collectable_snapshot(
        initial_snapshot, allow_dirty=allow_dirty, phase="before collection"
    )
    configurations = manifest.get("configurations", {})
    if config_name not in configurations:
        raise ValueError(f"unknown accuracy configuration: {config_name}")
    config = configurations[config_name]
    selected = _selected_backends(backends, config.get("backends", []))
    iterations = int(config.get("iterations", 1))
    warmup = int(config.get("warmup", 0))
    if iterations < 1 or warmup < 0:
        raise ValueError("manifest iterations/warmup are invalid")

    specs = _case_specs()
    runs: List[Dict[str, Any]] = []
    case_evidence: Dict[str, Dict[str, Any]] = {}
    for label in config.get("cases", []):
        if label not in specs:
            raise ValueError(f"manifest references unknown case: {label}")
        spec = specs[label]
        case = _prepare_case(label, spec)
        declared_case = manifest.get("cases", {}).get(label, {})
        declared_id = declared_case.get("case_id")
        if declared_id and declared_id != case["case_id"]:
            raise ValueError(
                f"manifest case_id drift for {label}: {declared_id} != {case['case_id']}"
            )
        case_evidence[case["case_id"]] = _jsonable(case)
        print(f"--- {label} ({case['case_id']}) ---")
        for backend in selected:
            method = _method_config(spec, case, backend)
            bench_result, error = safe_run(
                _run_case, spec, case, backend, warmup, iterations
            )
            if error is not None:
                print(f"  {backend}: ERROR {error['error_type']}: {error['error']}")
                for iteration in range(iterations):
                    parameters = {**method, "iteration": iteration}
                    runs.append(make_raw_run(
                        f"{label}-{backend}-{iteration}",
                        case["case_id"],
                        make_method_config_id(method),
                        spec["model_id"],
                        "statgpu",
                        backend,
                        parameters,
                        None,
                        None,
                        status="error",
                        error=error["error"],
                        error_type=error["error_type"],
                        traceback_text=error["traceback"],
                    ))
                continue
            if len(bench_result) != iterations:
                raise RuntimeError(
                    f"{label}/{backend} returned {len(bench_result)} runs, expected {iterations}"
                )
            for measured in bench_result:
                iteration = int(measured["iteration"])
                parameters = {**method, "iteration": iteration}
                runs.append(make_raw_run(
                    f"{label}-{backend}-{iteration}",
                    case["case_id"],
                    make_method_config_id(method),
                    spec["model_id"],
                    "statgpu",
                    backend,
                    parameters,
                    {"fit_warm_s": measured["fit_time_s"]},
                    measured["results"],
                ))
            median = float(np.median([item["fit_time_s"] for item in bench_result]))
            print(f"  {backend}: {median * 1000:.1f} ms")

    environment = record_environment()
    final_snapshot = _git_snapshot()
    _require_collectable_snapshot(
        final_snapshot, allow_dirty=allow_dirty, phase="after collection"
    )
    provenance = _repository_provenance(
        initial_snapshot, final_snapshot, allow_dirty=allow_dirty
    )
    sha = str(initial_snapshot.get("git_sha", "unknown"))
    if not provenance["sha_unchanged_during_collection"] and not allow_dirty:
        raise RepositoryIntegrityError(
            "refusing canonical PR79 evidence because HEAD changed during collection"
        )
    return {
        "source_schema_version": "pr79-benchmark-source-2.1",
        "benchmark_session_id": f"pr79-{sha[:7]}-{config_name}",
        "git_sha": sha,
        "repository_provenance": provenance,
        "configuration": config_name,
        "selected_backends": selected,
        "environment": environment,
        "cases": case_evidence,
        "runs": runs,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = _load_manifest(args.manifest)
    try:
        raw = collect_accuracy(
            config_name=args.config,
            manifest=manifest,
            backends=args.backend,
            allow_dirty=args.allow_dirty,
        )
    except RepositoryIntegrityError as exc:
        print(f"PR79 accuracy collection refused: {exc}", file=sys.stderr)
        return 2
    output = args.output or Path("results/pr79/accuracy") / (
        f"{args.config}_accuracy_results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, allow_nan=False)
        handle.write("\n")
    failed = sum(1 for run in raw["runs"] if run["status"] != "success")
    print(f"Saved {len(raw['runs'])} raw runs ({failed} errors): {output}")
    return 0


def _backend_inputs(X, y, backend, sample_weight=None):
    if backend == "cupy":
        import cupy as cp

        return (
            cp.asarray(X),
            cp.asarray(y),
            cp.asarray(sample_weight) if sample_weight is not None else None,
        )
    if backend == "torch":
        import torch

        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            torch.as_tensor(sample_weight, dtype=torch.float64, device="cuda")
            if sample_weight is not None
            else None,
        )
    return X, y, sample_weight


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract(model) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    covariance = getattr(model, "_var_matrix", None)
    if covariance is not None:
        covariance_array = _to_numpy(covariance).astype(np.float64)
        results["_info_cond"] = float(np.linalg.cond(covariance_array))
    attributes = (
        "coef_",
        "intercept_",
        "rank_",
        "rsquared",
        "aic",
        "bic",
        "_df_model",
        "_df_resid",
        "_bse",
        "_pvalues",
        "_log_likelihood",
        "_penalized_objective",
        "_final_kkt_inf",
        "_final_kkt_normalized",
        "_var_matrix",
        "_converged",
        "_termination_reason",
        "_iterations",
    )
    for attribute in attributes:
        value = getattr(model, attribute, None)
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            results[attribute] = value
            continue
        converted = _to_numpy(value)
        results[attribute] = converted.tolist() if converted.ndim else converted.item()
    if "_bse" not in results and getattr(model, "bse_", None) is not None:
        results["_bse"] = _to_numpy(model.bse_).tolist()
    return results


def _add_prediction_contract(
    results: Dict[str, Any], predictions: Any, y: np.ndarray
) -> None:
    prediction_array = _to_numpy(predictions).astype(np.float64).reshape(-1)
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    if prediction_array.shape != y_array.shape:
        raise ValueError("prediction shape does not match y")
    results["predictions"] = prediction_array.tolist()
    results["residual_sum_squares"] = float(np.sum((y_array - prediction_array) ** 2))


def _require_finite_results(results: Mapping[str, Any]) -> None:
    for name, value in results.items():
        if value is None or isinstance(value, (str, bool)):
            continue
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FloatingPointError(f"result {name} is not numeric") from exc
        if not np.isfinite(array).all():
            raise FloatingPointError(f"result {name} contains NaN or Inf")


def _bench_linear(
    X,
    y,
    backend,
    sample_weight=None,
    n_warm=0,
    n_meas=1,
    cov_type="nonrobust",
):
    from statgpu.linear_model import LinearRegression

    X_device, y_device, weight_device = _backend_inputs(X, y, backend, sample_weight)
    device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    measured = []
    for iteration in range(n_warm + n_meas):
        model = LinearRegression(
            fit_intercept=True,
            cov_type=cov_type,
            compute_inference=True,
            device=device,
        )
        _, elapsed = synchronized_time(
            model.fit, X_device, y_device, sample_weight=weight_device
        )
        if iteration >= n_warm:
            results = _extract(model)
            _add_prediction_contract(results, model.predict(X_device), y)
            _require_finite_results(results)
            measured.append({
                "iteration": iteration - n_warm,
                "fit_time_s": round(elapsed, 6),
                "results": results,
            })
    return measured


def _bench_coxph(
    X,
    time,
    event,
    backend,
    entry=None,
    penalty=0.0,
    n_warm=0,
    n_meas=1,
):
    from statgpu.survival import CoxPH

    X_device, _, _ = _backend_inputs(X, np.zeros_like(time), backend)
    device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    measured = []
    for iteration in range(n_warm + n_meas):
        model = CoxPH(
            ties="efron",
            penalty=penalty,
            compute_inference=True,
            device=device,
            compute_cindex=False,
            tol=1e-6,
            max_iter=30,
        )
        _, elapsed = synchronized_time(
            model.fit, X_device, time=time, event=event, entry=entry
        )
        if iteration >= n_warm:
            results = _extract(model)
            risk_score = model.predict_risk_score(X_device)
            results["predictions"] = _to_numpy(risk_score).astype(np.float64).reshape(-1).tolist()
            _require_finite_results(results)
            measured.append({
                "iteration": iteration - n_warm,
                "fit_time_s": round(elapsed, 6),
                "results": results,
            })
    return measured


def _bench_pooled(
    X,
    y,
    entity,
    time_index,
    cluster=None,
    backend="numpy",
    n_warm=0,
    n_meas=1,
):
    from statgpu.panel import PooledOLS

    X_device, y_device, _ = _backend_inputs(X, y, backend)
    device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    covariance = "clustered" if cluster is not None else "nonrobust"
    measured = []
    # PooledOLS + GPU: pass entity/time_index explicitly to avoid
    # internal np.asarray(cupy_array) triggers on CuPy 13.x.
    entity_device = _to_numpy(entity) if backend != "numpy" else entity
    time_device = _to_numpy(time_index) if backend != "numpy" else time_index

    for iteration in range(n_warm + n_meas):
        model = PooledOLS(cov_type=covariance, device=device)
        _, elapsed = synchronized_time(
            model.fit,
            X_device,
            y_device,
            cluster=cluster if cluster is not None else None,
            time_index=time_device,
        )
        if iteration >= n_warm:
            results = _extract(model)
            _add_prediction_contract(results, model.predict(X_device), y)
            _require_finite_results(results)
            measured.append({
                "iteration": iteration - n_warm,
                "fit_time_s": round(elapsed, 6),
                "results": results,
            })
    return measured


if __name__ == "__main__":
    raise SystemExit(main())
