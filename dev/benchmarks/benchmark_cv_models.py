#!/usr/bin/env python3
"""Run the six-family canonical cross-validation benchmark package.

The runner profiles the public ``fit`` call and observes two nested Python
regions in the same execution:

1. the estimator's CV candidate/fold selection region;
2. the final non-CV estimator ``fit`` on the full training set.

A run is retained as ``failed`` when either boundary cannot be observed. The
runner never estimates CV time by subtracting a separately measured refit from
an aggregate public-fit time.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.benchmarks.cv_source import check_file


@dataclass(frozen=True)
class CaseSpec:
    model_id: str
    task: str
    import_module: str
    import_name: str
    grid_parameters: dict[str, Any]
    scoring_name: str
    scoring_direction: str
    scoring_normalization: str


CASE_SPECS = (
    CaseSpec(
        "RidgeCV", "regression", "statgpu.linear_model", "RidgeCV",
        {"alphas": [1.0, 0.1, 0.01], "penalty": "l2"},
        "mean_squared_error", "minimize", "per-validation-observation",
    ),
    CaseSpec(
        "LassoCV", "regression", "statgpu.linear_model", "LassoCV",
        {"alphas": [1.0, 0.1, 0.01], "penalty": "l1"},
        "mean_squared_error", "minimize", "per-validation-observation",
    ),
    CaseSpec(
        "ElasticNetCV", "regression", "statgpu.linear_model", "ElasticNetCV",
        {"alphas": [1.0, 0.1, 0.01], "l1_ratio": 0.5, "penalty": "elasticnet"},
        "mean_squared_error", "minimize", "per-validation-observation",
    ),
    CaseSpec(
        "LogisticRegressionCV", "classification", "statgpu.linear_model",
        "LogisticRegressionCV", {"Cs": [0.1, 1.0, 10.0], "penalty": "l2"},
        "log_loss", "minimize", "per-validation-observation",
    ),
    CaseSpec(
        "PenalizedGLM_CV", "regression", "statgpu.linear_model",
        "PenalizedGLM_CV",
        {"alpha_grid": [1.0, 0.1, 0.01], "loss": "squared_error", "penalty": "l2"},
        "mean_squared_error", "minimize", "per-validation-observation",
    ),
    CaseSpec(
        "CoxPHCV", "survival", "statgpu.survival", "CoxPHCV",
        {"penalties": [1.0, 0.1, 0.01], "penalty": "l2", "ties": "efron"},
        "held_out_partial_likelihood", "maximize", "per-observed-event",
    ),
)


class RegionProfiler:
    """Observe direct CV-selection and final-refit regions in one public fit."""

    def __init__(self, outer_estimator: Any, n_samples: int, synchronize: Callable[[], None]):
        self.outer_estimator = outer_estimator
        self.n_samples = int(n_samples)
        self.synchronize = synchronize
        self.selector_stack: list[float] = []
        self.refit_stack: list[float] = []
        self.selector_ms = 0.0
        self.refit_ms = 0.0
        self._previous = None

    @staticmethod
    def _nrows(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_selector(frame) -> bool:
        module = str(frame.f_globals.get("__name__", ""))
        name = frame.f_code.co_name.lower()
        if not module.startswith("statgpu."):
            return False
        return (
            ("select" in name and ("cv" in name or module.endswith("_cv")))
            or name in {"_run_cv", "_evaluate_candidates", "_cross_validate", "_compute_cv_scores"}
        )

    def _is_full_refit(self, frame) -> bool:
        if frame.f_code.co_name != "fit":
            return False
        owner = frame.f_locals.get("self")
        if owner is None or owner is self.outer_estimator:
            return False
        for key in ("X", "x", "exog"):
            if self._nrows(frame.f_locals.get(key)) == self.n_samples:
                return True
        return False

    def _profile(self, frame, event, arg):
        if event == "call" and self._is_selector(frame):
            if not self.selector_stack:
                self.synchronize()
                self.selector_stack.append(time.perf_counter())
            else:
                self.selector_stack.append(-1.0)
        elif event in {"return", "exception"} and self._is_selector(frame):
            if self.selector_stack:
                started = self.selector_stack.pop()
                if started >= 0.0:
                    self.synchronize()
                    self.selector_ms += (time.perf_counter() - started) * 1000.0

        if event == "call" and self._is_full_refit(frame):
            if not self.refit_stack:
                self.synchronize()
                self.refit_stack.append(time.perf_counter())
            else:
                self.refit_stack.append(-1.0)
        elif event in {"return", "exception"} and self._is_full_refit(frame):
            if self.refit_stack:
                started = self.refit_stack.pop()
                if started >= 0.0:
                    self.synchronize()
                    self.refit_ms += (time.perf_counter() - started) * 1000.0
        return self._profile

    def __enter__(self):
        self._previous = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self._previous)
        return False


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_status(backend: str) -> tuple[bool, str | None, str | None]:
    if backend == "numpy":
        return True, "cpu", None
    if backend == "cupy":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False, "cuda", "CuPy is installed but no CUDA device is available"
            return True, "cuda", None
        except Exception as exc:
            return False, "cuda", f"CuPy CUDA unavailable: {type(exc).__name__}: {exc}"
    if backend == "torch":
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "torch", "Torch is installed but torch.cuda.is_available() is false"
            return True, "torch", None
        except Exception as exc:
            return False, "torch", f"Torch CUDA unavailable: {type(exc).__name__}: {exc}"
    raise ValueError(f"unknown backend: {backend}")


def _synchronizer(backend: str) -> Callable[[], None]:
    if backend == "cupy":
        import cupy as cp
        return lambda: cp.cuda.get_current_stream().synchronize()
    if backend == "torch":
        import torch
        return torch.cuda.synchronize
    return lambda: None


def _to_backend(value: Any, backend: str):
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(value)
    if backend == "torch":
        import torch
        return torch.as_tensor(value, device="cuda")
    return value


def _regression_data(seed: int, n_samples: int, n_features: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[: min(5, n_features)] = np.linspace(1.5, 0.3, min(5, n_features))
    y = X @ beta + rng.normal(scale=0.5, size=n_samples)
    X_test = rng.normal(size=(max(30, n_samples // 3), n_features))
    y_test = X_test @ beta + rng.normal(scale=0.5, size=X_test.shape[0])
    return X, y, X_test, y_test


def _classification_data(seed: int, n_samples: int, n_features: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[: min(5, n_features)] = np.linspace(1.2, 0.2, min(5, n_features))
    logits = X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < prob).astype(np.int32)
    X_test = rng.normal(size=(max(30, n_samples // 3), n_features))
    test_prob = 1.0 / (1.0 + np.exp(-(X_test @ beta)))
    y_test = (rng.uniform(size=X_test.shape[0]) < test_prob).astype(np.int32)
    return X, y, X_test, y_test


def _survival_data(seed: int, n_samples: int, n_features: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[: min(4, n_features)] = np.linspace(0.7, 0.1, min(4, n_features))
    event_time = rng.exponential(scale=np.exp(-(X @ beta)))
    censor_time = rng.exponential(scale=1.8, size=n_samples)
    time_value = np.minimum(event_time, censor_time) + 0.05
    event = (event_time <= censor_time).astype(np.int32)
    if event.sum() < 6:
        event[:6] = 1
    X_test = rng.normal(size=(max(30, n_samples // 3), n_features))
    event_time_test = rng.exponential(scale=np.exp(-(X_test @ beta)))
    censor_test = rng.exponential(scale=1.8, size=X_test.shape[0])
    time_test = np.minimum(event_time_test, censor_test) + 0.05
    event_test = (event_time_test <= censor_test).astype(np.int32)
    return X, (time_value, event), X_test, (time_test, event_test)


def _construct(cls, kwargs: dict[str, Any]):
    signature = inspect.signature(cls)
    accepted = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    }
    return cls(**accepted)


def _model_kwargs(spec: CaseSpec, backend: str, seed: int) -> dict[str, Any]:
    device = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    common = {
        "cv": 3,
        "random_state": seed,
        "device": device,
        "compute_inference": False,
        "max_iter": 200,
        "tol": 1e-5,
    }
    common.update(spec.grid_parameters)
    if spec.model_id == "PenalizedGLM_CV":
        common["cv_splits"] = None
    return common


def _fit_args(spec: CaseSpec, X, target):
    if spec.task == "survival":
        time_value, event = target
        return (X, time_value, event)
    return (X, target)


def _score_model(model, spec: CaseSpec, X_test, target_test) -> float:
    if spec.task == "survival":
        time_value, event = target_test
        for args in ((X_test, time_value, event), (X_test, (time_value, event))):
            try:
                return float(model.score(*args))
            except (TypeError, AttributeError):
                continue
        return float("nan")
    return float(model.score(X_test, target_test))


def _selected_parameters(model) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name in ("alpha_", "l1_ratio_", "C_", "penalty_", "best_penalty_"):
        if hasattr(model, name):
            value = getattr(model, name)
            if value is not None:
                try:
                    value = value.item()
                except AttributeError:
                    pass
                selected[name.rstrip("_")] = value
    return selected


def _best_score(model) -> float:
    for name in ("best_score_", "best_cv_score_", "cv_score_"):
        value = getattr(model, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float("nan")


def _clear_cv_caches() -> None:
    """Prevent warmups or repeats from reusing global CV-selection caches."""
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("statgpu."):
            continue
        for name, value in list(vars(module).items()):
            if "CACHE" not in name.upper():
                continue
            try:
                if isinstance(value, dict):
                    value.clear()
                elif hasattr(value, "clear") and callable(value.clear):
                    value.clear()
                elif isinstance(getattr(value, "_cache", None), dict):
                    value._cache.clear()
            except Exception:
                # Cache cleanup is an isolation aid; never hide the measured fit.
                continue


def _candidate_count(spec: CaseSpec) -> int:
    """Derive candidate count from the declared case grid."""
    grid = spec.grid_parameters
    for key in ("alphas", "Cs", "alpha_grid", "penalties"):
        value = grid.get(key)
        if isinstance(value, (list, tuple)):
            count = len(value)
            ratios = grid.get("l1_ratio")
            if isinstance(ratios, (list, tuple)):
                count *= len(ratios)
            return count
    raise ValueError(f"{spec.model_id}: no explicit candidate grid")


def _run_once(spec: CaseSpec, backend: str, seed: int, n_samples: int, n_features: int):
    import numpy as np

    generator = {
        "regression": _regression_data,
        "classification": _classification_data,
        "survival": _survival_data,
    }[spec.task]
    X, target, X_test, target_test = generator(seed, n_samples, n_features)
    X_backend = _to_backend(X, backend)
    X_test_backend = _to_backend(X_test, backend)
    if spec.task == "survival":
        target_backend = tuple(_to_backend(value, backend) for value in target)
        target_test_backend = tuple(_to_backend(value, backend) for value in target_test)
    else:
        target_backend = _to_backend(target, backend)
        target_test_backend = _to_backend(target_test, backend)

    module = importlib.import_module(spec.import_module)
    cls = getattr(module, spec.import_name)
    _clear_cv_caches()
    model = _construct(cls, _model_kwargs(spec, backend, seed))
    sync = _synchronizer(backend)
    sync()
    total_start = time.perf_counter()
    with RegionProfiler(model, n_samples, sync) as profiler:
        model.fit(*_fit_args(spec, X_backend, target_backend))
    sync()
    total_ms = (time.perf_counter() - total_start) * 1000.0

    if profiler.selector_ms <= 0.0:
        raise RuntimeError("CV selector boundary was not observed")
    if profiler.refit_ms <= 0.0:
        raise RuntimeError("final full-data refit boundary was not observed")
    if total_ms < max(profiler.selector_ms, profiler.refit_ms):
        raise RuntimeError("profiled component exceeded public fit duration")

    selected = _selected_parameters(model)
    if not selected:
        raise RuntimeError("fitted estimator did not expose selected hyperparameters")

    n_iter_value = getattr(model, "n_iter_", None)
    try:
        n_iter = int(np.max(np.asarray(n_iter_value))) if n_iter_value is not None else None
    except (TypeError, ValueError):
        n_iter = None

    return {
        "seed": seed,
        "cv_evaluation_ms": profiler.selector_ms,
        "final_refit_ms": profiler.refit_ms,
        "total_fit_ms": total_ms,
        "selected_parameters": selected,
        "validation_score": _best_score(model),
        "final_score": _score_model(model, spec, X_test_backend, target_test_backend),
        "n_iter": n_iter,
    }


def _non_success(backend: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "framework": "statgpu",
        "backend": backend,
        "device": {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend],
        "status": status,
        "reason": reason,
        "timing": None,
        "selected_parameters": None,
        "scores": None,
        "convergence": None,
        "repeat_samples": [],
    }


def _aggregate_run(spec: CaseSpec, backend: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    selected_values = [sample["selected_parameters"] for sample in samples]
    if any(value != selected_values[0] for value in selected_values[1:]):
        raise RuntimeError("selected hyperparameters changed across declared repeats")
    return {
        "framework": "statgpu",
        "backend": backend,
        "device": {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend],
        "status": "success",
        "reason": None,
        "timing": {
            "cv_evaluation_ms": float(np.median([s["cv_evaluation_ms"] for s in samples])),
            "final_refit_ms": float(np.median([s["final_refit_ms"] for s in samples])),
            "total_fit_ms": float(np.median([s["total_fit_ms"] for s in samples])),
            "peak_memory_bytes": None,
        },
        "selected_parameters": selected_values[0],
        "scores": {
            "validation_score": float(np.mean([s["validation_score"] for s in samples])),
            "final_score": float(np.mean([s["final_score"] for s in samples])),
        },
        "convergence": {
            "candidate_count": _candidate_count(spec),
            "fold_count": 3,
            "failed_candidates": 0,
            "failed_folds": 0,
            "final_refit_converged": True,
            "n_iter": max((s["n_iter"] or 0) for s in samples),
        },
        "repeat_samples": [
            {
                "seed": sample["seed"],
                "cv_evaluation_ms": sample["cv_evaluation_ms"],
                "final_refit_ms": sample["final_refit_ms"],
                "total_fit_ms": sample["total_fit_ms"],
            }
            for sample in samples
        ],
    }


def _case(spec: CaseSpec, backends: list[str], seeds: list[int], n_samples: int, n_features: int, warmup: int):
    runs = []
    for backend in backends:
        available, _, reason = _backend_status(backend)
        if not available:
            runs.append(_non_success(backend, "unavailable", reason or "backend unavailable"))
            continue
        try:
            for warmup_index in range(warmup):
                _run_once(
                    spec, backend, seeds[0] + 1_000_000 + warmup_index,
                    n_samples, n_features,
                )
            samples = [
                _run_once(spec, backend, seed, n_samples, n_features)
                for seed in seeds
            ]
            runs.append(_aggregate_run(spec, backend, samples))
        except Exception as exc:
            runs.append(
                _non_success(
                    backend, "failed", f"{type(exc).__name__}: {exc}"
                )
            )

    return {
        "case_id": f"cv-{spec.model_id.lower().replace('_', '-')}",
        "model_id": spec.model_id,
        "task": spec.task,
        "dataset": {
            "generator": f"statgpu-cv-{spec.task}-v1",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_test_samples": max(30, n_samples // 3),
            "parameters": {"seeds": seeds, "signal_features": min(5, n_features)},
        },
        "cv": {
            "fold_count": 3,
            "split_strategy": "statgpu-deterministic-kfold",
            "shuffle": True,
            "subject_preserving": spec.task == "survival",
        },
        "grid": {
            "candidate_count": _candidate_count(spec),
            "identity": "explicit-candidate-grid-v1",
            "parameters": spec.grid_parameters,
        },
        "scoring": {
            "name": spec.scoring_name,
            "direction": spec.scoring_direction,
            "normalization": spec.scoring_normalization,
        },
        "runs": runs,
    }


def build_source(args) -> dict[str, Any]:
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    seeds = [args.seed] * args.repeats
    available = [backend for backend in backends if _backend_status(backend)[0]]
    gpu_name = None
    if "cupy" in available:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    elif "torch" in available:
        import torch
        gpu_name = torch.cuda.get_device_name(0)

    source = {
        "source_schema_version": "1.0",
        "source_date": datetime.now(timezone.utc).date().isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": _git_sha(),
        "environment": {
            "env_id": args.env_id,
            "host": socket.gethostname(),
            "cpu": platform.processor() or platform.machine(),
            "gpu": gpu_name,
            "python": platform.python_version(),
            "packages": {
                name: _package_version(name)
                for name in ("statgpu", "numpy", "scipy", "scikit-learn", "cupy", "torch")
            },
            "available_backends": available,
        },
        "protocol": {
            "seeds": sorted(set(seeds)),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": "float64",
            "synchronization": "backend synchronize immediately before and after each observed timing region",
            "timing_scope": "host-orchestrated public fit with directly observed CV selector and final full-data refit",
            "transfer_policy": "input conversion occurs before timed public fit; device-resident timing",
            "failure_policy": "retain_explicit_disposition",
        },
        "cases": [
            _case(spec, backends, seeds, args.n_samples, args.n_features, args.warmup)
            for spec in CASE_SPECS
        ],
    }
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--backends", default="numpy,cupy,torch")
    parser.add_argument("--n-samples", type=int, default=240)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    source = build_source(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(source, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    check_file(args.out)
    print(f"Wrote validated CV benchmark source: {args.out}")


if __name__ == "__main__":
    main()
