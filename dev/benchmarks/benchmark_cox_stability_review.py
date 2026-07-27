"""Audit ordinary-Cox stability and warm three-backend fit performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from statgpu.survival import CoxPH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _device_module(device: str):
    if device == "cuda":
        import cupy as xp

        return xp
    if device == "torch":
        import torch as xp

        return xp
    return np


def _to_device(device: str, value):
    xp = _device_module(device)
    if device == "cuda":
        return xp.asarray(value)
    if device == "torch":
        return xp.as_tensor(value, dtype=xp.float64, device="cuda")
    return np.asarray(value)


def _synchronize(device: str) -> None:
    if device == "cuda":
        _device_module(device).cuda.Stream.null.synchronize()
    elif device == "torch":
        _device_module(device).cuda.synchronize()


def _to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_data(n: int, p: int, seed: int, *, heavy_ties: bool):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.25, -0.15, p)
    failure = rng.exponential(np.exp(np.clip(-(X @ beta), -5.0, 5.0)))
    censor = rng.exponential(1.8, size=n)
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[0] = 1.0
    if heavy_ties:
        stop = np.maximum(np.ceil(stop * 16.0) / 16.0, 1.0 / 16.0)
    return X.astype(np.float64), stop.astype(np.float64), event


def _fit_once(device: str, ties: str, X, stop, event):
    model = CoxPH(
        device=device,
        ties=ties,
        compute_inference=False,
        compute_cindex=False,
        max_iter=100,
        tol=1e-8,
    )
    _synchronize(device)
    started = time.perf_counter()
    model.fit(X, stop, event)
    _synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "coef": model.coef_.tolist(),
        "log_likelihood": float(model.log_likelihood),
        "iterations": int(model.n_iter_),
        "converged": bool(model.converged_),
        "finite": bool(
            np.all(np.isfinite(model.coef_))
            and np.isfinite(model.log_likelihood)
        ),
    }


def _timed_case(device, ties, X_np, stop_np, event_np, warmups, repeats):
    X = _to_device(device, X_np)
    stop = _to_device(device, stop_np)
    event = _to_device(device, event_np)
    for _ in range(warmups):
        _fit_once(device, ties, X, stop, event)
    runs = [
        _fit_once(device, ties, X, stop, event) for _ in range(repeats)
    ]
    seconds = np.asarray([run["seconds"] for run in runs])
    representative = runs[int(np.argsort(seconds)[len(seconds) // 2])]
    return {
        "median_seconds": float(np.median(seconds)),
        "runs": runs,
        "coef": representative["coef"],
        "log_likelihood": representative["log_likelihood"],
        "all_converged": all(run["converged"] for run in runs),
        "all_finite": all(run["finite"] for run in runs),
    }


def _extreme_case(device: str, ties: str):
    X = _to_device(device, np.array([[-1000.0], [0.0], [1000.0]]))
    stop = _to_device(device, np.array([1.0, 2.0, 3.0]))
    event = _to_device(device, np.array([1.0, 1.0, 0.0]))
    init = _to_device(device, np.array([1.0]))
    model = CoxPH(
        device=device,
        ties=ties,
        compute_inference=False,
        compute_cindex=False,
        max_iter=40,
    ).fit(X, stop, event, init_coef=init)
    return {
        "coef": model.coef_.tolist(),
        "log_likelihood": float(model.log_likelihood),
        "iterations": int(model.n_iter_),
        "converged": bool(model.converged_),
        "finite": bool(
            np.all(np.isfinite(model.coef_))
            and np.isfinite(model.log_likelihood)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--p", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    devices = ["cpu", "cuda", "torch"]
    cases = {}
    for heavy_ties in (False, True):
        scenario = "heavy_ties" if heavy_ties else "continuous"
        X, stop, event = _make_data(
            args.n, args.p, 8101 + int(heavy_ties), heavy_ties=heavy_ties
        )
        for ties in ("breslow", "efron"):
            for device in devices:
                cases[f"{scenario}:{ties}:{device}"] = _timed_case(
                    device,
                    ties,
                    X,
                    stop,
                    event,
                    args.warmups,
                    args.repeats,
                )

    extreme = {
        f"{ties}:{device}": _extreme_case(device, ties)
        for ties in ("breslow", "efron")
        for device in devices
    }
    cp = _device_module("cuda")
    torch = _device_module("torch")
    artifact = {
        "schema_version": 1,
        "validation_tier": "remote-full",
        "timing_contract": {
            "kind": "warm synchronized fit timing",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "input_conversion_included": False,
            "fresh_process_cold_start_measured": False,
        },
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "clean": _git("status", "--porcelain") == "",
            "hashes": {
                path: _sha256(root / path)
                for path in (
                    "statgpu/survival/_cox.py",
                    "statgpu/survival/_cox_counting.py",
                    "statgpu/losses/_cox_ph.py",
                    "dev/benchmarks/benchmark_cox_stability_review.py",
                )
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
        "shape": {"n": args.n, "p": args.p},
        "cases": cases,
        "extreme_predictor_cases": extreme,
        "gate_failures": [
            key
            for key, value in {**cases, **extreme}.items()
            if not value.get("all_finite", value.get("finite", False))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "gate_failures": artifact["gate_failures"],
        "source": artifact["source"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
