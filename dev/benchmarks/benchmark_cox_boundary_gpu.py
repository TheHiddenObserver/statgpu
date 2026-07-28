"""Physical-GPU audit for the final PR80 Cox public-boundary fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import Device  # noqa: E402
from statgpu.survival import CoxPH, CoxPHCV  # noqa: E402
from statgpu.survival import _risk_sets as risk_sets  # noqa: E402
from statgpu.survival._risk_sets import (  # noqa: E402
    cox_counting_process_objective,
)


SOURCE_FILES = (
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_fit_adapter.py",
    "statgpu/survival/_risk_sets.py",
    "dev/benchmarks/benchmark_cox_boundary_gpu.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _sample(seed: int = 2280, n: int = 72, p: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.35, -0.2, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=1.8, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[:4] = 1.0
    return X, stop, event


def _backend(name: str):
    if name == "cupy":
        import cupy as xp

        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy has no physical CUDA device")
        return xp
    import torch as xp

    if not xp.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    return xp


def _array(name: str, xp, value, *, complex_value: bool = False):
    if name == "cupy":
        dtype = xp.complex128 if complex_value else xp.float64
        return xp.asarray(value, dtype=dtype)
    dtype = xp.complex128 if complex_value else xp.float64
    return xp.as_tensor(value, dtype=dtype, device="cuda")


def _numpy(name: str, value):
    if name == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return value.detach().cpu().numpy()


def _sync(name: str, xp) -> None:
    if name == "cupy":
        xp.cuda.Stream.null.synchronize()
    else:
        xp.cuda.synchronize()


def _case_boundary(name: str, xp) -> dict:
    device = "cuda" if name == "cupy" else "torch"
    expected = Device.CUDA if name == "cupy" else Device.TORCH
    X_np, stop_np, event_np = _sample()
    X = _array(name, xp, X_np)
    target = _array(name, xp, np.column_stack((stop_np, event_np)))
    constructor_rejections = {}
    for parameter in (
        "compute_inference",
        "compute_cindex",
        "gpu_memory_cleanup",
    ):
        try:
            CoxPH(device=device, **{parameter: "False"})
        except ValueError as exc:
            constructor_rejections[parameter] = parameter in str(exc)
        else:
            constructor_rejections[parameter] = False
    model = CoxPH(
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=80,
    )
    model.set_params(device=device)

    def reject_public_host_copy(*_args, **_kwargs):
        raise AssertionError("packed target crossed the public host boundary")

    model._to_numpy = reject_public_host_copy
    started = time.perf_counter()
    model.fit(X, target)
    _sync(name, xp)
    fit_seconds = time.perf_counter() - started

    complex_X = _array(
        name,
        xp,
        X_np[:3].astype(np.complex128) + 1j,
        complex_value=True,
    )
    complex_rejected = False
    try:
        model.predict_survival(complex_X)
    except ValueError as exc:
        complex_rejected = "real-valued" in str(exc)

    device_normalized = model.device is expected
    packed_target_stayed_native = model._entry is None
    finite = bool(np.all(np.isfinite(model.coef_)))

    failed_refit_cleared = False
    try:
        model.fit(complex_X, target)
    except ValueError:
        failed_refit_cleared = (
            not model._fitted
            and model.coef_ is None
            and model._X is None
            and model._time is None
            and model._event is None
        )

    return {
        "backend": name,
        "fit_seconds": fit_seconds,
        "packed_target_stayed_native": packed_target_stayed_native,
        "complex_prediction_rejected": complex_rejected,
        "device_normalized": device_normalized,
        "failed_refit_cleared": failed_refit_cleared,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "finite": finite,
        "passed": all(
            (
                packed_target_stayed_native,
                complex_rejected,
                device_normalized,
                failed_refit_cleared,
                all(constructor_rejections.values()),
                finite,
            )
        ),
    }


def _case_cv(name: str, xp) -> dict:
    device = "cuda" if name == "cupy" else "torch"
    expected = Device.CUDA if name == "cupy" else Device.TORCH
    X_np, stop_np, event_np = _sample(seed=2293, n=36, p=2)
    constructor_rejections = {}
    for parameter in ("compute_inference", "gpu_memory_cleanup"):
        try:
            CoxPHCV(
                penalties=np.array([0.1]),
                cv=2,
                device=device,
                **{parameter: "False"},
            )
        except ValueError as exc:
            constructor_rejections[parameter] = parameter in str(exc)
        else:
            constructor_rejections[parameter] = False
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
    )
    model.set_params(device=device)
    started = time.perf_counter()
    model.fit(
        _array(name, xp, X_np),
        _array(name, xp, stop_np),
        _array(name, xp, event_np),
    )
    _sync(name, xp)
    fit_seconds = time.perf_counter() - started
    passed = (
        model.device is expected
        and model.estimator_ is not None
        and model.estimator_.device is expected
        and model.effective_device_ == device
        and bool(np.all(np.isfinite(model.coef_)))
        and all(constructor_rejections.values())
    )
    return {
        "backend": name,
        "fit_seconds": fit_seconds,
        "effective_device": model.effective_device_,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "finite": bool(np.all(np.isfinite(model.coef_))),
        "passed": bool(passed),
    }


def _case_workspace(name: str, xp) -> dict:
    rng = np.random.default_rng(2294)
    n, p = 8192, 3
    X_np = rng.normal(size=(n, p))
    stop_np = np.full(n, 6.0)
    stop_np[:4] = 5.0
    event_np = np.zeros(n)
    event_np[:4] = 1.0
    start_np = rng.uniform(0.0, 4.0, size=n)
    beta_np = np.array([0.2, -0.15, 0.1])
    reference = cox_counting_process_objective(
        beta_np,
        X_np,
        stop_np,
        event_np,
        start=start_np,
        ties="efron",
        score_residuals=True,
    )
    previous = os.environ.get("STATGPU_COX_GROUP_MAX_BYTES")
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = "4096"
    try:
        started = time.perf_counter()
        result = cox_counting_process_objective(
            _array(name, xp, beta_np),
            _array(name, xp, X_np),
            _array(name, xp, stop_np),
            _array(name, xp, event_np),
            start=_array(name, xp, start_np),
            ties="efron",
            score_residuals=True,
        )
        _sync(name, xp)
        seconds = time.perf_counter() - started
    finally:
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(reference[key])
                    - np.asarray(_numpy(name, result[key]))
                )
            )
        )
        for key in ("score", "information", "score_residuals")
    }
    differences["log_likelihood"] = float(
        abs(
            float(reference["log_likelihood"])
            - float(np.asarray(_numpy(name, result["log_likelihood"])))
        )
    )
    passed = max(differences.values()) <= 1e-9
    return {
        "backend": name,
        "n": n,
        "p": p,
        "workspace_limit_bytes": 4096,
        "seconds": seconds,
        "max_abs_differences": differences,
        "passed": passed,
    }


def _case_wide_workspace_route(name: str, xp) -> dict:
    rng = np.random.default_rng(2304)
    n, p = 4096, 128
    workspace_limit = 8 * 1024 * 1024
    X_np = rng.normal(size=(n, p))
    stop_np = np.full(n, 6.0)
    stop_np[:4] = 5.0
    event_np = np.zeros(n)
    event_np[:4] = 1.0
    start_np = rng.uniform(0.0, 4.0, size=n)
    beta_np = np.linspace(0.08, -0.04, p)
    itemsize = np.dtype(np.float64).itemsize

    # This is the exact pre-fdc5f00 estimate. It omitted the two possible
    # n-by-p weighted-design intermediates used by three-operand einsum.
    old_estimate = n * (2 + 8 * itemsize) + 6 * p * p * itemsize
    corrected_estimate = risk_sets._estimate_dense_group_workspace_bytes(
        n,
        p,
        itemsize,
        compute_derivatives=True,
        score_residuals=True,
    )
    routing_boundary_passed = (
        old_estimate <= workspace_limit < corrected_estimate
    )

    previous = os.environ.get("STATGPU_COX_GROUP_MAX_BYTES")
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = str(1 << 50)
    try:
        reference = cox_counting_process_objective(
            beta_np,
            X_np,
            stop_np,
            event_np,
            start=start_np,
            ties="efron",
            score_residuals=True,
        )
    finally:
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    streaming_calls = []
    original_streamed = risk_sets._streamed_stratum_group_objective

    def recording_streamed(*args, **kwargs):
        streaming_calls.append(
            {
                "n": int(args[1].shape[0]),
                "p": int(args[1].shape[1]),
                "workspace_limit_bytes": int(kwargs["max_workspace_bytes"]),
            }
        )
        return original_streamed(*args, **kwargs)

    risk_sets._streamed_stratum_group_objective = recording_streamed
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = str(workspace_limit)
    try:
        started = time.perf_counter()
        result = cox_counting_process_objective(
            _array(name, xp, beta_np),
            _array(name, xp, X_np),
            _array(name, xp, stop_np),
            _array(name, xp, event_np),
            start=_array(name, xp, start_np),
            ties="efron",
            score_residuals=True,
        )
        _sync(name, xp)
        seconds = time.perf_counter() - started
    finally:
        risk_sets._streamed_stratum_group_objective = original_streamed
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(reference[key])
                    - np.asarray(_numpy(name, result[key]))
                )
            )
        )
        for key in ("score", "information", "score_residuals")
    }
    differences["log_likelihood"] = float(
        abs(
            float(reference["log_likelihood"])
            - float(np.asarray(_numpy(name, result["log_likelihood"])))
        )
    )
    route_was_streamed = streaming_calls == [
        {
            "n": n,
            "p": p,
            "workspace_limit_bytes": workspace_limit,
        }
    ]
    passed = (
        routing_boundary_passed
        and route_was_streamed
        and max(differences.values()) <= 1e-9
    )
    return {
        "backend": name,
        "n": n,
        "p": p,
        "workspace_limit_bytes": workspace_limit,
        "old_estimate_bytes": old_estimate,
        "corrected_estimate_bytes": corrected_estimate,
        "old_estimate_selects_dense": old_estimate <= workspace_limit,
        "corrected_estimate_selects_streaming": (
            corrected_estimate > workspace_limit
        ),
        "observed_streaming_calls": streaming_calls,
        "seconds": seconds,
        "max_abs_differences": differences,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    report = {
        "schema_version": 2,
        "validation_tier": "remote-full",
        "source_commit": head,
        "source_clean": not dirty,
        "source_sha256": {
            path: _sha256(REPO_ROOT / path) for path in SOURCE_FILES
        },
        "python": sys.version,
        "numpy": np.__version__,
        "backends": {},
        "gate_failures": [],
        "command": "python dev/benchmarks/benchmark_cox_boundary_gpu.py --output <path>",
    }
    for name in ("cupy", "torch"):
        try:
            xp = _backend(name)
            device_name = (
                xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if name == "cupy"
                else xp.cuda.get_device_name(0)
            )
            cases = {
                "public_boundary": _case_boundary(name, xp),
                "cv_device_normalization": _case_cv(name, xp),
                "single_group_workspace": _case_workspace(name, xp),
                "wide_workspace_route": _case_wide_workspace_route(name, xp),
            }
            report["backends"][name] = {
                "version": xp.__version__,
                "device": device_name,
                "cases": cases,
            }
            for case_name, case in cases.items():
                if not case["passed"]:
                    report["gate_failures"].append(f"{name}:{case_name}")
        except Exception as exc:
            report["backends"][name] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            report["gate_failures"].append(f"{name}:execution")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
