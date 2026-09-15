#!/usr/bin/env python3
"""Issue #160 diagnostic runner for float32 GLM L-BFGS parity.

This runner is diagnostic evidence, not a production solver variant. It mirrors
``lbfgs_solver`` using the same private solver primitives while recording the
iteration-level state needed to explain backend divergence. Each traced solve
is checked against the production solver on the same backend before its trace
is accepted.

The historical float32 case reproduces the schema-v6 weight construction
exactly: finite float32 weights are multiplied by ``3e38`` so their raw float32
sum overflows, then divided by the same float32 scale to obtain the ordinary-
scale weights used by the failing fit. The runner also bridges the low-level
trace back to the public ordinary ``GeneralizedLinearModel`` route whenever
that backend is publicly executable in the current environment.

The runner works on NumPy and Torch CPU in hosted environments. When CuPy and
Torch CUDA are available it additionally records both CUDA backends on their
concrete device. Absence of CUDA is reported explicitly and is not treated as
physical evidence. Use ``--require-cuda`` for a physical acceptance run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from dev.benchmarks import validate_pr151_final_gpu_v5 as v5
from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._array_ops import (
    _copy_arr,
    _device_gt,
    _device_leq,
    _dot_dev,
    _norm2_dev,
    _sync_scalars,
)
from statgpu.glm_core._logistic import LogisticLoss
from statgpu.linear_model import GeneralizedLinearModel
from statgpu.solvers import lbfgs_solver
from statgpu.solvers._lbfgs import (
    _call_loss_with_weight,
    _domain_step_or_raise,
    _prepare_lbfgs_sample_weight,
)
from statgpu.solvers._smooth_domain import (
    _domain_feasible,
    _floating_eps,
    _initial_smooth_params,
)
from statgpu.solvers._utils import (
    _smooth_penalty_gradient,
    _smooth_penalty_value_dev,
)


SEEDS = (151025, 16001, 16002, 16003)
N_SAMPLES = 96
N_FEATURES = 3
MAX_ITER = 1000
TOL = 1.0e-8
HISTORY_SIZE = 10
FLOAT32_OVERFLOW_SCALE = 3.0e38


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_identity():
    try:
        return {
            "sha": _git("rev-parse", "HEAD"),
            "clean": not bool(_git("status", "--porcelain")),
        }
    except Exception:
        return {"sha": None, "clean": None}


def _environment(cuda):
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cuda_available": dict(cuda),
    }
    try:
        import torch

        env["torch"] = str(torch.__version__)
        env["torch_cuda_build"] = str(torch.version.cuda)
        if cuda.get("torch"):
            env["torch_cuda_device"] = {
                "ordinal": 0,
                "name": str(torch.cuda.get_device_name(0)),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
    except Exception as exc:
        env["torch"] = None
        env["torch_probe_error"] = type(exc).__name__

    try:
        import cupy as cp

        env["cupy"] = str(cp.__version__)
        env["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        env["cuda_driver_version"] = int(cp.cuda.runtime.driverGetVersion())
        if cuda.get("cupy"):
            props = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = props.get("name", "")
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode(errors="replace")
            env["cupy_cuda_device"] = {
                "ordinal": 0,
                "name": str(raw_name),
                "major": int(props.get("major", -1)),
                "minor": int(props.get("minor", -1)),
                "total_global_mem": int(props.get("totalGlobalMem", 0)),
            }
    except Exception as exc:
        env["cupy"] = None
        env["cupy_probe_error"] = type(exc).__name__
    return env


def _to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _trace_armijo(
    loss,
    penalty,
    X,
    y,
    params,
    direction,
    old_val_dev,
    gdd,
    domain_cap,
    *,
    sample_weight,
    objective_roundoff,
    direction_norm,
    parameter_tol,
):
    step = min(1.0, domain_cap) if domain_cap is not None else 1.0
    evaluated_domain_trial = False
    rejected_by_domain = False
    for backtrack in range(25):
        candidate = params + step * direction
        if not _domain_feasible(
            loss, X, candidate, sample_weight=sample_weight
        ):
            rejected_by_domain = True
            step *= 0.5
            continue
        evaluated_domain_trial = True
        cand_val_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X,
            y,
            candidate,
            sample_weight=sample_weight,
        )
        cand_val_dev = cand_val_dev + _smooth_penalty_value_dev(
            penalty, candidate
        )
        armijo_rhs = old_val_dev + 1.0e-4 * step * gdd
        if _device_leq(cand_val_dev, armijo_rhs):
            return (
                candidate,
                True,
                step,
                evaluated_domain_trial,
                rejected_by_domain,
                "armijo",
                backtrack,
                _to_float(cand_val_dev),
            )

        required_decrease = max(0.0, -1.0e-4 * step * gdd)
        parameter_displacement = step * direction_norm
        if (
            required_decrease <= objective_roundoff
            and parameter_displacement <= parameter_tol
            and _device_leq(cand_val_dev, old_val_dev + objective_roundoff)
        ):
            return (
                candidate,
                True,
                step,
                evaluated_domain_trial,
                rejected_by_domain,
                "roundoff",
                backtrack,
                _to_float(cand_val_dev),
            )
        step *= 0.5
    return (
        params,
        False,
        step,
        evaluated_domain_trial,
        rejected_by_domain,
        "failed",
        25,
        None,
    )


def traced_lbfgs(
    loss,
    penalty,
    X,
    y,
    *,
    sample_weight=None,
    max_iter=MAX_ITER,
    tol=TOL,
    history_size=HISTORY_SIZE,
):
    """Mirror production L-BFGS and return ``(params, n_iter, trace)``."""
    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    sample_weight = _prepare_lbfgs_sample_weight(
        sample_weight, X_proc.shape[0], backend, X_proc, loss
    )
    params = _initial_smooth_params(
        loss,
        X_proc,
        y_proc,
        backend=backend,
        n_features=X_proc.shape[1],
        init_coef=None,
        sample_weight=sample_weight,
    )

    s_hist = []
    y_hist = []
    rho_hist = []
    _, grad = _call_loss_with_weight(
        loss.fused_value_and_gradient,
        X_proc,
        y_proc,
        params,
        sample_weight=sample_weight,
    )
    grad = grad + _smooth_penalty_gradient(penalty, params)

    records = []
    termination = "max_iter"
    iteration = -1

    for iteration in range(max_iter):
        grad_norm_dev = _norm2_dev(grad)
        q = _copy_arr(grad)
        alphas = []
        for s_vec, y_vec, rho in reversed(list(zip(s_hist, y_hist, rho_hist))):
            alpha = rho * _dot_dev(s_vec, q)
            alphas.append(alpha)
            q = q - alpha * y_vec

        if y_hist:
            sy = _dot_dev(s_hist[-1], y_hist[-1])
            yy = _dot_dev(y_hist[-1], y_hist[-1])
            gamma_dev = sy / yy if _device_gt(yy, 1e-30) else 1.0
            gamma = _to_float(gamma_dev)
        else:
            gamma_dev = 1.0
            gamma = 1.0
        r = gamma_dev * q

        for s_vec, y_vec, rho, alpha in zip(
            s_hist, y_hist, rho_hist, reversed(alphas)
        ):
            beta = rho * _dot_dev(y_vec, r)
            r = r + s_vec * (alpha - beta)

        direction = -r
        gdd_dev = _dot_dev(grad, direction)
        gn, gdd = _sync_scalars(grad_norm_dev, gdd_dev, backend=backend)
        if gn < tol:
            termination = "gradient"
            records.append({
                "iteration": iteration,
                "gradient_norm": gn,
                "history_size": len(s_hist),
                "gamma": gamma,
                "termination": termination,
            })
            break

        fallback_direction = False
        if gdd >= 0 or not np.isfinite(gdd):
            direction = -grad
            gdd = -gn * gn
            fallback_direction = True

        direction_norm_dev = _norm2_dev(direction)
        (direction_norm,) = _sync_scalars(direction_norm_dev, backend=backend)
        domain_cap = _domain_step_or_raise(
            loss,
            X_proc,
            params,
            direction,
            tol,
            sample_weight,
            backend,
        )

        old_val_dev, _ = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params,
            sample_weight=sample_weight,
        )
        old_val_dev = old_val_dev + _smooth_penalty_value_dev(
            penalty, params
        )
        (old_val,) = _sync_scalars(old_val_dev, backend=backend)
        objective_roundoff = (
            64.0 * _floating_eps(old_val_dev) * max(1.0, abs(old_val))
        )

        (
            params_new,
            accepted,
            step,
            evaluated_domain_trial,
            rejected_by_domain,
            acceptance_mode,
            backtracks,
            candidate_value,
        ) = _trace_armijo(
            loss,
            penalty,
            X_proc,
            y_proc,
            params,
            direction,
            old_val_dev,
            gdd,
            domain_cap,
            sample_weight=sample_weight,
            objective_roundoff=objective_roundoff,
            direction_norm=direction_norm,
            parameter_tol=tol,
        )

        if not accepted and domain_cap is not None:
            direction = -grad
            gdd = -gn * gn
            direction_norm = gn
            fallback_direction = True
            domain_cap = _domain_step_or_raise(
                loss,
                X_proc,
                params,
                direction,
                tol,
                sample_weight,
                backend,
            )
            (
                params_new,
                accepted,
                step,
                evaluated_domain_trial,
                rejected_by_domain,
                acceptance_mode,
                backtracks,
                candidate_value,
            ) = _trace_armijo(
                loss,
                penalty,
                X_proc,
                y_proc,
                params,
                direction,
                old_val_dev,
                gdd,
                domain_cap,
                sample_weight=sample_weight,
                objective_roundoff=objective_roundoff,
                direction_norm=direction_norm,
                parameter_tol=tol,
            )

        record = {
            "iteration": iteration,
            "objective": old_val,
            "gradient_norm": gn,
            "directional_derivative": gdd,
            "direction_norm": direction_norm,
            "history_size_before": len(s_hist),
            "gamma": gamma,
            "fallback_direction": fallback_direction,
            "step": step,
            "backtracks": backtracks,
            "acceptance_mode": acceptance_mode,
            "objective_roundoff": objective_roundoff,
            "candidate_objective": candidate_value,
        }

        if not accepted:
            termination = "line_search_failed"
            record["termination"] = termination
            records.append(record)
            break

        _, grad_new = _call_loss_with_weight(
            loss.fused_value_and_gradient,
            X_proc,
            y_proc,
            params_new,
            sample_weight=sample_weight,
        )
        grad_new = grad_new + _smooth_penalty_gradient(penalty, params_new)

        s_vec = params_new - params
        y_vec = grad_new - grad
        ys_dev = _dot_dev(y_vec, s_vec)
        s_norm_dev = _norm2_dev(s_vec)
        ys, s_norm = _sync_scalars(ys_dev, s_norm_dev, backend=backend)
        pair_accepted = ys > 1.0e-12
        if pair_accepted:
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho_hist.append(1.0 / ys)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        record.update({
            "curvature_ys": ys,
            "step_norm": s_norm,
            "curvature_pair_accepted": pair_accepted,
            "history_size_after": len(s_hist),
        })
        records.append(record)

        params = params_new
        grad = grad_new
        if s_norm < tol and domain_cap is None:
            termination = "parameter_step"
            records[-1]["termination"] = termination
            break
        if s_norm < tol and domain_cap is not None:
            grad_new_norm_dev = _norm2_dev(grad_new)
            (grad_new_norm,) = _sync_scalars(
                grad_new_norm_dev, backend=backend
            )
            if grad_new_norm < tol:
                termination = "parameter_and_gradient"
                records[-1]["termination"] = termination
                break
    else:
        termination = "max_iter"

    return params, iteration + 1, {
        "backend": backend,
        "dtype": str(getattr(X_proc, "dtype", None)),
        "termination": termination,
        "iterations": records,
    }


def _data(seed, dtype):
    X, y = v5._logistic_data(seed=seed, n=N_SAMPLES, p=N_FEATURES)
    X = X.astype(dtype)
    y = y.astype(dtype)
    raw = np.linspace(0.75, 1.05, N_SAMPLES, dtype=dtype)
    if np.dtype(dtype) == np.dtype(np.float32):
        scale = np.float32(FLOAT32_OVERFLOW_SCALE)
        overflow_weights = raw * scale
        if not np.all(np.isfinite(overflow_weights)):
            raise AssertionError("schema-v6 float32 fixture has non-finite entries")
        with np.errstate(over="ignore"):
            raw_sum = np.sum(overflow_weights, dtype=np.float32)
        if np.isfinite(raw_sum):
            raise AssertionError("schema-v6 float32 fixture raw sum did not overflow")
        weights = overflow_weights / scale
    else:
        weights = raw
    return X, y, weights


def _augment_intercept(X, backend):
    if backend == "numpy":
        return np.column_stack([X, np.ones(X.shape[0], dtype=X.dtype)])
    if backend == "torch":
        import torch
        return torch.column_stack([
            X,
            torch.ones(X.shape[0], dtype=X.dtype, device=X.device),
        ])
    import cupy as cp
    return cp.column_stack([X, cp.ones(X.shape[0], dtype=X.dtype)])


def _backend_arrays(name, X, y, w, *, use_cuda):
    if name == "numpy":
        return X, y, w, nullcontext()
    if name == "torch":
        import torch
        if use_cuda:
            device = torch.device("cuda:0")
            ctx = torch.cuda.device(device)
        else:
            device = torch.device("cpu")
            ctx = nullcontext()
        dtype = torch.float32 if X.dtype == np.float32 else torch.float64
        return (
            torch.as_tensor(X, dtype=dtype, device=device),
            torch.as_tensor(y, dtype=dtype, device=device),
            torch.as_tensor(w, dtype=dtype, device=device),
            ctx,
        )
    import cupy as cp
    if not use_cuda:
        raise RuntimeError("CuPy diagnostics require a CUDA device")
    return cp.asarray(X), cp.asarray(y), cp.asarray(w), cp.cuda.Device(0)


def _final_metrics(loss, X, y, params, weights):
    val, grad = loss.fused_value_and_gradient(
        X, y, params, sample_weight=weights
    )
    grad_np = np.asarray(_to_numpy(grad), dtype=np.float64)
    return {
        "objective": _to_float(val),
        "gradient_norm": float(np.linalg.norm(grad_np)),
        "params": np.asarray(_to_numpy(params), dtype=np.float64).tolist(),
    }


def _public_estimator_bridge(backend_name, X, y, w, production_params, *, use_cuda):
    if backend_name == "torch" and not use_cuda:
        return {
            "available": False,
            "reason": "public device='torch' is a CUDA route; Torch CPU is trace-only",
        }

    if backend_name == "numpy":
        device = "cpu"
    elif backend_name == "cupy":
        device = "cuda"
    else:
        device = "torch"

    with v5.v4.v3._solver_warning_gate():
        model = GeneralizedLinearModel(
            family="binomial",
            fit_intercept=True,
            solver="lbfgs",
            device=device,
            max_iter=MAX_ITER,
            tol=TOL,
            compute_inference=False,
        ).fit(X, y, sample_weight=w)

    estimator_params = np.concatenate([
        np.asarray(model.coef_, dtype=np.float64),
        np.asarray([model.intercept_], dtype=np.float64),
    ])
    production_np = np.asarray(_to_numpy(production_params), dtype=np.float64)
    error = float(np.max(np.abs(estimator_params - production_np)))
    if error > 1.0e-12:
        raise AssertionError(
            f"ordinary estimator drifted from traced low-level route on {backend_name}: "
            f"error={error:.3e}"
        )
    return {
        "available": True,
        "params_max_abs": error,
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
        "n_iter": int(model.n_iter_),
    }


def _one_case(backend_name, X_np, y_np, w_np, *, use_cuda):
    X, y, w, ctx = _backend_arrays(
        backend_name, X_np, y_np, w_np, use_cuda=use_cuda
    )
    with ctx:
        X_work = _augment_intercept(X, backend_name)
        loss = LogisticLoss()
        traced, traced_iter, trace = traced_lbfgs(
            loss, None, X_work, y, sample_weight=w
        )
        production, production_iter = lbfgs_solver(
            loss,
            None,
            X_work,
            y,
            max_iter=MAX_ITER,
            tol=TOL,
            history_size=HISTORY_SIZE,
            sample_weight=w,
        )
        traced_np = np.asarray(_to_numpy(traced), dtype=np.float64)
        production_np = np.asarray(_to_numpy(production), dtype=np.float64)
        reproduction_error = float(np.max(np.abs(traced_np - production_np)))
        if production_iter != traced_iter or reproduction_error > 1.0e-12:
            raise AssertionError(
                f"trace runner drifted from production on {backend_name}: "
                f"n_iter={traced_iter}/{production_iter}, error={reproduction_error:.3e}"
            )
        metrics = _final_metrics(loss, X_work, y, production, w)
        estimator_bridge = _public_estimator_bridge(
            backend_name, X, y, w, production, use_cuda=use_cuda
        )
    return {
        "production_n_iter": int(production_iter),
        "trace_matches_production_max_abs": reproduction_error,
        "ordinary_estimator_bridge": estimator_bridge,
        "final": metrics,
        "trace": trace,
    }


def _available_backends():
    result = [("numpy", False), ("torch", False)]
    cuda = {"cupy": False, "torch": False}
    try:
        import torch
        cuda["torch"] = bool(torch.cuda.is_available())
        if cuda["torch"]:
            result.append(("torch_cuda", True))
    except Exception:
        pass
    try:
        import cupy as cp
        cuda["cupy"] = bool(cp.cuda.runtime.getDeviceCount() > 0)
        if cuda["cupy"]:
            result.append(("cupy", True))
    except Exception:
        pass
    return result, cuda


def run(output: Path, *, require_cuda: bool = False):
    source = _source_identity()
    if source.get("clean") is not True:
        raise RuntimeError(
            "Issue #160 diagnostic evidence requires an exact clean source tree"
        )

    backend_specs, cuda = _available_backends()
    if require_cuda and not (cuda.get("cupy") and cuda.get("torch")):
        raise RuntimeError(
            "--require-cuda needs both CuPy CUDA and Torch CUDA on the physical device"
        )

    payload = {
        "schema": 2,
        "source": source,
        "environment": _environment(cuda),
        "fixture": {
            "historical_seed": 151025,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "float32_weight_construction": (
                "base_weights=(linspace(float32)*float32(3e38))/float32(3e38)"
            ),
            "float32_raw_sum_overflow_required": True,
        },
        "solver": {
            "tol": TOL,
            "max_iter": MAX_ITER,
            "history_size": HISTORY_SIZE,
        },
        "physical_cuda_complete": bool(cuda.get("cupy") and cuda.get("torch")),
        "cases": {},
    }

    for dtype in (np.float32, np.float64):
        dtype_name = np.dtype(dtype).name
        payload["cases"][dtype_name] = {}
        for seed in SEEDS:
            X, y, w = _data(seed, dtype)
            seed_result = {}
            for backend_label, use_cuda in backend_specs:
                backend_name = "torch" if backend_label == "torch_cuda" else backend_label
                seed_result[backend_label] = _one_case(
                    backend_name, X, y, w, use_cuda=use_cuda
                )
            ref = seed_result["numpy"]["final"]
            for label, result in seed_result.items():
                p_ref = np.asarray(ref["params"], dtype=np.float64)
                p_other = np.asarray(result["final"]["params"], dtype=np.float64)
                result["errors_vs_numpy"] = {
                    "params_max_abs": float(np.max(np.abs(p_other - p_ref))),
                    "objective_abs": abs(
                        float(result["final"]["objective"])
                        - float(ref["objective"])
                    ),
                    "gradient_norm_abs": abs(
                        float(result["final"]["gradient_norm"])
                        - float(ref["gradient_norm"])
                    ),
                }
            payload["cases"][dtype_name][str(seed)] = seed_result

    for seed in SEEDS:
        seed_key = str(seed)
        for backend_label in payload["cases"]["float32"][seed_key]:
            f32 = payload["cases"]["float32"][seed_key][backend_label]
            f64 = payload["cases"]["float64"][seed_key][backend_label]
            p32 = np.asarray(f32["final"]["params"], dtype=np.float64)
            p64 = np.asarray(f64["final"]["params"], dtype=np.float64)
            f32["errors_vs_same_backend_float64"] = {
                "params_max_abs": float(np.max(np.abs(p32 - p64))),
                "objective_abs": abs(
                    float(f32["final"]["objective"])
                    - float(f64["final"]["objective"])
                ),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "diagnostic_complete",
        "source": payload["source"],
        "cuda_available": cuda,
        "physical_cuda_complete": payload["physical_cuda_complete"],
        "output": str(output),
    }, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/issue160_float32_lbfgs_diagnostics.json"),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless both CuPy CUDA and Torch CUDA are available",
    )
    args = parser.parse_args()
    run(args.output, require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
