"""Profile hot spots inside PenalizedGLM_CV targeted benchmarks.

This script is dev-only. It monkeypatches solver entry points to count calls,
iterations, and time spent in solver/refit/scoring buckets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark_cv_full import make_data


ALL_LOSSES = [
    "squared_error",
    "logistic",
    "poisson",
    "gamma",
    "inverse_gaussian",
    "negative_binomial",
    "tweedie",
]
ALL_PENALTIES = ["l2", "l1", "elasticnet", "scad", "mcp"]
ALL_DEVICES = ["cpu", "cuda", "torch", "auto"]


def _csv(value, choices=None):
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if choices is not None:
        unknown = sorted(set(items) - set(choices))
        if unknown:
            raise ValueError(f"unknown values {unknown}; choices={choices}")
    return items


def _parse_sizes(value):
    out = []
    for item in _csv(value):
        n_str, p_str = item.lower().replace(":", "x").split("x", 1)
        out.append((int(n_str), int(p_str)))
    return out


def _device_name(x):
    mod = type(x).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _summarize_iters(values):
    if not values:
        return {"count": 0, "sum": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "sum": int(np.sum(arr)),
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "max": int(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", default="poisson,gamma,inverse_gaussian,negative_binomial,tweedie")
    parser.add_argument("--penalties", default="l1,elasticnet")
    parser.add_argument("--devices", default="cpu,cuda,torch")
    parser.add_argument("--sizes", default="500x20")
    parser.add_argument("--n-alphas", type=int, default=8)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--cv-strategy", choices=["strict", "two_stage"], default="strict")
    parser.add_argument("--acknowledge-approx", action="store_true")
    parser.add_argument("--refine-top-k", type=int, default=3)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    losses = _csv(args.losses, ALL_LOSSES)
    penalties = _csv(args.penalties, ALL_PENALTIES)
    devices = _csv(args.devices, ALL_DEVICES)
    sizes = _parse_sizes(args.sizes)

    import statgpu.glm_core._irls as irls_mod
    import statgpu.glm_core._solver as solver_mod
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    records = []
    active = {"solver": [], "refit_time": 0.0, "scoring_time": 0.0}

    orig_fista = solver_mod.fista_solver
    orig_fista_bb = solver_mod.fista_bb_solver
    orig_irls_fit = irls_mod.IRLSSolver.fit
    orig_refit = cv_mod.PenalizedGLM_CV._refit_best
    orig_eval = cv_mod._evaluate_loss_numpy

    def wrap_fista(loss, penalty, X, y, *a, **kw):
        t0 = time.perf_counter()
        coef, n_iter = orig_fista(loss, penalty, X, y, *a, **kw)
        active["solver"].append({
            "solver": "fista",
            "loss": getattr(loss, "name", ""),
            "penalty": getattr(penalty, "name", ""),
            "backend": _device_name(X),
            "cv_mode": bool(kw.get("cv_mode", False)),
            "n_iter": int(n_iter),
            "time_s": time.perf_counter() - t0,
        })
        return coef, n_iter

    def wrap_fista_bb(loss, penalty, X, y, *a, **kw):
        t0 = time.perf_counter()
        coef, n_iter = orig_fista_bb(loss, penalty, X, y, *a, **kw)
        active["solver"].append({
            "solver": "fista_bb",
            "loss": getattr(loss, "name", ""),
            "penalty": getattr(penalty, "name", ""),
            "backend": _device_name(X),
            "cv_mode": False,
            "n_iter": int(n_iter),
            "time_s": time.perf_counter() - t0,
        })
        return coef, n_iter

    def wrap_irls_fit(self, X, y, *a, **kw):
        t0 = time.perf_counter()
        params, n_iter = orig_irls_fit(self, X, y, *a, **kw)
        active["solver"].append({
            "solver": "irls",
            "loss": getattr(getattr(self, "family", None), "name", ""),
            "penalty": "l2",
            "backend": _device_name(X),
            "cv_mode": False,
            "n_iter": int(n_iter),
            "time_s": time.perf_counter() - t0,
        })
        return params, n_iter

    def wrap_refit(self, X, y, best_alpha):
        t0 = time.perf_counter()
        out = orig_refit(self, X, y, best_alpha)
        active["refit_time"] += time.perf_counter() - t0
        return out

    def wrap_eval(*a, **kw):
        t0 = time.perf_counter()
        out = orig_eval(*a, **kw)
        active["scoring_time"] += time.perf_counter() - t0
        return out

    solver_mod.fista_solver = wrap_fista
    solver_mod.fista_bb_solver = wrap_fista_bb
    irls_mod.IRLSSolver.fit = wrap_irls_fit
    cv_mod.PenalizedGLM_CV._refit_best = wrap_refit
    cv_mod._evaluate_loss_numpy = wrap_eval

    try:
        for n, p in sizes:
            for loss in losses:
                X_np, y_np = make_data(loss, n, p, args.seed)
                X_by_device = {"cpu": X_np, "auto": X_np}
                y_by_device = {"cpu": y_np, "auto": y_np}
                if "cuda" in devices:
                    try:
                        import cupy as cp
                        if cp.cuda.runtime.getDeviceCount() > 0:
                            X_by_device["cuda"] = cp.asarray(X_np)
                            y_by_device["cuda"] = cp.asarray(y_np)
                    except Exception:
                        pass
                if "torch" in devices:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            X_by_device["torch"] = torch.as_tensor(X_np, device="cuda", dtype=torch.float64)
                            y_by_device["torch"] = torch.as_tensor(y_np, device="cuda", dtype=torch.float64)
                    except Exception:
                        pass

                for penalty in penalties:
                    for device in devices:
                        if device not in X_by_device:
                            continue
                        active["solver"] = []
                        active["refit_time"] = 0.0
                        active["scoring_time"] = 0.0
                        model = PenalizedGLM_CV(
                            loss=loss,
                            penalty=penalty,
                            n_alphas=args.n_alphas,
                            cv=args.cv,
                            device=device,
                            max_iter=args.max_iter,
                            tol=args.tol,
                            cv_strategy=args.cv_strategy,
                            acknowledge_approx=args.acknowledge_approx,
                            refine_top_k=args.refine_top_k,
                        )
                        t0 = time.perf_counter()
                        model.fit(X_by_device[device], y_by_device[device])
                        total = time.perf_counter() - t0
                        solver_time = float(sum(x["time_s"] for x in active["solver"]))
                        iters = [x["n_iter"] for x in active["solver"]]
                        rec = {
                            "loss": loss,
                            "penalty": penalty,
                            "device": device,
                            "n": n,
                            "p": p,
                            "alpha": float(model.alpha_),
                            "cv_strategy": getattr(model, "cv_strategy_", args.cv_strategy),
                            "refined_count": int(np.sum(model.cv_results_.get("refined_mask", []))),
                            "total_s": total,
                            "solver_time_s": solver_time,
                            "refit_time_s": active["refit_time"],
                            "scoring_time_s": active["scoring_time"],
                            "other_time_s": total - solver_time - active["refit_time"] - active["scoring_time"],
                            "solver_calls": len(active["solver"]),
                            "iters": _summarize_iters(iters),
                            "solver_records": active["solver"],
                        }
                        records.append(rec)
                        print(
                            f"{loss}+{penalty:<10} {device:<5} "
                            f"total={total:.3f}s solver={solver_time:.3f}s "
                            f"refit={active['refit_time']:.3f}s calls={len(active['solver'])} "
                            f"iter_sum={rec['iters']['sum']}"
                        )
    finally:
        solver_mod.fista_solver = orig_fista
        solver_mod.fista_bb_solver = orig_fista_bb
        irls_mod.IRLSSolver.fit = orig_irls_fit
        cv_mod.PenalizedGLM_CV._refit_best = orig_refit
        cv_mod._evaluate_loss_numpy = orig_eval

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {path}")


if __name__ == "__main__":
    main()
