# -*- coding: utf-8 -*-
"""Configurable CV benchmark for PenalizedGLM_CV.

Default arguments preserve the previous broad matrix. Use --losses,
--penalties, --devices, --solvers, --sizes, --n-alphas, --cv, --repeat and
--warmup for targeted precision/performance gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
ALL_SOLVERS = ["auto", "fista", "fista_bb", "irls", "newton", "lbfgs"]


def _csv(value, choices=None):
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if choices is not None:
        unknown = sorted(set(items) - set(choices))
        if unknown:
            raise ValueError(f"unknown values {unknown}; choices={choices}")
    return items


def _parse_sizes(value):
    sizes = []
    for item in _csv(value):
        if "x" in item:
            n_str, p_str = item.lower().split("x", 1)
        elif ":" in item:
            n_str, p_str = item.split(":", 1)
        else:
            raise ValueError(f"bad size '{item}', expected NxP")
        sizes.append((int(n_str), int(p_str)))
    return sizes


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--losses", default=",".join(ALL_LOSSES))
    p.add_argument("--penalties", default=",".join(ALL_PENALTIES))
    p.add_argument("--devices", default=",".join(ALL_DEVICES))
    p.add_argument("--solvers", default="auto")
    p.add_argument("--sizes", default="500x20")
    p.add_argument("--n-alphas", type=int, default=20)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--cv-strategy", choices=["strict", "two_stage"], default="strict")
    p.add_argument("--acknowledge-approx", action="store_true")
    p.add_argument("--refine-top-k", type=int, default=3)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def _to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


def coef_corr(a, b):
    a, b = _to_numpy(a).ravel(), _to_numpy(b).ravel()
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0 if np.allclose(a, b, atol=1e-10) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def coef_l2(a, b):
    return float(np.linalg.norm(_to_numpy(a).ravel() - _to_numpy(b).ravel()))


def bench(func, warmup=1, repeat=3):
    for _ in range(max(0, warmup)):
        func()
    times = []
    result = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def gen_regression(n, p, rng):
    X = rng.randn(n, p)
    y = X @ np.linspace(2, 0.5, p) + rng.randn(n) * 0.5
    return X, y


def gen_classification(n, p, rng):
    X = rng.randn(n, p)
    y = (X @ np.linspace(2, 0.5, p) + rng.randn(n) > 0).astype(float)
    return X, y


def gen_count(n, p, rng):
    X = rng.randn(n, p)
    mu = np.exp(np.clip(X @ np.linspace(0.5, 0.1, p), -5, 5))
    y = rng.poisson(mu).astype(float)
    return X, y


def gen_positive(n, p, rng):
    X = rng.randn(n, p)
    y = np.abs(X @ np.linspace(2, 0.5, p)) + 0.1
    return X, y


def gen_inverse_gaussian(n, p, rng):
    X = rng.randn(n, p) * 0.3
    beta = np.zeros(p)
    beta[: min(5, p)] = [0.5, -0.3, 0.2, -0.1, 0.05][: min(5, p)]
    mu = np.exp(np.clip(X @ beta, -2, 3)) + 0.1
    nu = rng.normal(size=n) ** 2
    y = mu + mu**2 * nu / 2 - mu / 2 * np.sqrt(4 * mu * nu + mu**2 * nu**2)
    return np.asarray(X), np.clip(y, 1e-6, None)


def gen_negative_binomial(n, p, rng):
    X = rng.randn(n, p) * 0.3
    beta = np.zeros(p)
    beta[: min(5, p)] = [0.5, -0.3, 0.2, -0.1, 0.05][: min(5, p)]
    lam = np.exp(np.clip(X @ beta, -3, 5))
    size_p = 1.0
    prob_nb = size_p / (size_p + lam)
    y = rng.negative_binomial(size_p, prob_nb).astype(float)
    return X, np.maximum(y, 0.0)


def gen_tweedie(n, p, rng):
    X = rng.randn(n, p) * 0.3
    beta = np.zeros(p)
    beta[: min(5, p)] = [0.5, -0.3, 0.2, -0.1, 0.05][: min(5, p)]
    mu = np.exp(np.clip(X @ beta, -2, 4)) + 0.1
    pwr = 1.5
    phi = 1.0
    lam_tw = np.clip(mu ** (2 - pwr) / (phi * (2 - pwr)), 0.01, 100)
    alpha_tw = (2 - pwr) / (pwr - 1)
    beta_tw = phi * (pwr - 1) * mu ** (pwr - 1)
    counts = rng.poisson(lam_tw)
    y = np.array([rng.gamma(max(counts[i], 1) * alpha_tw, beta_tw[i]) for i in range(n)])
    return X, np.clip(y, 1e-6, None)


def make_data(loss, n, p, seed):
    rng = np.random.RandomState(seed)
    if loss == "squared_error":
        return gen_regression(n, p, rng)
    if loss == "logistic":
        return gen_classification(n, p, rng)
    if loss == "poisson":
        return gen_count(n, p, rng)
    if loss == "gamma":
        return gen_positive(n, p, rng)
    if loss == "inverse_gaussian":
        return gen_inverse_gaussian(n, p, rng)
    if loss == "negative_binomial":
        return gen_negative_binomial(n, p, rng)
    if loss == "tweedie":
        return gen_tweedie(n, p, rng)
    raise ValueError(loss)


def main():
    args = _parse_args()
    losses = _csv(args.losses, ALL_LOSSES)
    penalties = _csv(args.penalties, ALL_PENALTIES)
    devices = _csv(args.devices, ALL_DEVICES)
    solvers = _csv(args.solvers, ALL_SOLVERS)
    sizes = _parse_sizes(args.sizes)

    import cupy as cp
    import torch
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    has_cupy = cp.cuda.runtime.getDeviceCount() > 0
    has_torch = torch.cuda.is_available()
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if has_cupy else "unavailable"

    print("=" * 90)
    print("CV BENCHMARK: PenalizedGLM_CV")
    print("=" * 90)
    print(f"GPU: {gpu_name}")
    print(f"CuPy {cp.__version__} | Torch {torch.__version__} | NumPy {np.__version__}")
    print(f"losses={losses}")
    print(f"penalties={penalties}")
    print(f"devices={devices}")
    print(f"solvers={solvers}")
    print(
        f"cv_strategy={args.cv_strategy} "
        f"acknowledge_approx={args.acknowledge_approx} "
        f"refine_top_k={args.refine_top_k}"
    )
    print(f"sizes={sizes} n_alphas={args.n_alphas} cv={args.cv} repeat={args.repeat} warmup={args.warmup}")

    records = []
    for n, p in sizes:
        for loss in losses:
            print(f"\n{'=' * 90}")
            print(f"  LOSS: {loss} | n={n}, p={p}")
            print(f"{'=' * 90}")
            X_np, y_np = make_data(loss, n, p, args.seed)
            X_by_device = {"cpu": X_np, "auto": X_np}
            y_by_device = {"cpu": y_np, "auto": y_np}
            if "cuda" in devices and has_cupy:
                X_by_device["cuda"] = cp.asarray(X_np)
                y_by_device["cuda"] = cp.asarray(y_np)
            if "torch" in devices and has_torch:
                X_by_device["torch"] = torch.tensor(X_np, dtype=torch.float64, device="cuda")
                y_by_device["torch"] = torch.tensor(y_np, dtype=torch.float64, device="cuda")

            for penalty in penalties:
                for solver in solvers:
                    row = {
                        "loss": loss,
                        "penalty": penalty,
                        "solver": solver,
                        "n": n,
                        "p": p,
                        "devices": {},
                    }
                    cpu_coef = None
                    cpu_alpha = None
                    for device in devices:
                        if device not in X_by_device:
                            row["devices"][device] = {"status": "SKIP"}
                            continue
                        try:
                            def _fit(device=device, solver=solver):
                                return PenalizedGLM_CV(
                                    loss=loss,
                                    penalty=penalty,
                                    n_alphas=args.n_alphas,
                                    l1_ratio=0.5,
                                    cv=args.cv,
                                    device=device,
                                    max_iter=args.max_iter,
                                    tol=args.tol,
                                    solver=solver,
                                    cv_strategy=args.cv_strategy,
                                    acknowledge_approx=args.acknowledge_approx,
                                    refine_top_k=args.refine_top_k,
                                ).fit(X_by_device[device], y_by_device[device])

                            fitted, elapsed = bench(_fit, warmup=args.warmup, repeat=args.repeat)
                            coef = _to_numpy(fitted.coef_)
                            alpha = float(fitted.alpha_)
                            info = {
                                "status": "OK",
                                "alpha": alpha,
                                "time_ms": elapsed * 1000.0,
                                "selected_device": str(getattr(fitted, "_cv_selected_device_", device)),
                                "cv_strategy": getattr(fitted, "cv_strategy_", args.cv_strategy),
                                "refined_count": int(np.sum(fitted.cv_results_.get("refined_mask", []))),
                                "auto_reason": getattr(fitted, "_cv_auto_reason_", None),
                            }
                            if device == "cpu":
                                cpu_coef = coef
                                cpu_alpha = alpha
                                info.update({"corr_vs_cpu": 1.0, "l2_vs_cpu": 0.0, "alpha_match": True})
                            elif cpu_coef is not None:
                                info.update({
                                    "corr_vs_cpu": coef_corr(cpu_coef, coef),
                                    "l2_vs_cpu": coef_l2(cpu_coef, coef),
                                    "alpha_match": abs(cpu_alpha - alpha) / max(abs(cpu_alpha), 1e-10) < 0.1,
                                })
                            row["devices"][device] = info
                        except Exception as exc:
                            row["devices"][device] = {"status": "ERROR", "error": str(exc)}

                    records.append(row)
                    np_i = row["devices"].get("cpu", {})
                    cu_i = row["devices"].get("cuda", {})
                    to_i = row["devices"].get("torch", {})
                    auto_i = row["devices"].get("auto", {})

                    def _part(info, label):
                        if info.get("status") != "OK":
                            return f"{label}=--"
                        selected = info.get("selected_device")
                        suffix = f" sel={selected}" if label == "auto" and selected else ""
                        return f"{label}={info['time_ms']:.0f}ms a={info['alpha']:.4g}{suffix}"

                    alpha_state = "NA"
                    compare_infos = [
                        info for key, info in row["devices"].items()
                        if key != "cpu" and info.get("status") == "OK"
                    ]
                    if compare_infos:
                        alpha_ok = all(
                            info.get("alpha_match", True)
                            for info in compare_infos
                        )
                        alpha_state = "OK" if alpha_ok else "DIFF"
                    parts = [_part(np_i, "cpu")]
                    if "cuda" in row["devices"]:
                        parts.append(_part(cu_i, "cu"))
                    if "torch" in row["devices"]:
                        parts.append(_part(to_i, "to"))
                    if "auto" in row["devices"]:
                        parts.append(_part(auto_i, "auto"))
                    label = f"{penalty}/{solver}"
                    print(
                        f"  {label:22s} "
                        f"{' '.join(parts)} "
                        f"alpha={alpha_state} "
                        f"L2_cu={cu_i.get('l2_vs_cpu', float('nan')):.2e} "
                        f"L2_to={to_i.get('l2_vs_cpu', float('nan')):.2e}"
                    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON: {out}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
