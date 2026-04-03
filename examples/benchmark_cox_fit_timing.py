"""
Cox PH 拟合耗时：以 **statsmodels PHReg** 为成熟实现基线，对比 statgpu GPU（及可选 statgpu CPU）。

- **主参考**：``statsmodels.duration.api.PHReg(..., ties=...).fit(disp=False)``  
  （statsmodels 在拟合阶段会完成估计与标准误等；与 ``statgpu`` 的 ``compute_inference=False``  
  并非完全同工作量，但反映常见「直接调 fit」的成本。）
- **对比**：statgpu ``device=cuda`` / 可选 ``device=cpu``。

示例::

    python examples/benchmark_cox_fit_timing.py --n 8000 --p 10 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.survival import CoxPH
from statgpu._config import set_device, cuda_available

try:
    import statsmodels.duration.api as smd

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def make_data(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.28, size=p)
    lin = X @ beta
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    t_obs = -np.log(u) / (0.04 * np.exp(np.clip(lin, -18, 18)))
    event = rng.binomial(1, 0.72, size=n).astype(np.int32)
    return X.astype(np.float64), t_obs.astype(np.float64), event


def time_statsmodels_phreg(t_obs, X, event, ties: str) -> tuple[float, np.ndarray]:
    """Wall time for statsmodels PHReg.fit; returns (ms, coef)."""
    t0 = time.perf_counter()
    res = smd.PHReg(t_obs, X, status=event, ties=ties).fit(disp=False)
    t1 = time.perf_counter()
    coef = np.asarray(res.params, dtype=np.float64).ravel()
    return (t1 - t0) * 1000.0, coef


def time_statgpu(device: str, ties: str, X, t_obs, event, max_iter: int, tol: float) -> tuple[float, np.ndarray]:
    set_device(device)
    model = CoxPH(
        device=device,
        ties=ties,
        max_iter=max_iter,
        tol=tol,
        compute_inference=False,
    )
    t0 = time.perf_counter()
    if device == "cuda":
        import cupy as cp

        model.fit(cp.asarray(X), cp.asarray(t_obs), cp.asarray(event))
        cp.cuda.Stream.null.synchronize()
    else:
        model.fit(X, t_obs, event)
    t1 = time.perf_counter()
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    return (t1 - t0) * 1000.0, coef


def parse_args():
    p = argparse.ArgumentParser(description="Cox PH: statsmodels PHReg vs statgpu (timing).")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--p", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--repeats", type=int, default=3, help="repeats per config (median reported)")
    p.add_argument(
        "--include-statgpu-cpu",
        action="store_true",
        help="also print statgpu CPU times (secondary; not the main baseline).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import cupy as cp

        _ = cp.asarray([1.0, 2.0]) @ cp.asarray([3.0, 4.0])
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

    has_gpu = False
    try:
        import cupy as cp

        has_gpu = cuda_available()
    except Exception:
        pass

    if has_gpu:
        import cupy as cp

        for ties in ("breslow", "efron"):
            Xw, tw, ew = make_data(min(args.n, 512), min(args.p, 6), args.seed)
            time_statgpu("cuda", ties, Xw, tw, ew, min(args.max_iter, 20), args.tol)
        cp.cuda.Stream.null.synchronize()

    print("=== Cox PH fit time (ms) — baseline: statsmodels PHReg ===")
    print(f"n={args.n}, p={args.p}, max_iter={args.max_iter}, repeats={args.repeats}")
    print(f"statsmodels available: {HAS_STATSMODELS}  |  CUDA (statgpu GPU): {has_gpu}")
    print()
    if not HAS_STATSMODELS:
        print("Install statsmodels to compare against PHReg:  pip install statsmodels")
        print()

    if has_gpu:
        hdr = f"{'ties':<10} {'sm_PHReg':>12} {'sg_gpu':>12} {'sm/gpu':>10} {'|coef|max':>12}"
    else:
        hdr = f"{'ties':<10} {'sm_PHReg':>12} {'sg_cpu':>12} {'sm/cpu':>10} {'|coef|max':>12}"
    if args.include_statgpu_cpu and has_gpu:
        hdr += f" {'sg_cpu':>12} {'sm/cpu':>10}"
    print(hdr)
    print("-" * len(hdr))

    for ties in ("breslow", "efron"):
        sm_times: list[float] = []
        gpu_times: list[float] = []
        cpu_sg_times: list[float] = []
        coef_diffs: list[float] = []

        for r in range(args.repeats):
            seed_r = args.seed + r * 17
            Xr, tr, er = make_data(args.n, args.p, seed_r)

            if HAS_STATSMODELS:
                ms_sm, coef_sm = time_statsmodels_phreg(tr, Xr, er, ties)
                sm_times.append(ms_sm)
            else:
                coef_sm = None
                sm_times.append(float("nan"))

            if has_gpu:
                ms_g, coef_g = time_statgpu("cuda", ties, Xr, tr, er, args.max_iter, args.tol)
                gpu_times.append(ms_g)
                ref = coef_sm
                cmpc = coef_g
            else:
                ms_c, coef_c = time_statgpu("cpu", ties, Xr, tr, er, args.max_iter, args.tol)
                cpu_sg_times.append(ms_c)
                ref = coef_sm
                cmpc = coef_c

            if ref is not None:
                coef_diffs.append(float(np.max(np.abs(cmpc - ref))))
            else:
                coef_diffs.append(float("nan"))

            if args.include_statgpu_cpu and has_gpu:
                ms_cpu, _ = time_statgpu("cpu", ties, Xr, tr, er, args.max_iter, args.tol)
                cpu_sg_times.append(ms_cpu)

        sm_med = float(np.nanmedian(sm_times)) if HAS_STATSMODELS else float("nan")
        dcoef = float(np.nanmedian(coef_diffs)) if coef_diffs else float("nan")

        if has_gpu:
            gpu_med = float(np.nanmedian(gpu_times))
            ratio = sm_med / gpu_med if HAS_STATSMODELS and np.isfinite(sm_med) and gpu_med > 0 else float("nan")
            row = f"{ties:<10} {sm_med:>12.2f} {gpu_med:>12.2f} {ratio:>10.2f}x {dcoef:>12.3e}"
            if args.include_statgpu_cpu:
                cpu_med = float(np.median(cpu_sg_times))
                cr = sm_med / cpu_med if np.isfinite(sm_med) and cpu_med > 0 else float("nan")
                row += f" {cpu_med:>12.2f} {cr:>10.2f}x"
            if HAS_STATSMODELS and np.isfinite(ratio) and ratio < 1.0:
                print(
                    f"  !! {ties}: statgpu GPU slower than statsmodels PHReg (sm/gpu={ratio:.2f} < 1).",
                    file=sys.stderr,
                )
        else:
            cpu_med = float(np.median(cpu_sg_times))
            ratio = sm_med / cpu_med if HAS_STATSMODELS and np.isfinite(sm_med) and cpu_med > 0 else float("nan")
            row = f"{ties:<10} {sm_med:>12.2f} {cpu_med:>12.2f} {ratio:>10.2f}x {dcoef:>12.3e}"
        print(row)

    print()
    if has_gpu:
        print(
            "Baseline: statsmodels PHReg.fit.  "
            "sm/gpu = sm_PHReg / sg_gpu (>1 ⇒ statgpu GPU faster than statsmodels)."
        )
        if not args.include_statgpu_cpu:
            print("Use --include-statgpu-cpu to also time statgpu CPU alongside GPU.")
    else:
        print(
            "Baseline: statsmodels PHReg.fit.  "
            "sm/cpu = sm_PHReg / sg_cpu (no CUDA here; statgpu CPU vs reference)."
        )
    print()
    print(
        "Note: statsmodels fit builds the usual results (incl. SE); statgpu uses compute_inference=False. "
        "|coef|max vs sm should be small."
    )
    if has_gpu:
        print(
            "Efron GPU path: CuPy group loop for log-lik + cached efron_pre; compare sm/gpu to see if GPU beats statsmodels."
        )


if __name__ == "__main__":
    main()
