"""GPU inference precision + timing benchmark — sandwich, oracle, bootstrap, ordered."""
import numpy as np, time, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)


def bench_ordered(n=500, p=5, K=3):
    """Ordered logit inference: CPU vs CuPy vs Torch."""
    from statgpu.linear_model._ordered_logit import OrderedLogitRegression
    np.random.seed(42)
    X = np.random.randn(n, p)
    y = np.digitize(0.5 + X @ np.linspace(0.5, -0.3, p) + 0.5 * np.random.randn(n),
                     np.linspace(-1, 1, K - 1))
    results = {}
    for device, label in [("cpu", "NumPy"), ("cuda", "CuPy"), ("torch", "Torch")]:
        t0 = time.perf_counter()
        m = OrderedLogitRegression(n_categories=K, compute_inference=True, max_iter=50,
                                    device=device)
        m.fit(X, y)
        if device == "cuda":
            import cupy as cp; cp.cuda.Stream.null.synchronize()
        elif device == "torch":
            import torch; torch.cuda.synchronize()
        t = time.perf_counter() - t0
        results[label] = {"nll": m.loglikelihood/n, "bse0": float(m._bse[0]),
                          "iter": m.n_iter_, "time": t}
    return results


def bench_sandwich(n=500, p=5):
    """Penalized logistic + L2 sandwich: CPU vs CuPy vs Torch."""
    from statgpu.linear_model.penalized._penalized_logistic import PenalizedLogisticRegression
    np.random.seed(42)
    X = np.random.randn(n, p)
    y = (1.0 / (1 + np.exp(-(X @ np.linspace(0.3, -0.2, p) + 0.3 * np.random.randn(n)))) > 0.5).astype(int)
    results = {}
    for device, label in [("cpu", "NumPy"), ("cuda", "CuPy"), ("torch", "Torch")]:
        t0 = time.perf_counter()
        m = PenalizedLogisticRegression(penalty="l2", alpha=0.01,
                                         compute_inference=True, cov_type="hc0",
                                         max_iter=200, device=device)
        m.fit(X, y)
        if device == "cuda":
            import cupy as cp; cp.cuda.Stream.null.synchronize()
        elif device == "torch":
            import torch; torch.cuda.synchronize()
        t = time.perf_counter() - t0
        results[label] = {"bse0": float(m._bse[0]) if m._bse is not None else np.nan,
                          "time": t}
    return results


def bench_oracle(n=500, p=10):
    """SCAD + oracle inference: CPU vs CuPy vs Torch (logistic loss)."""
    from statgpu.linear_model.penalized._penalized_logistic import PenalizedLogisticRegression
    np.random.seed(42)
    X = np.random.randn(n, p)
    y = (1.0 / (1 + np.exp(-(X @ np.array([1.0, -0.5, 0.8] + [0]*(p-3)) + 0.3*np.random.randn(n)))) > 0.5).astype(int)
    results = {}
    for device, label in [("cpu", "NumPy"), ("cuda", "CuPy"), ("torch", "Torch")]:
        try:
            t0 = time.perf_counter()
            m = PenalizedLogisticRegression(penalty="scad", alpha=0.1,
                                             compute_inference=True,
                                             inference_method="oracle",
                                             max_iter=200, tol=1e-4, device=device)
            m.fit(X, y)
            if device == "cuda":
                import cupy as cp; cp.cuda.Stream.null.synchronize()
            elif device == "torch":
                import torch; torch.cuda.synchronize()
            t = time.perf_counter() - t0
            results[label] = {"bse0": float(m._bse[0]) if m._bse is not None else np.nan,
                              "time": t, "ok": True}
        except Exception as e:
            results[label] = {"time": np.nan, "ok": False, "error": str(e)[:100]}
    return results


def bench_bootstrap(n=500, p=10):
    """Lasso + bootstrap inference: CPU vs CuPy vs Torch (serial)."""
    from statgpu.linear_model import Lasso
    np.random.seed(42)
    X = np.random.randn(n, p)
    beta = np.zeros(p); beta[:3] = [1.0, -0.5, 0.8]
    y = X @ beta + 0.5 * np.random.randn(n)
    results = {}
    for device, label in [("cpu", "NumPy"), ("cuda", "CuPy"), ("torch", "Torch")]:
        try:
            t0 = time.perf_counter()
            m = Lasso(alpha=0.05, compute_inference=True,
                       inference_method="bootstrap", n_bootstrap=50,
                       max_iter=200, tol=1e-4, device=device)
            m.fit(X, y)
            if device == "cuda":
                import cupy as cp; cp.cuda.Stream.null.synchronize()
            elif device == "torch":
                import torch; torch.cuda.synchronize()
            t = time.perf_counter() - t0
            results[label] = {"bse0": float(m._bse[0]) if m._bse is not None else np.nan,
                              "time": t, "ok": True}
        except Exception as e:
            results[label] = {"time": np.nan, "ok": False, "error": str(e)[:100]}
    return results


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    p = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    for name, fn, args in [
        ("Ordered", bench_ordered, (n, p, 3)),
        ("Sandwich", bench_sandwich, (n, p)),
        ("Oracle", bench_oracle, (n, 10)),
        ("Bootstrap", bench_bootstrap, (n, 10)),
    ]:
        if arg != "all" and name.lower() not in arg.lower():
            continue
        print(f"\n=== {name} (n={args[0]}, p={args[1]}) ===")
        try:
            res = fn(*args)
            for label, r in res.items():
                ok = r.get("ok", True)
                status = "OK" if ok else f"FAIL: {r.get('error','?')}"
                bse = r.get("bse0", np.nan)
                nll = r.get("nll", np.nan)
                t = r.get("time", np.nan)
                extra = f" NLL={nll:.6f}" if not np.isnan(nll) else ""
                print(f"  {label:6s}: time={t:.4f}s  bse0={bse:.6f}{extra}  {status}")
            # Cross-backend comparison
            cpu_time = res.get("NumPy", {}).get("time", np.nan)
            for label in ["CuPy", "Torch"]:
                if label in res and not np.isnan(res[label].get("time", np.nan)) and not np.isnan(cpu_time):
                    speedup = cpu_time / res[label]["time"]
                    print(f"  {label} vs NumPy: {speedup:.2f}x")
        except Exception as e:
            print(f"  ERROR: {e}")
