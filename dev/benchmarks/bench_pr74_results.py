"""PR #74 inference validation benchmark — generates results/pr74_inference_validation.json.

Covers: Ordered (logit/probit), Sandwich, Oracle, Bootstrap, QuantileRegression.
All backends: NumPy (CPU), CuPy, Torch.  Warmup: 1 untimed fit before timing.
"""
import numpy as np, time, json, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)
OUT = os.path.join(PROJECT, "results", "pr74_inference_validation.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CFG = [(500, 5), (2000, 10)]  # (n, p) pairs
SEED = 42
WARMUP = True   # 1 untimed warmup before timing
TIMED_RUNS = 3  # median of 3 timed runs
K = 3           # ordered categories

def sync(dev):
    if dev == "cuda": __import__('cupy').cuda.Stream.null.synchronize()
    elif dev == "torch": __import__('torch').cuda.synchronize()

def bench_one(model_fn, backends, n, p, **kw):
    """model_fn(device_str) -> dict with keys: bse0, nll (optional), extra"""
    results = {}
    for device, label in backends:
        times = []
        for run in range(TIMED_RUNS + (1 if WARMUP else 0)):
            t0 = time.perf_counter()
            r = model_fn(device, n, p, **kw)
            sync(device)
            t = time.perf_counter() - t0
            if WARMUP and run == 0:
                continue  # warmup, skip
            times.append(t)
        r["time"] = float(np.median(times))
        r["n"] = n; r["p"] = p
        results[label] = r
    return results

# ---- Ordered Logit ----
def ordered_logit(device, n, p):
    from statgpu.linear_model._ordered_logit import OrderedLogitRegression
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = np.digitize(0.5 + X @ np.linspace(0.5, -0.3, p) + 0.5 * np.random.randn(n),
                     np.linspace(-1, 1, K - 1))
    m = OrderedLogitRegression(n_categories=K, compute_inference=True, max_iter=50, device=device)
    m.fit(X, y)
    from scipy.stats import norm as _norm
    wald = float(m._zvalues[0]**2) if m._zvalues is not None else np.nan
    pval = float(2*_norm.sf(abs(float(m._zvalues[0])))) if m._zvalues is not None else np.nan
    return {"bse0": float(m._bse[0]), "nll": float(m.loglikelihood/n),
            "wald_stat": wald, "wald_pval": pval,
            "iter": int(m.n_iter_), "ok": True}

# ---- Ordered Probit ----
def ordered_probit(device, n, p):
    from statgpu.linear_model._ordered_probit import OrderedProbitRegression
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = np.digitize(0.5 + X @ np.linspace(0.5, -0.3, p) + 0.5 * np.random.randn(n),
                     np.linspace(-1, 1, K - 1))
    m = OrderedProbitRegression(n_categories=K, compute_inference=True, max_iter=50, device=device)
    m.fit(X, y)
    return {"bse0": float(m._bse[0]), "nll": float(m.loglikelihood/n),
            "iter": int(m.n_iter_), "ok": True}

# ---- Sandwich (logistic + L2) ----
def sandwich(device, n, p):
    from statgpu.linear_model.penalized._penalized_logistic import PenalizedLogisticRegression
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = (1.0/(1+np.exp(-(X@np.linspace(0.3,-0.2,p)+0.3*np.random.randn(n))))>0.5).astype(int)
    m = PenalizedLogisticRegression(penalty="l2", alpha=0.01, compute_inference=True,
                                     cov_type="hc0", max_iter=200, device=device)
    m.fit(X, y)
    return {"bse0": float(m._bse[0]) if m._bse is not None else np.nan, "ok": True}

# ---- Oracle (SCAD + logistic) ----
def oracle(device, n, p):
    from statgpu.linear_model.penalized._penalized_logistic import PenalizedLogisticRegression
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = (1.0/(1+np.exp(-(X@np.array([1.0,-0.5,0.8]+[0]*(p-3))+0.3*np.random.randn(n))))>0.5).astype(int)
    try:
        m = PenalizedLogisticRegression(penalty="scad", alpha=0.1, compute_inference=True,
                                         inference_method="oracle", max_iter=200, tol=1e-4, device=device)
        m.fit(X, y)
        return {"bse0": float(m._bse[0]) if m._bse is not None else np.nan, "ok": True}
    except Exception as e:
        return {"bse0": np.nan, "ok": False, "error": str(e)[:120]}

# ---- Bootstrap (Lasso) ----
def bootstrap(device, n, p):
    from statgpu.linear_model import Lasso
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = X @ np.array([1.0,-0.5,0.3]+[0]*(p-3)) + 0.5*np.random.randn(n)
    try:
        m = Lasso(alpha=0.05, compute_inference=True, inference_method="bootstrap",
                  n_bootstrap=50, max_iter=200, tol=1e-4, device=device)
        m.fit(X, y)
        return {"bse0": float(m._bse[0]) if m._bse is not None else np.nan, "ok": True}
    except Exception as e:
        return {"bse0": np.nan, "ok": False, "error": str(e)[:120]}

# ---- QuantileRegression (CPU-only inference) ----
def quantile_reg(device, n, p, method):
    from statgpu.linear_model import QuantileRegression
    np.random.seed(SEED)
    X = np.random.randn(n, p)
    y = 1.0 + X @ np.linspace(0.5,-0.3,p) + 0.5*np.random.randn(n)
    try:
        t0 = time.perf_counter()
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method=method, n_bootstrap=50, device=device)
        m.fit(X, y)
        t = time.perf_counter() - t0
        return {"bse0": float(m._bse[0]), "time": t, "ok": True}
    except NotImplementedError as e:
        return {"bse0": np.nan, "time": 0, "ok": True, "note": "CPU-only as expected"}
    except Exception as e:
        return {"bse0": np.nan, "ok": False, "error": str(e)[:120]}

# ---- Run ----
all_results = {}
backends_ordered = [("cpu","NumPy"),("cuda","CuPy"),("torch","Torch")]
backends_bootstrap = [("cpu","NumPy"),("cuda","CuPy"),("torch","Torch")]

for n, p in CFG:
    print(f"\n=== n={n} p={p} ===")
    for name, fn, bks in [
        ("ordered_logit", ordered_logit, backends_ordered),
        ("ordered_probit", ordered_probit, backends_ordered),
        ("sandwich", sandwich, backends_ordered),
        ("oracle", oracle, backends_ordered),
        ("bootstrap", bootstrap, backends_bootstrap),
    ]:
        key = f"{name}_n{n}_p{p}"
        all_results[key] = bench_one(fn, bks, n, p)
        for lbl, r in all_results[key].items():
            ok = r.get("ok", True)
            print(f"  {name:20s} {lbl:6s} bse0={r.get('bse0',np.nan):.6f} time={r.get('time',np.nan):.4f}s {'OK' if ok else 'FAIL:'+r.get('error','?')}")

    # QuantileRegression (CPU-only inference, GPU raises NotImplementedError)
    for method in ["kernel", "bootstrap"]:
        qr_results = {}
        for device, label in backends_ordered:
            r = quantile_reg(device, n, p, method)
            qr_results[label] = r
            note = r.get("note", "")
            ok = r.get("ok", True)
            print(f"  quantile_{method:9s} {label:6s} bse0={r.get('bse0',np.nan):.6f} time={r.get('time',np.nan):.4f}s {'OK' if ok else 'FAIL'} {note}")
        all_results[f"quantile_{method}_n{n}_p{p}"] = qr_results

with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults written to {OUT}")
