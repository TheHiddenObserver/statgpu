"""Compare statgpu vs pygam with aligned parameters."""
import json, time, sys
sys.path.insert(0, '/root/statgpu')
import numpy as np

try:
    import torch
    Xw = torch.randn(100, 10, device='cuda')
    _ = torch.mm(Xw, Xw.t())
    torch.cuda.synchronize()
    print("GPU warmup done", flush=True)
except:
    pass

from statgpu.semiparametric import GAM
from pygam import LinearGAM, s

def make_gam(n, nf, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, nf)
    y = np.sin(X[:, 0] * 2) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

N_SPLINES = 20
DEGREE = 3

results = {}

# Test 1: Same fixed lam, uniform knots, gamma=1.4
print("=== Test 1: Fixed lam, uniform knots, gamma=1.4 ===", flush=True)
LAM = 1.0

for n, nf, label in [(1000, 3, "small"), (10000, 5, "medium"), (100000, 10, "large")]:
    X, y = make_gam(n, nf)
    print(f"\n  {label} ({n} obs, {nf} feat):", flush=True)

    # pygam with fixed lam
    terms = s(0, n_splines=N_SPLINES, spline_order=DEGREE)
    for j in range(1, nf):
        terms = terms + s(j, n_splines=N_SPLINES, spline_order=DEGREE)
    pg = LinearGAM(terms, lam=LAM).fit(X, y)
    pg_pred = pg.predict(X)
    t0 = time.perf_counter()
    _ = LinearGAM(terms, lam=LAM).fit(X, y)
    pg_time = time.perf_counter() - t0

    # statgpu with same params (uniform knots, gamma=1.4)
    for backend, device in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            t0 = time.perf_counter()
            gam = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=LAM,
                      knot_method="uniform", gamma=1.4, device=device)
            gam.fit(X, y)
            sg_time = time.perf_counter() - t0
            sg_pred = gam.predict(X)

            pred_diff = float(np.linalg.norm(sg_pred - pg_pred))
            pred_ref = float(np.linalg.norm(pg_pred))
            rel_diff = pred_diff / max(pred_ref, 1e-12)

            key = f"gam_fixed_{label}_{backend}"
            results[key] = {"statgpu_time": sg_time, "pygam_time": pg_time,
                           "speedup": pg_time / sg_time, "pred_rel_diff": rel_diff}
            print(f"    {backend}: {sg_time:.4f}s (pygam={pg_time:.4f}s) spd={pg_time/sg_time:.1f}x pred_rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    {backend}: FAIL - {e}", flush=True)

# Test 2: Auto GCV, uniform knots, gamma=1.4
print("\n=== Test 2: Auto GCV, uniform knots, gamma=1.4 ===", flush=True)

for n, nf, label in [(1000, 3, "small"), (10000, 5, "medium"), (100000, 10, "large")]:
    X, y = make_gam(n, nf)
    print(f"\n  {label} ({n} obs, {nf} feat):", flush=True)

    # pygam with GCV
    terms = s(0, n_splines=N_SPLINES, spline_order=DEGREE)
    for j in range(1, nf):
        terms = terms + s(j, n_splines=N_SPLINES, spline_order=DEGREE)
    pg = LinearGAM(terms).gridsearch(X, y, progress=False)
    pg_pred = pg.predict(X)
    pg_lam = pg.lam
    t0 = time.perf_counter()
    _ = LinearGAM(terms).gridsearch(X, y, progress=False)
    pg_time = time.perf_counter() - t0

    # statgpu with same GCV settings (uniform knots, gamma=1.4)
    for backend, device in [("numpy", "cpu"), ("torch", "cuda")]:
        try:
            t0 = time.perf_counter()
            gam = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=None,
                      knot_method="uniform", gamma=1.4, device=device)
            gam.fit(X, y)
            sg_time = time.perf_counter() - t0
            sg_pred = gam.predict(X)

            pred_diff = float(np.linalg.norm(sg_pred - pg_pred))
            pred_ref = float(np.linalg.norm(pg_pred))
            rel_diff = pred_diff / max(pred_ref, 1e-12)

            print(f"    {backend}: lam={gam.lam_:.4f} (pygam={pg_lam}) pred_rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    {backend}: FAIL - {e}", flush=True)

# Test 3: Fixed lam, quantile knots (statgpu default) vs uniform (pygam)
print("\n=== Test 3: Fixed lam, quantile knots (statgpu) vs uniform (pygam) ===", flush=True)

for n, nf, label in [(100000, 10, "large")]:
    X, y = make_gam(n, nf)

    terms = s(0, n_splines=N_SPLINES, spline_order=DEGREE)
    for j in range(1, nf):
        terms = terms + s(j, n_splines=N_SPLINES, spline_order=DEGREE)
    pg = LinearGAM(terms, lam=LAM).fit(X, y)
    pg_pred = pg.predict(X)

    # statgpu quantile knots
    gam_q = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=LAM,
                knot_method="quantile", gamma=1.0, device="cpu")
    gam_q.fit(X, y)
    pred_q = gam_q.predict(X)
    diff_q = float(np.linalg.norm(pred_q - pg_pred)) / float(np.linalg.norm(pg_pred))

    # statgpu uniform knots
    gam_u = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=LAM,
                knot_method="uniform", gamma=1.0, device="cpu")
    gam_u.fit(X, y)
    pred_u = gam_u.predict(X)
    diff_u = float(np.linalg.norm(pred_u - pg_pred)) / float(np.linalg.norm(pg_pred))

    print(f"  quantile knots: pred_rel={diff_q:.2e}", flush=True)
    print(f"  uniform knots:  pred_rel={diff_u:.2e}", flush=True)

with open("/root/statgpu/results_gam_aligned.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nTotal: {len(results)} results")
print("DONE")
