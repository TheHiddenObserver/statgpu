"""Compare statgpu vs external frameworks: precision + performance (v2).

Fixes:
- All 3 backends (numpy, cupy, torch)
- Matched GAM parameters (same lam, n_splines, degree)
- Panel: compare with same intercept handling
"""
import json, time, sys, warnings
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

results = {}

# ===========================================================================
# Panel Data: statgpu vs linearmodels (3 backends)
# ===========================================================================
print("=== Panel Data: statgpu vs linearmodels ===", flush=True)

from statgpu.panel import PanelOLS, RandomEffects, PooledOLS
import linearmodels.panel as lmp
import pandas as pd

def make_panel(n_entities, n_times, n_vars, seed=42):
    rng = np.random.RandomState(seed)
    n = n_entities * n_times
    X = rng.randn(n, n_vars)
    beta = rng.randn(n_vars) * 0.5
    eids = np.repeat(np.arange(n_entities), n_times)
    tids = np.tile(np.arange(n_times), n_entities)
    alpha_i = rng.randn(n_entities) * 0.3
    y = X @ beta + alpha_i[eids] + rng.randn(n) * 0.5
    return X, y, eids, tids

scales = {
    "medium": (500, 20, 10),
    "large": (2000, 50, 20),
}

for scale_name, (n_ent, n_time, n_var) in scales.items():
    X, y, eids, tids = make_panel(n_ent, n_time, n_var)
    n = n_ent * n_time
    print(f"\n  Panel {scale_name} ({n} obs):", flush=True)

    # linearmodels reference
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_var)])
    df['y'] = y
    df['entity'] = eids
    df['time'] = tids
    df = df.set_index(['entity', 'time'])

    # PanelOLS
    lm_model = lmp.PanelOLS(df['y'], df.drop(columns=['y']), entity_effects=True)
    lm_res = lm_model.fit()
    lm_coef = lm_res.params.values[:n_var]
    t0 = time.perf_counter()
    _ = lm_model.fit()
    lm_time = time.perf_counter() - t0

    for backend, device in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            t0 = time.perf_counter()
            m = PanelOLS(entity_effects=True, device=device)
            m.fit(X, y, entity_ids=eids)
            sg_time = time.perf_counter() - t0
            sg_coef = m.coef_[:n_var]
            l2_diff = float(np.linalg.norm(sg_coef - lm_coef))
            rel_diff = l2_diff / max(float(np.linalg.norm(lm_coef)), 1e-12)

            key = f"panel_{scale_name}_PanelOLS_{backend}"
            results[key] = {"statgpu_time": sg_time, "external_time": lm_time,
                           "speedup": lm_time / sg_time, "coef_rel_diff": rel_diff}
            print(f"    PanelOLS {backend}: {sg_time:.4f}s (lm={lm_time:.4f}s) spd={lm_time/sg_time:.1f}x rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    PanelOLS {backend}: FAIL - {e}", flush=True)

    # RandomEffects
    lm_model = lmp.RandomEffects(df['y'], df.drop(columns=['y']))
    lm_res = lm_model.fit()
    lm_coef = lm_res.params.values[:n_var]
    t0 = time.perf_counter()
    _ = lm_model.fit()
    lm_time = time.perf_counter() - t0

    for backend, device in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            t0 = time.perf_counter()
            m = RandomEffects(device=device)
            m.fit(X, y, entity_ids=eids)
            sg_time = time.perf_counter() - t0
            sg_coef = m.coef_[:n_var]
            l2_diff = float(np.linalg.norm(sg_coef - lm_coef))
            rel_diff = l2_diff / max(float(np.linalg.norm(lm_coef)), 1e-12)

            key = f"panel_{scale_name}_RE_{backend}"
            results[key] = {"statgpu_time": sg_time, "external_time": lm_time,
                           "speedup": lm_time / sg_time, "coef_rel_diff": rel_diff}
            print(f"    RE {backend}: {sg_time:.4f}s (lm={lm_time:.4f}s) spd={lm_time/sg_time:.1f}x rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    RE {backend}: FAIL - {e}", flush=True)

# ===========================================================================
# GAM: statgpu vs pygam (matched params, 3 backends)
# ===========================================================================
print("\n=== GAM: statgpu vs pygam (matched params) ===", flush=True)

from statgpu.semiparametric import GAM
from pygam import LinearGAM, s

def make_gam(n, nf, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, nf)
    y = np.sin(X[:, 0] * 2) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

gam_scales = {
    "small": (1000, 3),
    "medium": (10000, 5),
    "large": (100000, 10),
}

for scale_name, (n, nf) in gam_scales.items():
    X, y = make_gam(n, nf)
    print(f"\n  GAM {scale_name} ({n} obs, {nf} feat):", flush=True)

    # pygam reference (fixed lam for fair comparison)
    LAM = 0.6  # pygam default
    N_SPLINES = 20
    DEGREE = 3

    terms = s(0, n_splines=N_SPLINES, spline_order=DEGREE)
    for j in range(1, nf):
        terms = terms + s(j, n_splines=N_SPLINES, spline_order=DEGREE)

    pg = LinearGAM(terms, lam=LAM).fit(X, y)
    pg_pred = pg.predict(X)
    t0 = time.perf_counter()
    _ = LinearGAM(terms, lam=LAM).fit(X, y)
    pg_time = time.perf_counter() - t0

    # statgpu with same lam
    for backend, device in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            t0 = time.perf_counter()
            gam = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=LAM, device=device)
            gam.fit(X, y)
            sg_time = time.perf_counter() - t0
            sg_pred = gam.predict(X)

            pred_diff = float(np.linalg.norm(sg_pred - pg_pred))
            pred_ref = float(np.linalg.norm(pg_pred))
            rel_diff = pred_diff / max(pred_ref, 1e-12)

            key = f"gam_{scale_name}_{backend}"
            results[key] = {"statgpu_time": sg_time, "external_time": pg_time,
                           "speedup": pg_time / sg_time, "pred_rel_diff": rel_diff}
            print(f"    {backend}: {sg_time:.4f}s (pygam={pg_time:.4f}s) spd={pg_time/sg_time:.1f}x pred_rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    {backend}: FAIL - {e}", flush=True)

    # Also test with auto lam (GCV) for both
    print(f"    --- Auto GCV comparison ---", flush=True)
    pg_auto = LinearGAM(terms).gridsearch(X, y, progress=False)
    pg_auto_pred = pg_auto.predict(X)
    pg_auto_lam = pg_auto.lam

    for backend, device in [("numpy", "cpu"), ("torch", "cuda")]:
        try:
            sg_auto = GAM(n_splines=N_SPLINES, degree=DEGREE, lam=None, device=device)
            sg_auto.fit(X, y)
            sg_auto_pred = sg_auto.predict(X)
            rel_diff = float(np.linalg.norm(sg_auto_pred - pg_auto_pred)) / max(float(np.linalg.norm(pg_auto_pred)), 1e-12)
            print(f"    {backend} auto: statgpu_lam={sg_auto.lam_:.4f}, pygam_lam={pg_auto_lam}, pred_rel={rel_diff:.2e}", flush=True)
        except Exception as e:
            print(f"    {backend} auto: FAIL - {e}", flush=True)

# ===========================================================================
# ANOVA: statgpu vs scipy (3 backends)
# ===========================================================================
print("\n=== ANOVA: statgpu vs scipy ===", flush=True)

from statgpu.anova import f_oneway, f_twoway, f_welch
from scipy import stats as sp_stats

anova_scales = {
    "small": (100, 5),
    "medium": (10000, 10),
    "large": (100000, 20),
}

for scale_name, (npg, ng) in anova_scales.items():
    rng = np.random.RandomState(42)
    groups = [rng.randn(npg) + i * 0.5 for i in range(ng)]
    print(f"\n  ANOVA {scale_name} ({npg}/group, {ng} groups):", flush=True)

    # f_oneway
    scipy_f, scipy_p = sp_stats.f_oneway(*groups)
    t0 = time.perf_counter()
    _ = sp_stats.f_oneway(*groups)
    scipy_time = time.perf_counter() - t0

    for backend in ["numpy", "cupy", "torch"]:
        try:
            t0 = time.perf_counter()
            sg = f_oneway(*groups, backend=backend)
            sg_time = time.perf_counter() - t0
            f_rel = abs(sg.statistic - scipy_f) / max(abs(scipy_f), 1e-12)

            key = f"anova_{scale_name}_f_oneway_{backend}"
            results[key] = {"statgpu_time": sg_time, "external_time": scipy_time,
                           "speedup": scipy_time / sg_time, "f_rel_diff": f_rel}
            print(f"    f_oneway {backend}: {sg_time*1000:.2f}ms (scipy={scipy_time*1000:.2f}ms) spd={scipy_time/sg_time:.1f}x F_rel={f_rel:.2e}", flush=True)
        except Exception as e:
            print(f"    f_oneway {backend}: FAIL - {e}", flush=True)

    # f_twoway
    data = [[rng.randn(npg) + i * 0.3 + j * 0.2 for j in range(4)] for i in range(3)]

    for backend in ["numpy", "cupy", "torch"]:
        try:
            t0 = time.perf_counter()
            sg = f_twoway(data, interaction=True, backend=backend)
            sg_time = time.perf_counter() - t0

            key = f"anova_{scale_name}_f_twoway_{backend}"
            results[key] = {"statgpu_time": sg_time, "f_a": sg.factor_a_statistic}
            print(f"    f_twoway {backend}: {sg_time*1000:.2f}ms F_A={sg.factor_a_statistic:.4f}", flush=True)
        except Exception as e:
            print(f"    f_twoway {backend}: FAIL - {e}", flush=True)

# ===========================================================================
# Save
# ===========================================================================
with open("/root/statgpu/results_compare_external_v2.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nTotal: {len(results)} results")
print("DONE")
