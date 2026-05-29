"""Stage 3: Full single-module benchmark with timing (freMTPL2 full + adjust_pvalues 1M)."""
import sys, time, warnings, ssl, gc
sys.path.insert(0, '/root/statgpu')
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import cupy as cp

def result(name, **kwargs):
    print(f'=== BENCH_RESULT ===')
    print(f'name: {name}')
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print(f'=== END ===')

def cleanup():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

def coef_diff(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    return float(np.max(np.abs(a - b))), float(np.corrcoef(a, b)[0, 1])

def timed_fit(model, X, y, **kwargs):
    cleanup()
    t0 = time.perf_counter()
    model.fit(X, y, **kwargs)
    if hasattr(cp, 'cuda'): cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) * 1000

# ── Load freMTPL2 ──
print('\n[Data] Loading freMTPL2...')
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
y_freq = df['ClaimNb'].values.astype(np.float64)
exposure = df['Exposure'].values.astype(np.float64)
feature_cols = [c for c in df.columns if c not in ['IDpol', 'ClaimNb', 'ClaimAmount', 'Exposure']]
X_raw = df[feature_cols].copy()
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
X_processed = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=np.float64)
scaler = StandardScaler()
X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
X = X_processed.values.astype(np.float64)
print(f'  Shape: {X.shape}')

# ── 3.1 Poisson GLM full ──
print('\n[3.1] Poisson GLM full (n=678K)...')
from statgpu.linear_model import PoissonRegression
from sklearn.linear_model import PoissonRegressor

# sklearn (use low tol for fair comparison — default tol=1e-4 doesn't converge on this dataset)
cleanup()
t0 = time.perf_counter()
sk_p = PoissonRegressor(alpha=0, max_iter=200, tol=1e-8)
sk_p.fit(X, y_freq)
t_sk = (time.perf_counter() - t0) * 1000

# statgpu CPU
cleanup()
t0 = time.perf_counter()
cpu_p = PoissonRegression(max_iter=200, C=1e10, tol=1e-8)
cpu_p.fit(X, y_freq)
t_cpu = (time.perf_counter() - t0) * 1000

# statgpu CuPy
cleanup()
t0 = time.perf_counter()
gpu_p = PoissonRegression(max_iter=200, C=1e10, tol=1e-8, device='cuda')
gpu_p.fit(X, y_freq)
cp.cuda.Stream.null.synchronize()
t_gpu = (time.perf_counter() - t0) * 1000

ma_cpu, cr_cpu = coef_diff(cpu_p.coef_, sk_p.coef_)
ma_gpu, cr_gpu = coef_diff(gpu_p.coef_, sk_p.coef_)
speedup = t_sk / t_gpu if t_gpu > 0 else 0

print(f'  sklearn: {t_sk:.0f}ms, CPU: {t_cpu:.0f}ms, CuPy: {t_gpu:.0f}ms')
print(f'  CPU vs sklearn: corr={cr_cpu:.6f}, max_abs={ma_cpu:.2e}')
print(f'  CuPy vs sklearn: corr={cr_gpu:.6f}, max_abs={ma_gpu:.2e}')
print(f'  Speedup (CuPy vs sklearn): {speedup:.2f}x')
result('poisson_full', time_ms=f'{t_gpu:.1f}', coef_corr=f'{cr_gpu:.6f}',
       max_abs_diff=f'{ma_gpu:.2e}', speedup=f'{speedup:.2f}')

# ── 3.2 Gamma GLM full (synthetic) ──
print('\n[3.2] Gamma GLM full (synthetic n=678K)...')
np.random.seed(123)
n_g_full = len(X)
beta_g_full = np.random.randn(X.shape[1]) * 0.1
mu_g_full = np.exp(np.clip(X @ beta_g_full, -5, 5))
y_gamma = np.random.gamma(2, mu_g_full / 2)

from statgpu.linear_model import GammaRegression
from sklearn.linear_model import GammaRegressor

cleanup()
t0 = time.perf_counter()
sk_g = GammaRegressor(alpha=0, max_iter=200, tol=1e-8)
sk_g.fit(X, y_gamma)
t_sk_g = (time.perf_counter() - t0) * 1000

cleanup()
t0 = time.perf_counter()
gpu_g = GammaRegression(max_iter=200, C=1e10, tol=1e-8, device='cuda')
gpu_g.fit(X, y_gamma)
cp.cuda.Stream.null.synchronize()
t_gpu_g = (time.perf_counter() - t0) * 1000

ma_g, cr_g = coef_diff(gpu_g.coef_, sk_g.coef_)
speedup_g = t_sk_g / t_gpu_g if t_gpu_g > 0 else 0

print(f'  sklearn: {t_sk_g:.0f}ms, CuPy: {t_gpu_g:.0f}ms')
print(f'  CuPy vs sklearn: corr={cr_g:.6f}, max_abs={ma_g:.2e}')
print(f'  Speedup: {speedup_g:.2f}x')
result('gamma_full', time_ms=f'{t_gpu_g:.1f}', coef_corr=f'{cr_g:.6f}',
       max_abs_diff=f'{ma_g:.2e}', speedup=f'{speedup_g:.2f}')

# ── 3.3 adjust_pvalues 1M ──
print('\n[3.3] adjust_pvalues 1M...')
from statgpu.inference import adjust_pvalues
import statsmodels.stats.multitest as smm

np.random.seed(42)
pvals_1m = np.random.uniform(0, 1, 1_000_000).astype(np.float64)

cleanup()
t0 = time.perf_counter()
rej_sm, adj_sm, _, _ = smm.multipletests(pvals_1m, method='fdr_bh')
t_sm = (time.perf_counter() - t0) * 1000

cleanup()
t0 = time.perf_counter()
rej_gpu, adj_gpu = adjust_pvalues(pvals_1m, method='bh', backend='cupy')
cp.cuda.Stream.null.synchronize()
t_adj_gpu = (time.perf_counter() - t0) * 1000

adj_gpu_np = adj_gpu.get() if hasattr(adj_gpu, 'get') else adj_gpu
rej_gpu_np = rej_gpu.get() if hasattr(rej_gpu, 'get') else rej_gpu
agree = float(np.mean(rej_sm == rej_gpu_np))
max_diff_adj = float(np.max(np.abs(adj_sm - adj_gpu_np)))
speedup_adj = t_sm / t_adj_gpu if t_adj_gpu > 0 else 0

print(f'  statsmodels: {t_sm:.0f}ms, CuPy: {t_adj_gpu:.0f}ms')
print(f'  reject agreement: {agree*100:.2f}%')
print(f'  Speedup: {speedup_adj:.2f}x')
result('adj_pval_1m', time_ms=f'{t_adj_gpu:.1f}', reject_agreement=f'{agree:.6f}',
       max_abs_diff=f'{max_diff_adj:.2e}', speedup=f'{speedup_adj:.2f}')

# Summary
print('\n' + '='*50)
all_pass = cr_gpu > 0.98 and cr_g > 0.98 and agree > 0.99
print(f'Stage 3 {"PASSED" if all_pass else "FAILED"}')
print('='*50)
if not all_pass:
    sys.exit(1)
print('__STAGE_DONE__')
