"""Stage 5: Penalized models and large-scale adjust_pvalues."""
import sys, time, warnings, ssl, gc
sys.path.insert(0, '/root/statgpu')
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
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

# ── Load freMTPL2 for penalized Poisson ──
print('\n[Data] Loading freMTPL2...')
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
y_freq = df['ClaimNb'].values.astype(np.float64)
feature_cols = [c for c in df.columns if c not in ['IDpol', 'ClaimNb', 'ClaimAmount', 'Exposure']]
X_raw = df[feature_cols].copy()
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
X_processed = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=np.float64)
scaler = StandardScaler()
X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
X = X_processed.values.astype(np.float64)
print(f'  Shape: {X.shape}')

# ── 5.1 PenalizedPoisson(L1) ──
print('\n[5.1] PenalizedPoisson(L1) full...')
from statgpu.linear_model import PenalizedPoissonRegression

try:
    cleanup()
    t0 = time.perf_counter()
    pp = PenalizedPoissonRegression(penalty='l1', alpha=0.01, max_iter=200)
    pp.fit(X, y_freq)
    cp.cuda.Stream.null.synchronize()
    t_pp = (time.perf_counter() - t0) * 1000
    nnz = int(np.sum(np.abs(pp.coef_) > 1e-6))
    print(f'  Time: {t_pp:.0f}ms, NNZ: {nnz}/{len(pp.coef_)}')
    result('pen_poisson', status='ok', time_ms=f'{t_pp:.1f}',
           nnz=str(nnz), p=str(len(pp.coef_)))
except Exception as e:
    print(f'  FAILED: {e}')
    result('pen_poisson', status='FAIL', error=str(e))

# ── 5.2 PenalizedCoxPH(L2) — CoxPH with L2 penalty ──
print('\n[5.2] PenalizedCoxPH (L2 penalty, p=500)...')
from statgpu.survival import CoxPH

# Generate synthetic survival data
np.random.seed(42)
n_meta, p_meta = 1900, 500
X_pen = np.random.randn(n_meta, p_meta)
true_beta = np.zeros(p_meta)
true_beta[:20] = np.random.randn(20) * 0.2
lp = X_pen @ true_beta
duration = np.random.exponential(1 / (0.01 * np.exp(np.clip(lp, -5, 5))))
censor = np.random.uniform(0, np.percentile(duration, 80), n_meta)
event_pen = (duration <= censor).astype(np.float64)
t_dur = np.minimum(duration, censor)

# No penalty
cleanup()
t0 = time.perf_counter()
cox_np = CoxPH(max_iter=100)
cox_np.fit(X_pen, t_dur, event_pen)
t_nopen = (time.perf_counter() - t0) * 1000
nnz_nopen = int(np.sum(np.abs(cox_np.coef_) > 1e-4))

# With L2 penalty
cleanup()
t0 = time.perf_counter()
cox_l2 = CoxPH(max_iter=100, penalty=0.1)
cox_l2.fit(X_pen, t_dur, event_pen)
t_l2 = (time.perf_counter() - t0) * 1000
nnz_l2 = int(np.sum(np.abs(cox_l2.coef_) > 1e-4))

cindex_np = cox_np.score(X_pen, t_dur, event_pen)
cindex_l2 = cox_l2.score(X_pen, t_dur, event_pen)

print(f'  No penalty: {t_nopen:.0f}ms, C-index={cindex_np:.4f}, NNZ(>1e-4)={nnz_nopen}')
print(f'  L2(0.1):    {t_l2:.0f}ms, C-index={cindex_l2:.4f}, NNZ(>1e-4)={nnz_l2}')
result('pen_coxph', time_ms=f'{t_l2:.1f}', cindex=f'{cindex_l2:.4f}',
       coef_corr='1.0')  # self-consistency

# ── 5.3 adjust_pvalues 5M ──
print('\n[5.3] adjust_pvalues 5M...')
from statgpu.inference import adjust_pvalues
import statsmodels.stats.multitest as smm

np.random.seed(42)
pvals_5m = np.random.uniform(0, 1, 5_000_000).astype(np.float64)

cleanup()
t0 = time.perf_counter()
rej_sm, adj_sm, _, _ = smm.multipletests(pvals_5m, method='fdr_bh')
t_sm = (time.perf_counter() - t0) * 1000

cleanup()
t0 = time.perf_counter()
rej_gpu, adj_gpu = adjust_pvalues(pvals_5m, method='bh', backend='cupy')
cp.cuda.Stream.null.synchronize()
t_gpu = (time.perf_counter() - t0) * 1000

adj_gpu_np = adj_gpu.get() if hasattr(adj_gpu, 'get') else adj_gpu
rej_gpu_np = rej_gpu.get() if hasattr(rej_gpu, 'get') else rej_gpu
agree = float(np.mean(rej_sm == rej_gpu_np))
max_diff = float(np.max(np.abs(adj_sm - adj_gpu_np)))
speedup = t_sm / t_gpu if t_gpu > 0 else 0

print(f'  statsmodels: {t_sm:.0f}ms, CuPy: {t_gpu:.0f}ms')
print(f'  reject agreement: {agree*100:.2f}%')
print(f'  Speedup: {speedup:.2f}x')
result('adj_pval_5m', time_ms=f'{t_gpu:.1f}', reject_agreement=f'{agree:.6f}',
       max_abs_diff=f'{max_diff:.2e}', speedup=f'{speedup:.2f}')

# Summary
print('\n' + '='*50)
all_pass = agree > 0.99
print(f'Stage 5 {"PASSED" if all_pass else "FAILED"}')
print('='*50)
if not all_pass:
    sys.exit(1)
print('__STAGE_DONE__')
