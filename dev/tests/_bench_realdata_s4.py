"""Stage 4: High-dimensional CoxPH — synthetic METABRIC-like (n=1900, p=500)."""
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

def coef_diff(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    return float(np.max(np.abs(a - b))), float(np.corrcoef(a, b)[0, 1])

# ── Generate synthetic survival data ──
print('\n[Data] Generating synthetic survival data (n=1900, p=500)...')
np.random.seed(42)
n_meta, p_meta = 1900, 500
X_meta = np.random.randn(n_meta, p_meta)
true_beta = np.zeros(p_meta)
true_beta[:20] = np.random.randn(20) * 0.2
lp = X_meta @ true_beta
duration = np.random.exponential(1 / (0.01 * np.exp(np.clip(lp, -5, 5))))
censor = np.random.uniform(0, np.percentile(duration, 80), n_meta)
event = (duration <= censor).astype(np.float64)
t_meta = np.minimum(duration, censor)

print(f'  X shape: {X_meta.shape}')
print(f'  Events: {event.sum():.0f}/{len(event)} ({event.mean()*100:.1f}%)')

# ── 4.1 lifelines reference (small subset for speed) ──
print('\n[4.1] lifelines reference (p=50 subset)...')
from lifelines import CoxPHFitter

X_ref = X_meta[:, :50]
df_cox = pd.DataFrame(X_ref, columns=[f'x{i}' for i in range(50)])
df_cox['duration'] = t_meta
df_cox['event'] = event

cleanup()
t0 = time.perf_counter()
cph = CoxPHFitter()
cph.fit(df_cox, duration_col='duration', event_col='event')
t_ll = (time.perf_counter() - t0) * 1000
ll_coef = cph.params_.values
cindex_ll = cph.concordance_index_
print(f'  lifelines: {t_ll:.0f}ms, C-index={cindex_ll:.4f}')

# ── 4.2 statgpu CoxPH — all backends on full p=500 ──
print('\n[4.2] statgpu CoxPH (p=500, all backends)...')
from statgpu.survival import CoxPH

# CPU
cleanup()
t0 = time.perf_counter()
cox_cpu = CoxPH(max_iter=100)
cox_cpu.fit(X_meta, t_meta, event)
t_cpu = (time.perf_counter() - t0) * 1000

# CuPy
cleanup()
t0 = time.perf_counter()
cox_gpu = CoxPH(max_iter=100, device='cuda')
cox_gpu.fit(X_meta, t_meta, event)
cp.cuda.Stream.null.synchronize()
t_gpu = (time.perf_counter() - t0) * 1000

# Torch
cleanup()
t0 = time.perf_counter()
cox_torch = CoxPH(max_iter=100, device='torch')
cox_torch.fit(X_meta, t_meta, event)
torch.cuda.synchronize()
t_torch = (time.perf_counter() - t0) * 1000

# Compare backends vs CPU
ma_gpu, cr_gpu = coef_diff(cox_gpu.coef_, cox_cpu.coef_)
ma_torch, cr_torch = coef_diff(cox_torch.coef_, cox_cpu.coef_)

# Compare with lifelines on shared features
ma_ll_cpu, cr_ll_cpu = coef_diff(cox_cpu.coef_[:50], ll_coef)
ma_ll_gpu, cr_ll_gpu = coef_diff(cox_gpu.coef_[:50], ll_coef)

print(f'  CPU:   {t_cpu:.0f}ms')
print(f'  CuPy:  {t_gpu:.0f}ms, corr vs CPU={cr_gpu:.6f}')
print(f'  Torch: {t_torch:.0f}ms, corr vs CPU={cr_torch:.6f}')
print(f'  CuPy vs lifelines (first 50): corr={cr_ll_gpu:.6f}')

# C-index
cindex_cpu = cox_cpu.score(X_meta, t_meta, event)
cindex_gpu = cox_gpu.score(X_meta, t_meta, event)
print(f'  C-index: lifelines={cindex_ll:.4f}, CPU={cindex_cpu:.4f}, CuPy={cindex_gpu:.4f}')

result('metabric_coxph_cpu', time_ms=f'{t_cpu:.1f}', cindex=f'{cindex_cpu:.4f}')
result('metabric_coxph_cupy', time_ms=f'{t_gpu:.1f}', coef_corr=f'{cr_gpu:.6f}',
       cindex=f'{cindex_gpu:.4f}')
result('metabric_coxph_torch', time_ms=f'{t_torch:.1f}', coef_corr=f'{cr_torch:.6f}')

# Summary
print('\n' + '='*50)
cindex_ok = abs(cindex_gpu - cindex_cpu) < 0.001
all_pass = cr_gpu > 0.99 and cr_torch > 0.99 and cindex_ok
print(f'  CuPy vs CPU corr: {cr_gpu:.6f}')
print(f'  Torch vs CPU corr: {cr_torch:.6f}')
print(f'  C-index diff (CuPy vs CPU): {abs(cindex_gpu - cindex_cpu):.6f}')
print(f'Stage 4 {"PASSED" if all_pass else "FAILED"}')
print('='*50)
if not all_pass:
    sys.exit(1)
print('__STAGE_DONE__')
