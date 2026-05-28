"""Stage 1: Smoke test — verify code runs without errors."""
import sys, time, warnings, ssl
sys.path.insert(0, '/root/statgpu')
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np

def result(name, **kwargs):
    print(f'=== BENCH_RESULT ===')
    print(f'name: {name}')
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print(f'=== END ===')

# ── 1.1 freMTPL2 data download ──
print('\n[1.1] freMTPL2 data download...')
try:
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    result('fremtpl2_download', status='ok', shape=str(df.shape))
except Exception as e:
    result('fremtpl2_download', status='FAIL', error=str(e))
    sys.exit(1)

# ── 1.2 PoissonRegression CPU (small subset) ──
print('\n[1.2] PoissonRegression CPU...')
try:
    from statgpu.linear_model import PoissonRegression
    np.random.seed(42)
    n, p = 1000, 10
    X = np.random.randn(n, p)
    beta = np.random.randn(p) * 0.3
    mu = np.exp(np.clip(X @ beta, -5, 5))
    y = np.random.poisson(mu).astype(np.float64)
    model = PoissonRegression(max_iter=50)
    model.fit(X, y)
    print(f'  coef[:3]: {model.coef_[:3]}')
    result('poisson_cpu', status='ok', coef_norm=str(np.linalg.norm(model.coef_)))
except Exception as e:
    result('poisson_cpu', status='FAIL', error=str(e))
    sys.exit(1)

# ── 1.3 PoissonRegression CuPy (small subset) ──
print('\n[1.3] PoissonRegression CuPy...')
try:
    model_gpu = PoissonRegression(max_iter=50, device='cuda')
    model_gpu.fit(X, y)
    print(f'  coef[:3]: {model_gpu.coef_[:3]}')
    result('poisson_cupy', status='ok', coef_norm=str(np.linalg.norm(model_gpu.coef_)))
except Exception as e:
    result('poisson_cupy', status='FAIL', error=str(e))
    sys.exit(1)

# ── 1.4 adjust_pvalues CuPy ──
print('\n[1.4] adjust_pvalues CuPy...')
try:
    from statgpu.inference import adjust_pvalues
    pvals = np.random.uniform(0, 1, 10000).astype(np.float64)
    rej, adj = adjust_pvalues(pvals, method='bh', backend='cupy')
    adj_np = adj.get() if hasattr(adj, 'get') else adj
    rej_np = rej.get() if hasattr(rej, 'get') else rej
    has_nan = bool(np.any(np.isnan(adj_np)))
    print(f'  reject: {rej_np.sum()}, has_nan: {has_nan}')
    result('adj_pval_cupy', status='ok' if not has_nan else 'FAIL',
           reject=str(rej_np.sum()), has_nan=str(has_nan))
    if has_nan:
        sys.exit(1)
except Exception as e:
    result('adj_pval_cupy', status='FAIL', error=str(e))
    sys.exit(1)

print('\n' + '='*50)
print('Stage 1 PASSED: All smoke tests OK')
print('='*50)
print('__STAGE_DONE__')
