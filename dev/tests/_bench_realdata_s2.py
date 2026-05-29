"""Stage 2: Precision validation — GPU results match reference frameworks."""
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

def coef_diff(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    max_abs = float(np.max(np.abs(a - b)))
    max_rel = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-15)))
    corr = float(np.corrcoef(a, b)[0, 1])
    return max_abs, max_rel, corr

# ── 2.1 Poisson GLM vs sklearn (subset) ──
print('\n[2.1] Poisson GLM vs sklearn (n=10K)...')
from sklearn.datasets import fetch_openml
df = fetch_openml(data_id=41214, as_frame=True, parser='auto').frame

# Preprocess
y_freq = df['ClaimNb'].values.astype(np.float64)
feature_cols = [c for c in df.columns if c not in ['IDpol', 'ClaimNb', 'ClaimAmount', 'Exposure']]
X_raw = df[feature_cols].copy()
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
X_processed = __import__('pandas').get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=np.float64)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_num = scaler.fit_transform(X_processed[numeric_cols])
X_processed[numeric_cols] = X_num
X_full = X_processed.values.astype(np.float64)

# Subset
n_sub = min(10000, len(X_full))
idx = np.random.choice(len(X_full), n_sub, replace=False)
X_sub = X_full[idx]
y_sub = y_freq[idx]

# sklearn reference (no regularization to match statgpu)
from sklearn.linear_model import PoissonRegressor
sk_model = PoissonRegressor(alpha=0, max_iter=200)
sk_model.fit(X_sub, y_sub)
sk_coef = sk_model.coef_

# statgpu CPU
from statgpu.linear_model import PoissonRegression
cpu_model = PoissonRegression(max_iter=200)
cpu_model.fit(X_sub, y_sub)
cpu_coef = cpu_model.coef_

# statgpu CuPy
gpu_model = PoissonRegression(max_iter=200, device='cuda')
gpu_model.fit(X_sub, y_sub)
gpu_coef = gpu_model.coef_

# Precision
ma_cpu, mr_cpu, cr_cpu = coef_diff(cpu_coef, sk_coef)
ma_gpu, mr_gpu, cr_gpu = coef_diff(gpu_coef, sk_coef)

print(f'  CPU vs sklearn:  corr={cr_cpu:.6f}, max_abs={ma_cpu:.2e}')
print(f'  CuPy vs sklearn: corr={cr_gpu:.6f}, max_abs={ma_gpu:.2e}')
result('poisson_cpu_sklearn', coef_corr=f'{cr_cpu:.6f}', max_abs_diff=f'{ma_cpu:.2e}')
result('poisson_cupy_sklearn', coef_corr=f'{cr_gpu:.6f}', max_abs_diff=f'{ma_gpu:.2e}')

# ── 2.2 Gamma GLM vs sklearn (synthetic, since freMTPL2 has no ClaimAmount) ──
print('\n[2.2] Gamma GLM vs sklearn (synthetic)...')
np.random.seed(123)
n_g, p_g = 5000, 15
X_g = np.random.randn(n_g, p_g)
beta_g = np.random.randn(p_g) * 0.3
mu_g = np.exp(X_g @ beta_g)
y_g = np.random.gamma(2, mu_g / 2)  # Gamma with shape=2

from sklearn.linear_model import GammaRegressor
sk_g = GammaRegressor(alpha=0, max_iter=200)
sk_g.fit(X_g, y_g)

from statgpu.linear_model import GammaRegression
cpu_g = GammaRegression(max_iter=200)
cpu_g.fit(X_g, y_g)
gpu_g = GammaRegression(max_iter=200, device='cuda')
gpu_g.fit(X_g, y_g)

ma_g_cpu, mr_g_cpu, cr_g_cpu = coef_diff(cpu_g.coef_, sk_g.coef_)
ma_g_gpu, mr_g_gpu, cr_g_gpu = coef_diff(gpu_g.coef_, sk_g.coef_)
print(f'  CPU vs sklearn:  corr={cr_g_cpu:.6f}, max_abs={ma_g_cpu:.2e}')
print(f'  CuPy vs sklearn: corr={cr_g_gpu:.6f}, max_abs={ma_g_gpu:.2e}')
result('gamma_cpu_sklearn', coef_corr=f'{cr_g_cpu:.6f}', max_abs_diff=f'{ma_g_cpu:.2e}')
result('gamma_cupy_sklearn', coef_corr=f'{cr_g_gpu:.6f}', max_abs_diff=f'{ma_g_gpu:.2e}')

# ── 2.3 adjust_pvalues vs statsmodels (100K) ──
print('\n[2.3] adjust_pvalues vs statsmodels (100K)...')
from statgpu.inference import adjust_pvalues
import statsmodels.stats.multitest as smm

np.random.seed(42)
pvals = np.random.uniform(0, 1, 100000).astype(np.float64)

rej_sm, adj_sm, _, _ = smm.multipletests(pvals, method='fdr_bh')
rej_gpu, adj_gpu = adjust_pvalues(pvals, method='bh', backend='cupy')
adj_gpu_np = adj_gpu.get() if hasattr(adj_gpu, 'get') else adj_gpu
rej_gpu_np = rej_gpu.get() if hasattr(rej_gpu, 'get') else rej_gpu

agree = float(np.mean(rej_sm == rej_gpu_np))
max_diff = float(np.max(np.abs(adj_sm - adj_gpu_np)))
print(f'  reject agreement: {agree*100:.2f}%')
print(f'  adj_pval max_diff: {max_diff:.2e}')
result('adj_pval_100k', reject_agreement=f'{agree:.6f}', max_abs_diff=f'{max_diff:.2e}')

# ── 2.4 CoxPH vs lifelines (synthetic small) ──
print('\n[2.4] CoxPH vs lifelines (n=500, p=20)...')
from statgpu.survival import CoxPH
from lifelines import CoxPHFitter

np.random.seed(42)
n_cox, p_cox = 500, 20
X_cox = np.random.randn(n_cox, p_cox)
true_beta = np.random.randn(p_cox) * 0.3
lp = X_cox @ true_beta
st = np.random.exponential(1 / (0.01 * np.exp(np.clip(lp, -5, 5))))
ct = np.random.uniform(0, np.percentile(st, 80), n_cox)
e = (st <= ct).astype(np.float64)
t = np.minimum(st, ct)

# statgpu CPU
cox_cpu = CoxPH(max_iter=100)
cox_cpu.fit(X_cox, t, e)

# statgpu CuPy
cox_gpu = CoxPH(max_iter=100, device='cuda')
cox_gpu.fit(X_cox, t, e)

# lifelines reference
import pandas as pd
df_cox = pd.DataFrame(X_cox, columns=[f'x{i}' for i in range(p_cox)])
df_cox['duration'] = t
df_cox['event'] = e
cph = CoxPHFitter()
cph.fit(df_cox, duration_col='duration', event_col='event')
ll_coef = cph.params_.values

ma_cpu_c, mr_cpu_c, cr_cpu_c = coef_diff(cox_cpu.coef_, ll_coef)
ma_gpu_c, mr_gpu_c, cr_gpu_c = coef_diff(cox_gpu.coef_, ll_coef)

print(f'  CPU vs lifelines:  corr={cr_cpu_c:.6f}, max_abs={ma_cpu_c:.2e}')
print(f'  CuPy vs lifelines: corr={cr_gpu_c:.6f}, max_abs={ma_gpu_c:.2e}')
result('coxph_cpu_lifelines', coef_corr=f'{cr_cpu_c:.6f}', max_abs_diff=f'{ma_cpu_c:.2e}')
result('coxph_cupy_lifelines', coef_corr=f'{cr_gpu_c:.6f}', max_abs_diff=f'{ma_gpu_c:.2e}')

# Summary
print('\n' + '='*50)
all_pass = (
    cr_cpu > 0.98 and cr_gpu > 0.98 and
    cr_g_cpu > 0.98 and cr_g_gpu > 0.98 and
    agree > 0.99 and cr_gpu_c > 0.99
)
print(f'Stage 2 {"PASSED" if all_pass else "FAILED"}')
print('='*50)
if not all_pass:
    sys.exit(1)
print('__STAGE_DONE__')
