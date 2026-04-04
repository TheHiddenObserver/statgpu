"""
Validate statgpu CoxPH against R's survival::coxph().
"""

import numpy as np
import subprocess
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Validation: statgpu CoxPH vs R survival::coxph()")
print("=" * 80)

# Generate survival data with fixed seed
np.random.seed(42)
n_samples = 200
n_features = 3

# Generate covariates
X = np.random.randn(n_samples, n_features)

# True coefficients
true_coef = np.array([0.5, -0.3, 0.8])

# Generate survival times (exponential distribution with hazard exp(X @ coef))
hazard = np.exp(X @ true_coef)
# Add some censoring
survival_time = np.random.exponential(1.0 / hazard)
censoring_time = np.random.exponential(2.0, size=n_samples)  # Mean censoring time

# Observed time is minimum of survival and censoring
time = np.minimum(survival_time, censoring_time)
event = (survival_time <= censoring_time).astype(int)

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Number of events: {np.sum(event)}")
print(f"Censoring rate: {1 - np.mean(event):.2%}")

# Save data for R
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data_file = f.name
    f.write("time,event," + ",".join([f"x{i+1}" for i in range(n_features)]) + "\n")
    for i in range(n_samples):
        row = [time[i], event[i]] + list(X[i])
        f.write(",".join([str(v) for v in row]) + "\n")

# ============================================
# 1. statgpu CoxPH
# ============================================
print("\n" + "=" * 80)
print("statgpu CoxPH (Breslow)")
print("=" * 80)

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from statgpu.survival import CoxPH
from statgpu._config import set_device

set_device('cpu')
model = CoxPH(ties='breslow', device='cpu')
model.fit(X, time, event)
model.summary()

# Store results for comparison
statgpu_coef = model.coef_.copy()
statgpu_hr = model.hazard_ratios_.copy()
statgpu_se = model._bse.copy()
statgpu_z = model._zvalues.copy()
statgpu_p = model._pvalues.copy()
statgpu_cindex = model._cindex
statgpu_ll = model._log_likelihood
statgpu_lr = model._lr_test_stat
statgpu_wald = model._wald_test_stat
statgpu_score = model._score_test_stat

# ============================================
# 2. R survival::coxph()
# ============================================
print("\n" + "=" * 80)
print("R survival::coxph() output")
print("=" * 80)

r_script = f'''
library(survival)
data <- read.csv("{data_file}")
fit <- coxph(Surv(time, event) ~ x1 + x2 + x3, data=data, ties="breslow")
summary(fit)

# Extract values for comparison
cat("\n===EXTRACTED_VALUES===\n")
cat("coef:", coef(fit), "\\n")
cat("exp_coef:", exp(coef(fit)), "\\n")
cat("se:", sqrt(diag(vcov(fit))), "\\n")
cat("z:", coef(fit) / sqrt(diag(vcov(fit))), "\\n")
cat("p:", summary(fit)$coefficients[,5], "\\n")
cat("loglik:", fit$loglik[2], "\\n")
cat("loglik_null:", fit$loglik[1], "\\n")
cat("lr_test:", 2*(fit$loglik[2] - fit$loglik[1]), "\\n")
cat("wald_test:", summary(fit)$waldtest[1], "\\n")
cat("score_test:", summary(fit)$sctest[1], "\\n")
cat("concordance:", summary(fit)$concordance[1], "\\n")
'''

r_results = {}
try:
    result = subprocess.run(
        ['Rscript', '-e', r_script],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr)
    
    # Parse extracted values
    in_extract = False
    for line in result.stdout.split('\n'):
        if '===EXTRACTED_VALUES===' in line:
            in_extract = True
            continue
        if in_extract and ':' in line:
            key, val = line.split(':', 1)
            try:
                r_results[key] = float(val.strip())
            except:
                r_results[key] = [float(x) for x in val.strip().split()]
except Exception as e:
    print(f"Error running R: {e}")

# Cleanup
os.unlink(data_file)

# ============================================
# 3. Comparison
# ============================================
print("\n" + "=" * 80)
print("Detailed Comparison: statgpu vs R")
print("=" * 80)

if r_results:
    print(f"\n{'Metric':<25} {'R':>15} {'statgpu':>15} {'Diff':>15}")
    print("-" * 75)
    
    comparisons = [
        ('x1 coef', r_results.get('coef', [0,0,0])[0] if isinstance(r_results.get('coef'), list) else 0, statgpu_coef[0]),
        ('x2 coef', r_results.get('coef', [0,0,0])[1] if isinstance(r_results.get('coef'), list) else 0, statgpu_coef[1]),
        ('x3 coef', r_results.get('coef', [0,0,0])[2] if isinstance(r_results.get('coef'), list) else 0, statgpu_coef[2]),
        ('x1 HR', r_results.get('exp_coef', [0,0,0])[0] if isinstance(r_results.get('exp_coef'), list) else 0, statgpu_hr[0]),
        ('x2 HR', r_results.get('exp_coef', [0,0,0])[1] if isinstance(r_results.get('exp_coef'), list) else 0, statgpu_hr[1]),
        ('x3 HR', r_results.get('exp_coef', [0,0,0])[2] if isinstance(r_results.get('exp_coef'), list) else 0, statgpu_hr[2]),
        ('x1 se', r_results.get('se', [0,0,0])[0] if isinstance(r_results.get('se'), list) else 0, statgpu_se[0]),
        ('x2 se', r_results.get('se', [0,0,0])[1] if isinstance(r_results.get('se'), list) else 0, statgpu_se[1]),
        ('x3 se', r_results.get('se', [0,0,0])[2] if isinstance(r_results.get('se'), list) else 0, statgpu_se[2]),
        ('Log-likelihood', r_results.get('loglik', 0), statgpu_ll),
        ('LR test', r_results.get('lr_test', 0), statgpu_lr),
        ('Wald test', r_results.get('wald_test', 0), statgpu_wald),
        ('Score test', r_results.get('score_test', 0), statgpu_score),
        ('Concordance', r_results.get('concordance', 0), statgpu_cindex),
    ]
    
    max_diff = 0
    for name, r_val, sg_val in comparisons:
        diff = abs(r_val - sg_val)
        max_diff = max(max_diff, diff)
        print(f"{name:<25} {r_val:>15.6f} {sg_val:>15.6f} {diff:>15.2e}")
    
    print("\n" + "=" * 80)
    if max_diff < 0.01:
        print("✓ VALIDATION PASSED: All differences < 0.01")
    elif max_diff < 0.1:
        print("⚠ VALIDATION WARNING: Some differences < 0.1 (acceptable)")
    else:
        print("✗ VALIDATION FAILED: Large differences detected")
    print(f"Maximum difference: {max_diff:.6f}")
    print("=" * 80)
else:
    print("Could not compare with R (R not available or error occurred)")
    print("statgpu results:")
    print(f"  Coefficients: {statgpu_coef}")
    print(f"  Hazard Ratios: {statgpu_hr}")
    print(f"  Standard Errors: {statgpu_se}")
    print(f"  Concordance: {statgpu_cindex:.4f}")
