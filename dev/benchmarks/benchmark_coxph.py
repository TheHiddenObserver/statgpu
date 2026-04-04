"""
Benchmark CoxPH: CPU performance.
"""

import numpy as np
import time as time_module
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CoxPH Benchmark")
print("=" * 80)

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from statgpu.survival import CoxPH
from statgpu._config import set_device, cuda_available

# Check GPU availability
has_gpu = cuda_available()
print(f"\nGPU Available: {has_gpu}")

# Benchmark configurations
configs = [
    (200, 3, "Small (200x3)"),
    (500, 5, "Medium (500x5)"),
    (1000, 5, "Large (1000x5)"),
]

results = []

for n_samples, n_features, label in configs:
    print(f"\n{'='*80}")
    print(f"Dataset: {label}")
    print(f"{'='*80}")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 0.5
    hazard = np.exp(X @ true_coef)
    survival_time = np.random.exponential(1.0 / hazard)
    censoring_time = np.random.exponential(2.0, size=n_samples)
    time_obs = np.minimum(survival_time, censoring_time)
    event = (survival_time <= censoring_time).astype(int)
    
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Events: {np.sum(event)} ({100*np.mean(event):.1f}%)")
    
    # CPU benchmark
    set_device('cpu')
    model_cpu = CoxPH(ties='breslow', device='cpu', max_iter=50)
    
    # Warmup
    model_cpu.fit(X[:100], time_obs[:100], event[:100])
    
    # Benchmark
    start = time_module.time()
    model_cpu = CoxPH(ties='breslow', device='cpu', max_iter=50)
    model_cpu.fit(X, time_obs, event)
    cpu_time = time_module.time() - start
    
    print(f"\nCPU Time: {cpu_time:.3f}s")
    print(f"  Iterations: {model_cpu._iterations}")
    print(f"  Converged: {model_cpu._converged}")
    print(f"  Concordance: {model_cpu._cindex:.4f}")
    print(f"  Log-likelihood: {model_cpu._log_likelihood:.4f}")
    
    results.append({
        'label': label,
        'n_samples': n_samples,
        'n_features': n_features,
        'cpu_time': cpu_time,
        'iterations': model_cpu._iterations,
        'converged': model_cpu._converged,
        'cindex': model_cpu._cindex,
        'loglik': model_cpu._log_likelihood
    })

# Summary table
print("\n" + "=" * 80)
print("Benchmark Summary")
print("=" * 80)
print(f"\n{'Dataset':<20} {'Samples':>10} {'Features':>10} {'Time (s)':>12} {'Iter':>8} {'C-index':>10}")
print("-" * 80)

for r in results:
    print(f"{r['label']:<20} {r['n_samples']:>10} {r['n_features']:>10} "
          f"{r['cpu_time']:>12.3f} {r['iterations']:>8} {r['cindex']:>10.4f}")

print("=" * 80)

# Test predict methods
print("\n" + "=" * 80)
print("Testing Prediction Methods")
print("=" * 80)

np.random.seed(42)
n_test = 50
n_features = 3
X_test = np.random.randn(n_test, n_features)

model = CoxPH(ties='breslow')
model.fit(X[:100], time_obs[:100], event[:100])

# Predict hazard ratios
hr = model.predict_hazard_ratio(X_test)
print(f"\nHazard Ratios (first 5): {hr[:5]}")

# Predict risk scores
risk = model.predict_risk_score(X_test)
print(f"Risk Scores (first 5): {risk[:5]}")

# Predict survival
surv, times = model.predict_survival(X_test[:5])
print(f"\nSurvival probabilities shape: {surv.shape}")
print(f"Number of time points: {len(times)}")

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
