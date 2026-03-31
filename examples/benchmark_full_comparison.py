"""
Full benchmark: statgpu vs sklearn vs statsmodels vs R
Saves data to CSV for R comparison.
"""

import numpy as np
import time
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Add statgpu to path
sys.path.insert(0, '/root/.openclaw/workspace-coding/statgpu')

print("=" * 80)
print("StatGPU Full Benchmark: Python vs R")
print("=" * 80)

# Configuration
np.random.seed(42)
N_SAMPLES = 10000
N_FEATURES = 50
NOISE = 0.1

# Generate data
X = np.random.randn(N_SAMPLES, N_FEATURES)
true_coef = np.random.randn(N_FEATURES) * 2
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(N_SAMPLES) * NOISE
y_binary = (y > np.median(y)).astype(int)

print(f"\nDataset: {N_SAMPLES} samples × {N_FEATURES} features")
print(f"Data size: {X.nbytes / 1e6:.1f} MB")

# Save data to CSV for R
import pandas as pd
data = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(N_FEATURES)])
data['y'] = y
data['y_binary'] = y_binary
csv_path = '/tmp/statgpu_benchmark_data.csv'
data.to_csv(csv_path, index=False)
print(f"Data saved to: {csv_path}")

# Check GPU
from statgpu._config import cuda_available
has_gpu = cuda_available()
print(f"GPU available: {has_gpu}")
if has_gpu:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")

print()

# Import statgpu
from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu._config import set_device

# Results storage
results = {}

def benchmark_model(name, model_class, X, y, **kwargs):
    """Benchmark a model on CPU and GPU."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    res = {}
    
    # CPU
    print("\n--- statgpu CPU ---")
    set_device('cpu')
    model = model_class(device='cpu', **kwargs)
    t0 = time.perf_counter()
    model.fit(X, y)
    res['statgpu_cpu'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy() if hasattr(model, 'coef_') else None,
    }
    if hasattr(model, 'rsquared'):
        res['statgpu_cpu']['r2'] = model.rsquared
    if hasattr(model, 'n_iter_'):
        res['statgpu_cpu']['n_iter'] = model.n_iter_
    print(f"Time: {res['statgpu_cpu']['time']:.2f} ms")
    if 'r2' in res['statgpu_cpu']:
        print(f"R²: {res['statgpu_cpu']['r2']:.6f}")
    
    # GPU
    if has_gpu:
        print("\n--- statgpu GPU ---")
        set_device('cuda')
        model = model_class(device='cuda', **kwargs)
        t0 = time.perf_counter()
        model.fit(X, y)
        res['statgpu_gpu'] = {
            'time': (time.perf_counter() - t0) * 1000,
            'coef': model.coef_.copy() if hasattr(model, 'coef_') else None,
        }
        if hasattr(model, 'rsquared'):
            res['statgpu_gpu']['r2'] = model.rsquared
        print(f"Time: {res['statgpu_gpu']['time']:.2f} ms")
        if 'r2' in res['statgpu_gpu']:
            print(f"R²: {res['statgpu_gpu']['r2']:.6f}")
        speedup = res['statgpu_cpu']['time'] / res['statgpu_gpu']['time']
        print(f"Speedup: {speedup:.2f}x")
    
    # sklearn
    try:
        if name == "Linear Regression":
            from sklearn.linear_model import LinearRegression as SklearnModel
            sk_kwargs = {}
        elif name == "Ridge Regression":
            from sklearn.linear_model import Ridge as SklearnModel
            sk_kwargs = {'alpha': kwargs.get('alpha', 1.0)}
        elif name == "Lasso Regression":
            from sklearn.linear_model import Lasso as SklearnModel
            sk_kwargs = {'alpha': kwargs.get('alpha', 0.1), 'max_iter': kwargs.get('max_iter', 1000)}
        elif name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression as SklearnModel
            sk_kwargs = {'max_iter': kwargs.get('max_iter', 100)}
        else:
            raise ImportError
            
        print(f"\n--- sklearn ---")
        model = SklearnModel(**sk_kwargs)
        t0 = time.perf_counter()
        model.fit(X, y)
        res['sklearn'] = {
            'time': (time.perf_counter() - t0) * 1000,
            'coef': model.coef_.copy() if hasattr(model, 'coef_') else None,
        }
        if hasattr(model, 'score'):
            res['sklearn']['r2'] = model.score(X, y)
        print(f"Time: {res['sklearn']['time']:.2f} ms")
        if 'r2' in res['sklearn']:
            print(f"R²/Accuracy: {res['sklearn']['r2']:.6f}")
    except Exception as e:
        print(f"sklearn error: {e}")
    
    # statsmodels
    try:
        import statsmodels.api as sm
        print(f"\n--- statsmodels ---")
        
        if name == "Linear Regression":
            X_const = sm.add_constant(X)
            t0 = time.perf_counter()
            model = sm.OLS(y, X_const).fit()
            res['statsmodels'] = {
                'time': (time.perf_counter() - t0) * 1000,
                'coef': np.array(model.params[1:]),
                'r2': model.rsquared,
            }
        elif name == "Logistic Regression":
            X_const = sm.add_constant(X)
            t0 = time.perf_counter()
            model = sm.Logit(y, X_const).fit(disp=0)
            res['statsmodels'] = {
                'time': (time.perf_counter() - t0) * 1000,
                'coef': np.array(model.params[1:]),
            }
        else:
            raise NotImplementedError(f"{name} not in statsmodels")
            
        print(f"Time: {res['statsmodels']['time']:.2f} ms")
        if 'r2' in res['statsmodels']:
            print(f"R²: {res['statsmodels']['r2']:.6f}")
    except Exception as e:
        print(f"statsmodels error: {e}")
    
    # Accuracy comparison
    print(f"\n--- Accuracy (vs sklearn) ---")
    if 'sklearn' in res and res['sklearn']['coef'] is not None:
        for lib, data in res.items():
            if lib != 'sklearn' and data.get('coef') is not None:
                coef_diff = np.max(np.abs(data['coef'] - res['sklearn']['coef']))
                print(f"{lib:20s}: max coef diff = {coef_diff:.2e}")
    
    return res

# Run benchmarks
results['LinearRegression'] = benchmark_model(
    "Linear Regression", LinearRegression, X, y
)

results['Ridge'] = benchmark_model(
    "Ridge Regression", Ridge, X, y, alpha=1.0
)

results['Lasso'] = benchmark_model(
    "Lasso Regression", Lasso, X, y, alpha=0.1, max_iter=1000
)

results['Logistic'] = benchmark_model(
    "Logistic Regression", LogisticRegression, X, y_binary, max_iter=100
)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Model':<20} {'statgpu CPU':<15} {'statgpu GPU':<15} {'sklearn':<15} {'statsmodels':<15}")
print("-" * 80)

for model_name, res in results.items():
    sg_cpu = f"{res.get('statgpu_cpu', {}).get('time', 0):.1f}ms"
    sg_gpu = f"{res.get('statgpu_gpu', {}).get('time', 0):.1f}ms" if 'statgpu_gpu' in res else "N/A"
    sk = f"{res.get('sklearn', {}).get('time', 0):.1f}ms" if 'sklearn' in res else "N/A"
    sm = f"{res.get('statsmodels', {}).get('time', 0):.1f}ms" if 'statsmodels' in res else "N/A"
    print(f"{model_name:<20} {sg_cpu:<15} {sg_gpu:<15} {sk:<15} {sm:<15}")

print("\n" + "=" * 80)
print("R Comparison")
print("=" * 80)
print(f"""
Data saved to: {csv_path}

Run this R script to compare:

```r
# Read data
data <- read.csv("{csv_path}")
X <- data[, 1:{N_FEATURES}]
y <- data$y
y_binary <- data$y_binary

# Linear Regression
lm_model <- lm(y ~ ., data=data)
summary(lm_model)

# Logistic
logit_model <- glm(y_binary ~ ., data=data, family=binomial)
summary(logit_model)
```
""")

print("=" * 80)
print("Benchmark complete!")
print("=" * 80)