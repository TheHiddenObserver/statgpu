"""
Compare all five statgpu methods with R.
"""

import numpy as np
import time
import warnings
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
warnings.filterwarnings('ignore')

print("=" * 80)
print("StatGPU vs R - All Five Methods Comparison")
print("=" * 80)

# Generate data
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features) * 2
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5
y_binary = (y > np.median(y)).astype(int)
time_surv = np.random.exponential(10, n_samples)
event_surv = np.random.binomial(1, 0.7, n_samples)

print(f"\nDataset: {n_samples} samples × {n_features} features")

# Save to CSV for R
import pandas as pd
data = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
data['y'] = y
data['y_binary'] = y_binary
data['time'] = time_surv
data['event'] = event_surv
csv_path = '/tmp/statgpu_all_methods.csv'
data.to_csv(csv_path, index=False)
print(f"Data saved to: {csv_path}")

# Import statgpu
from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu.survival import CoxPH
from statgpu._config import set_device

set_device('cpu')

results = {}

# 1. Linear Regression
print("\n" + "=" * 80)
print("1. LINEAR REGRESSION")
print("=" * 80)

model = LinearRegression(device='cpu')
t0 = time.perf_counter()
model.fit(X, y)
results['Linear'] = {
    'python_time': (time.perf_counter() - t0) * 1000,
    'python_r2': model.rsquared,
    'python_coef': model.coef_.copy()
}
print(f"Python: {results['Linear']['python_time']:.2f} ms, R²={results['Linear']['python_r2']:.6f}")

# 2. Ridge
print("\n" + "=" * 80)
print("2. RIDGE REGRESSION")
print("=" * 80)

model = Ridge(alpha=1.0, device='cpu')
t0 = time.perf_counter()
model.fit(X, y)
results['Ridge'] = {
    'python_time': (time.perf_counter() - t0) * 1000,
    'python_r2': model.rsquared
}
print(f"Python: {results['Ridge']['python_time']:.2f} ms, R²={results['Ridge']['python_r2']:.6f}")

# 3. Lasso
print("\n" + "=" * 80)
print("3. LASSO REGRESSION")
print("=" * 80)

model = Lasso(alpha=0.1, max_iter=1000, device='cpu')
t0 = time.perf_counter()
model.fit(X, y)
results['Lasso'] = {
    'python_time': (time.perf_counter() - t0) * 1000,
    'python_r2': model.rsquared,
    'python_nonzero': np.sum(np.abs(model.coef_) > 1e-10)
}
print(f"Python: {results['Lasso']['python_time']:.2f} ms, R²={results['Lasso']['python_r2']:.6f}")

# 4. Logistic
print("\n" + "=" * 80)
print("4. LOGISTIC REGRESSION")
print("=" * 80)

model = LogisticRegression(max_iter=100, device='cpu')
t0 = time.perf_counter()
model.fit(X, y_binary)
acc = np.mean(model.predict(X) == y_binary)
results['Logistic'] = {
    'python_time': (time.perf_counter() - t0) * 1000,
    'python_acc': acc
}
print(f"Python: {results['Logistic']['python_time']:.2f} ms, Acc={results['Logistic']['python_acc']:.4f}")

# 5. CoxPH
print("\n" + "=" * 80)
print("5. COX PROPORTIONAL HAZARDS")
print("=" * 80)

try:
    model = CoxPH(ties='breslow', max_iter=50, device='cpu')
    t0 = time.perf_counter()
    model.fit(X, time_surv, event_surv)
    results['CoxPH'] = {
        'python_time': (time.perf_counter() - t0) * 1000,
        'python_converged': model._converged
    }
    print(f"Python: {results['CoxPH']['python_time']:.2f} ms, Converged={results['CoxPH']['python_converged']}")
except Exception as e:
    print(f"Python Error: {e}")
    results['CoxPH'] = {'python_error': str(e)}

# Summary
print("\n" + "=" * 80)
print("PYTHON RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Method':<20} {'Time (ms)':<12} {'Metric':<15} {'Value':<15}")
print("-" * 65)

for method, data in results.items():
    if 'python_error' in data:
        print(f"{method:<20} {'ERROR':<12}")
    elif method == 'Linear':
        print(f"{method:<20} {data['python_time']:<12.2f} {'R²':<15} {data['python_r2']:<15.6f}")
    elif method == 'Ridge':
        print(f"{method:<20} {data['python_time']:<12.2f} {'R²':<15} {data['python_r2']:<15.6f}")
    elif method == 'Lasso':
        print(f"{method:<20} {data['python_time']:<12.2f} {'Non-zero':<15} {data['python_nonzero']}")
    elif method == 'Logistic':
        print(f"{method:<20} {data['python_time']:<12.2f} {'Accuracy':<15} {data['python_acc']:<15.4f}")
    elif method == 'CoxPH':
        status = '✓' if data.get('python_converged') else '✗'
        print(f"{method:<20} {data['python_time']:<12.2f} {'Converged':<15} {status}")

print("\n" + "=" * 80)
print("R COMPARISON")
print("=" * 80)

print(f"""
Run the following R commands:

```r
data <- read.csv("{csv_path}")
X <- data[, 1:10]

# 1. Linear Regression
system.time({{ lm_model <- lm(y ~ ., data=data) }})
summary(lm_model)$r.squared

# 2. Ridge (requires glmnet)
# install.packages("glmnet")
library(glmnet)
cv.glmnet(as.matrix(X), data$y, alpha=0)

# 3. Lasso
cv.glmnet(as.matrix(X), data$y, alpha=1)

# 4. Logistic
system.time({{ logit_model <- glm(y_binary ~ ., data=data, family=binomial) }})

# 5. Cox (requires survival)
library(survival)
coxph(Surv(time, event) ~ ., data=data)
```
""")

print("=" * 80)
